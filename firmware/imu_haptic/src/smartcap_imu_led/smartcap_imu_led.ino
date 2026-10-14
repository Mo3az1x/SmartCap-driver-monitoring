/*
 * SmartCap - Team 1: IMU (MPU6050) + onboard LED (pin 13) version
 * Same as the original IMU+Haptic sketch, but the Arduino's own LED ("L", pin 13)
 * lights when the head is tilted DOWN (and also follows the vibration pattern).
 * No external parts needed on pin 9.
 *
 * Board: Arduino Uno / Nano.
 * Wiring: MPU6050 SDA->A4, SCL->A5, VCC->5V (GY-521 board), GND->GND, AD0->GND
 *
 * Serial monitor: 115200 baud, "Newline" line ending.
 * Serial IN shortcuts:
 *    0 1 2 3  test LED/vibration level for 3 s       D  local demo mode ON
 *    C        recalibrate (stay still ~5 s)           E  extended output ON
 *    S        spec-only output (default)              ?  print status
 *
 * Conventions:
 *   Chip X axis -> FORWARD (towards the brim), Z -> UP.
 *   pitch_deg > 0  : head tilting DOWN.   roll_deg > 0 : right ear down.
 *   If pitch comes out negative when you look down, set PITCH_SIGN to -1.0f.
 */
#include <Wire.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ======================= CONFIG =======================
#define DEVICE_ID        "smartcap_01"
#define MOTOR_PIN        9          // not used for the LED; leave unconnected
#define MOTOR_PWM_MAX    255
#define SAMPLE_HZ        50
#define SAMPLE_US        (1000000UL / SAMPLE_HZ)
#define OUTPUT_DIV       5          // one message every 5th sample (10 Hz) - keeps Serial from blocking the loop
#define SERIAL_BAUD      115200UL
#define MPU_ADDR         0x68       // AD0 -> GND
#define MPU_DLPF_CFG     0
#define PITCH_SIGN       1.0f
#define ROLL_SIGN        1.0f

// Calibration
#define CALIB_SAMPLES        250    // 5 s at 50 Hz
#define CALIB_MAX_ATTEMPTS   3
#define CALIB_MAX_GYRO_STD   3.0f
#define CALIB_MAX_ACC_STD    0.05f
#define CALIB_MAX_G_ERR      0.20f

// Filters
#define LPF_ALPHA        0.25f
#define COMP_ALPHA       0.98f
#define VEL_ALPHA        0.30f
#define ACC_ALPHA        0.20f

// Head-down (degrees relative to personal baseline) - the LED lights above ON, goes off below OFF
#define HEAD_DOWN_ON_DEG   15.0f
#define HEAD_DOWN_OFF_DEG  10.0f

// Nodding
#define NOD_ARM_DEG        6.0f
#define NOD_DEPTH_DEG      12.0f
#define NOD_RETURN_DEG     3.0f
#define NOD_MIN_MS         150
#define NOD_MAX_MS         2500
#define NOD_MIN_VEL_DPS    25.0f
#define NOD_DECAY_MS       1500.0f

#define WIN                25       // 0.5 s window

// Quality / status
#define I2C_FAIL_DISCONNECT   10
#define GYRO_SAT_DPS          450.0f
#define DEGRADED_BELOW        0.70f
#define INVALID_BELOW         0.30f
#define RECOVERY_DEGRADED_MS  3000UL

// Vibration
#define VIB_DEFAULT_MS   500
#define VIB_MANUAL_MS    3000
#define VIB_MAX_MS       10000UL

// ======================= TYPES / GLOBALS =======================
struct ImuSample { float ax, ay, az, gx, gy, gz; };
struct Features {
  float pitch, roll, pitchRaw, pitchLpf;
  float pitchVel, pitchAcc;
  float accMag, gyroMag;
  float energy, variance, nodScore;
  uint32_t headDownMs;
};
struct Pattern { uint16_t onMs; uint16_t offMs; uint8_t pwm; };

enum NodState   { NOD_IDLE, NOD_TRACK, NOD_LOCK };
enum ImuStatus  { ST_OK, ST_DEGRADED, ST_INVALID, ST_DISCONNECTED, ST_CALIBRATING, ST_UNKNOWN };
enum CalStatus  { CAL_NOT, CAL_RUNNING, CAL_DONE };

ImuSample imu;
Features  f;
float gyroBias[3] = {0, 0, 0};
float baselinePitch = 0, baselineRoll = 0;
float compPitch = 0, compRoll = 0, lpfPitch = 0, lpfRoll = 0;

float winG2[WIN], winA[WIN];
uint8_t winIdx = 0, winN = 0;

bool headDownActive = false;
float headDownAccumMs = 0;

NodState nodState = NOD_IDLE;
uint32_t nodStartMs = 0;
float nodPeakPitch = 0, nodPeakVel = 0;
uint32_t nodTimes[3] = {0, 0, 0};

ImuStatus imuStatus = ST_UNKNOWN;
CalStatus calStatus = CAL_NOT;
float imuQuality = 0.0f;
float failEma = 0.0f;
uint8_t failCount = 0;
uint32_t satUntilMs = 0, degradedUntilMs = 0, lastHeartbeatMs = 0, lastRecoverMs = 0;
bool wasDisconnected = false;

const Pattern PATTERNS[4] = {
  {0,   0,    0},     // level 0 STOP
  {200, 1800, 140},   // level 1 WARNING
  {300, 300,  200},   // level 2 HIGH_RISK
  {500, 150,  255}    // level 3 CRITICAL
};
uint8_t  vibLevel = 0;
uint32_t vibStartMs = 0, vibEndMs = 0;
bool     vibTimed = false;
long     vibCmdId = -1;
bool     demoMode = true;
bool     extendedOutput = false;
bool     motorOn = false;            // true while the vibration pattern is in its "on" phase

uint32_t seqId = 0, sampleCount = 0;
uint32_t lastSampleUs = 0;
const float RAD2DEG = 57.2957795f;

char cmdBuf[160];
uint8_t cmdLen = 0;
bool cmdInJson = false;

// ======================= LOW LEVEL I2C =======================
static void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static int16_t read16() {
  uint8_t h = Wire.read();
  uint8_t l = Wire.read();
  return (int16_t)((h << 8) | l);
}

bool initIMU() {
  mpuWrite(0x6B, 0x01);                 // wake up
  delay(100);
  mpuWrite(0x1A, MPU_DLPF_CFG);
  mpuWrite(0x1B, 0x08);                 // gyro  +-500 dps (65.5 LSB/dps)
  mpuWrite(0x1C, 0x00);                 // accel +-2 g     (16384 LSB/g)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);                     // WHO_AM_I
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1, (uint8_t)true) != 1) return false;
  uint8_t who = Wire.read();
  Serial.print(F("# WHO_AM_I=0x")); Serial.println(who, HEX);
  return (who != 0xFF && who != 0x00);
}

// ======================= REQUIRED FUNCTIONS =======================
bool readIMU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14, (uint8_t)true) != 14) return false;
  int16_t rax = read16(), ray = read16(), raz = read16();
  read16();                                  // temperature (ignored)
  int16_t rgx = read16(), rgy = read16(), rgz = read16();
  if ((rax == 0 && ray == 0 && raz == 0) || (rax == -1 && ray == -1 && raz == -1)) return false;
  imu.ax = rax / 16384.0f;
  imu.ay = ray / 16384.0f;
  imu.az = raz / 16384.0f;
  imu.gx = rgx / 65.5f - gyroBias[0];
  imu.gy = rgy / 65.5f - gyroBias[1];
  imu.gz = rgz / 65.5f - gyroBias[2];
  return true;
}

static void resetFeatures() {
  memset(&f, 0, sizeof(f));
  winIdx = 0; winN = 0;
  headDownActive = false; headDownAccumMs = 0;
  nodState = NOD_IDLE;
  for (uint8_t i = 0; i < 3; i++) nodTimes[i] = 0;
}

void logIMUData();   // forward declaration

bool calibrateIMU() {
  calStatus = CAL_RUNNING;
  imuStatus = ST_CALIBRATING;
  imuQuality = 0.0f;
  resetFeatures();
  bool accepted = false;
  float mean[6], m2[6];

  for (uint8_t attempt = 1; attempt <= CALIB_MAX_ATTEMPTS && !accepted; attempt++) {
    Serial.print(F("# CALIBRATION attempt ")); Serial.print(attempt);
    Serial.println(F(": keep head still in your normal driving posture (5 s)..."));
    gyroBias[0] = gyroBias[1] = gyroBias[2] = 0;
    for (uint8_t i = 0; i < 6; i++) { mean[i] = 0; m2[i] = 0; }
    uint16_t n = 0, fails = 0;
    uint32_t last = micros();
    while (n < CALIB_SAMPLES && fails <= 50) {
      if (micros() - last >= SAMPLE_US) {
        last += SAMPLE_US;
        if (!readIMU()) { fails++; continue; }
        float v[6] = {imu.ax, imu.ay, imu.az, imu.gx, imu.gy, imu.gz};
        n++;
        for (uint8_t i = 0; i < 6; i++) {          // Welford running mean / variance
          float d = v[i] - mean[i];
          mean[i] += d / n;
          m2[i] += d * (v[i] - mean[i]);
        }
        if (n % 25 == 0) {
          digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));   // blink while calibrating
          logIMUData();
        }
      }
    }
    digitalWrite(LED_BUILTIN, LOW);
    if (n < CALIB_SAMPLES) { Serial.println(F("# ERROR: IMU read failed during calibration")); break; }

    float sd[6];
    for (uint8_t i = 0; i < 6; i++) sd[i] = sqrt(m2[i] / (n - 1));
    float gMean = sqrt(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]);
    Serial.print(F("# static_noise_std accel_g="));
    for (uint8_t i = 0; i < 3; i++) { Serial.print(sd[i], 5); Serial.print(' '); }
    Serial.print(F("gyro_dps="));
    for (uint8_t i = 3; i < 6; i++) { Serial.print(sd[i], 4); Serial.print(' '); }
    Serial.print(F("mean_acc_mag_g=")); Serial.println(gMean, 4);

    bool still = sd[3] < CALIB_MAX_GYRO_STD && sd[4] < CALIB_MAX_GYRO_STD && sd[5] < CALIB_MAX_GYRO_STD &&
                 sd[0] < CALIB_MAX_ACC_STD && sd[1] < CALIB_MAX_ACC_STD && sd[2] < CALIB_MAX_ACC_STD &&
                 fabs(gMean - 1.0f) < CALIB_MAX_G_ERR;
    gyroBias[0] = mean[3]; gyroBias[1] = mean[4]; gyroBias[2] = mean[5];
    baselinePitch = PITCH_SIGN * atan2(-mean[0], sqrt(mean[1] * mean[1] + mean[2] * mean[2])) * RAD2DEG;
    baselineRoll  = ROLL_SIGN  * atan2(mean[1], mean[2]) * RAD2DEG;
    Serial.print(F("# gyro_bias_dps="));
    Serial.print(gyroBias[0], 3); Serial.print(' ');
    Serial.print(gyroBias[1], 3); Serial.print(' ');
    Serial.println(gyroBias[2], 3);
    Serial.print(F("# baseline_pitch_deg=")); Serial.print(baselinePitch, 2);
    Serial.print(F(" baseline_roll_deg="));   Serial.println(baselineRoll, 2);
    if (still || attempt == CALIB_MAX_ATTEMPTS) accepted = true;   // last attempt is always accepted
    else Serial.println(F("# CALIBRATION REJECTED: head moved or sensor unstable"));
  }

  compPitch = lpfPitch = baselinePitch;
  compRoll  = lpfRoll  = baselineRoll;
  resetFeatures();
  failEma = 0; failCount = 0;
  calStatus = accepted ? CAL_DONE : CAL_NOT;
  imuStatus = accepted ? ST_OK : ST_INVALID;
  Serial.println(accepted ? F("# CALIBRATED") : F("# NOT_CALIBRATED (press C to retry)"));
  if (accepted) {                                   // 3 fast blinks = calibration OK
    for (uint8_t i = 0; i < 3; i++) {
      digitalWrite(LED_BUILTIN, HIGH); delay(120);
      digitalWrite(LED_BUILTIN, LOW);  delay(120);
    }
  }
  return accepted;
}

void filterIMU(float dt) {
  float pAcc = PITCH_SIGN * atan2(-imu.ax, sqrt(imu.ay * imu.ay + imu.az * imu.az)) * RAD2DEG;
  float rAcc = ROLL_SIGN  * atan2(imu.ay, imu.az) * RAD2DEG;

  lpfPitch += LPF_ALPHA * (pAcc - lpfPitch);
  lpfRoll  += LPF_ALPHA * (rAcc - lpfRoll);

  compPitch = COMP_ALPHA * (compPitch + PITCH_SIGN * imu.gy * dt) + (1.0f - COMP_ALPHA) * pAcc;
  compRoll  = COMP_ALPHA * (compRoll  + ROLL_SIGN  * imu.gx * dt) + (1.0f - COMP_ALPHA) * rAcc;

  f.pitchRaw = pAcc - baselinePitch;
  f.pitchLpf = lpfPitch - baselinePitch;
  f.pitch    = compPitch - baselinePitch;
  f.roll     = compRoll - baselineRoll;
}

void calculateMotionFeatures(float dt) {
  float v = PITCH_SIGN * imu.gy;
  float prevVel = f.pitchVel;
  f.pitchVel += VEL_ALPHA * (v - f.pitchVel);
  f.pitchAcc += ACC_ALPHA * (((f.pitchVel - prevVel) / dt) - f.pitchAcc);

  f.accMag  = sqrt(imu.ax * imu.ax + imu.ay * imu.ay + imu.az * imu.az);
  f.gyroMag = sqrt(imu.gx * imu.gx + imu.gy * imu.gy + imu.gz * imu.gz);

  winG2[winIdx] = f.gyroMag * f.gyroMag;
  winA[winIdx]  = f.accMag;
  winIdx = (winIdx + 1) % WIN;
  if (winN < WIN) winN++;

  float sumG = 0, sumA = 0;
  for (uint8_t i = 0; i < winN; i++) { sumG += winG2[i]; sumA += winA[i]; }
  float meanA = sumA / winN, varA = 0;
  for (uint8_t i = 0; i < winN; i++) { float d = winA[i] - meanA; varA += d * d; }
  f.energy   = (sumG / winN) / 10000.0f;
  f.variance = varA / winN;
}

void updateHeadDownDuration(float dt) {
  if (f.pitch > HEAD_DOWN_ON_DEG) headDownActive = true;
  else if (f.pitch < HEAD_DOWN_OFF_DEG) headDownActive = false;
  if (headDownActive) headDownAccumMs += dt * 1000.0f;
  else headDownAccumMs = 0;
  f.headDownMs = (uint32_t)headDownAccumMs;
}

void calculateNoddingScore(float dt) {
  uint32_t now = millis();
  float p = f.pitch, v = f.pitchVel;

  switch (nodState) {
    case NOD_IDLE:
      if (p > NOD_ARM_DEG) {
        nodState = NOD_TRACK; nodStartMs = now; nodPeakPitch = p; nodPeakVel = v;
      }
      break;
    case NOD_TRACK: {
      if (p > nodPeakPitch) nodPeakPitch = p;
      if (v > nodPeakVel) nodPeakVel = v;
      uint32_t el = now - nodStartMs;
      if (p < NOD_RETURN_DEG) {
        if (el >= NOD_MIN_MS && el <= NOD_MAX_MS &&
            nodPeakPitch >= NOD_DEPTH_DEG && nodPeakVel >= NOD_MIN_VEL_DPS) {
          float depth = constrain(nodPeakPitch / 30.0f, 0.0f, 1.0f);
          float speed = constrain(nodPeakVel / 120.0f, 0.0f, 1.0f);
          float score = 0.5f * depth + 0.5f * speed;
          if (score > f.nodScore) f.nodScore = score;
          nodTimes[2] = nodTimes[1]; nodTimes[1] = nodTimes[0]; nodTimes[0] = now;
        }
        nodState = NOD_IDLE;
      } else if (el > NOD_MAX_MS) {
        nodState = NOD_LOCK;
      }
    } break;
    case NOD_LOCK:
      if (p < NOD_RETURN_DEG) nodState = NOD_IDLE;
      break;
  }
  f.nodScore *= expf(-dt * 1000.0f / NOD_DECAY_MS);
}

void updateImuQuality(bool readOk) {
  uint32_t now = millis();
  failEma += 0.05f * ((readOk ? 0.0f : 1.0f) - failEma);
  float q = 1.0f;
  q -= constrain(failEma * 5.0f, 0.0f, 0.6f);
  if (readOk) {
    float gm = fabs(imu.gx); if (fabs(imu.gy) > gm) gm = fabs(imu.gy); if (fabs(imu.gz) > gm) gm = fabs(imu.gz);
    if (gm > GYRO_SAT_DPS) satUntilMs = now + 200;
    if (fabs(f.accMag - 1.0f) > 0.6f) q -= 0.15f;
  }
  if ((int32_t)(satUntilMs - now) > 0) q -= 0.35f;
  if ((int32_t)(degradedUntilMs - now) > 0) q -= 0.4f;
  imuQuality = constrain(q, 0.0f, 1.0f);

  if (calStatus == CAL_RUNNING)      imuStatus = ST_CALIBRATING;
  else if (calStatus == CAL_NOT)     imuStatus = ST_INVALID;
  else if (imuQuality < INVALID_BELOW)  imuStatus = ST_INVALID;
  else if (imuQuality < DEGRADED_BELOW) imuStatus = ST_DEGRADED;
  else imuStatus = ST_OK;
}

// ---- vibration / LED pattern ----
void setVibrationLevel(uint8_t level, uint32_t durationMs) {
  if (level > 3) level = 3;
  uint32_t now = millis();
  if (level == 0) { vibLevel = 0; vibTimed = false; return; }
  if (durationMs > VIB_MAX_MS) durationMs = VIB_MAX_MS;
  if (level != vibLevel) vibStartMs = now;
  vibLevel = level;
  vibTimed = durationMs > 0;
  vibEndMs = now + durationMs;
}

// Motor output (pin 9, unused if nothing is connected). Remembers if the pattern is "on".
static void motorWrite(uint8_t pwm) {
  analogWrite(MOTOR_PIN, (uint16_t)pwm * MOTOR_PWM_MAX / 255);
  motorOn = (pwm > 0);
}

void updateVibration() {
  if (vibLevel && vibTimed && (int32_t)(millis() - vibEndMs) >= 0) { vibLevel = 0; vibTimed = false; }
  if (vibLevel == 0) { motorWrite(0); return; }
  const Pattern &p = PATTERNS[vibLevel];
  uint32_t t = (millis() - vibStartMs) % (uint32_t)(p.onMs + p.offMs);
  motorWrite((t < p.onMs) ? p.pwm : 0);
}

// Onboard LED: ON while the head is down (past HEAD_DOWN_ON_DEG), OR during a vibration pulse.
static void updateLed() {
  bool headDown = headDownActive && imuStatus != ST_DISCONNECTED && calStatus == CAL_DONE;
  digitalWrite(LED_BUILTIN, (headDown || motorOn) ? HIGH : LOW);
}

// ---- DEMO placeholder logic ----
static uint8_t recentNodCount() {
  uint32_t now = millis(); uint8_t c = 0;
  for (uint8_t i = 0; i < 3; i++) if (nodTimes[i] != 0 && now - nodTimes[i] < 10000UL) c++;
  return c;
}
uint8_t decideDemoLevel() {
  if (f.headDownMs >= 3000 || recentNodCount() >= 3) return 3;
  if (f.headDownMs >= 1500 || f.nodScore > 0.6f)     return 2;
  if (f.headDownMs >= 600  || f.nodScore > 0.3f)     return 1;
  return 0;
}

// ======================= OUTPUT =======================
static const char* statusStr(ImuStatus s) {
  switch (s) {
    case ST_OK: return "OK";
    case ST_DEGRADED: return "DEGRADED";
    case ST_INVALID: return "INVALID";
    case ST_DISCONNECTED: return "DISCONNECTED";
    case ST_CALIBRATING: return "CALIBRATING";
    default: return "UNKNOWN";
  }
}
static float fin(float v) { return isfinite(v) ? v : 0.0f; }

void logIMUData() {
  bool live = (imuStatus != ST_CALIBRATING && imuStatus != ST_DISCONNECTED);
  Serial.print(F("{\"type\":\"imu\",\"device_id\":\"" DEVICE_ID "\",\"sequence_id\":")); Serial.print(seqId);
  Serial.print(F(",\"device_timestamp_ms\":")); Serial.print(millis());
  Serial.print(F(",\"ax_g\":"));  Serial.print(live ? fin(imu.ax) : 0.0f, 3);
  Serial.print(F(",\"ay_g\":"));  Serial.print(live ? fin(imu.ay) : 0.0f, 3);
  Serial.print(F(",\"az_g\":"));  Serial.print(live ? fin(imu.az) : 0.0f, 3);
  Serial.print(F(",\"gx_dps\":")); Serial.print(live ? fin(imu.gx) : 0.0f, 2);
  Serial.print(F(",\"gy_dps\":")); Serial.print(live ? fin(imu.gy) : 0.0f, 2);
  Serial.print(F(",\"gz_dps\":")); Serial.print(live ? fin(imu.gz) : 0.0f, 2);
  Serial.print(F(",\"pitch_deg\":"));  Serial.print(live ? fin(f.pitch) : 0.0f, 2);
  Serial.print(F(",\"roll_deg\":"));   Serial.print(live ? fin(f.roll) : 0.0f, 2);
  Serial.print(F(",\"pitch_velocity_dps\":")); Serial.print(live ? fin(f.pitchVel) : 0.0f, 2);
  Serial.print(F(",\"head_down_duration_ms\":")); Serial.print(live ? f.headDownMs : 0UL);
  Serial.print(F(",\"nodding_score\":")); Serial.print(live ? fin(f.nodScore) : 0.0f, 3);
  Serial.print(F(",\"motion_energy\":")); Serial.print(live ? fin(f.energy) : 0.0f, 3);
  Serial.print(F(",\"imu_status\":\"")); Serial.print(statusStr(imuStatus));
  Serial.print(F("\",\"calibration_status\":\""));
  Serial.print(calStatus == CAL_DONE ? F("CALIBRATED") : (calStatus == CAL_RUNNING ? F("CALIBRATING") : F("NOT_CALIBRATED")));
  Serial.print(F("\",\"imu_quality\":")); Serial.print(imuQuality, 2);
  if (extendedOutput) {
    Serial.print(F(",\"acc_mag_g\":"));          Serial.print(live ? fin(f.accMag) : 0.0f, 3);
    Serial.print(F(",\"gyro_mag_dps\":"));       Serial.print(live ? fin(f.gyroMag) : 0.0f, 2);
    Serial.print(F(",\"pitch_accel_dps2\":"));   Serial.print(live ? fin(f.pitchAcc) : 0.0f, 1);
    Serial.print(F(",\"motion_variance_g2\":")); Serial.print(live ? fin(f.variance) : 0.0f, 6);
    Serial.print(F(",\"pitch_raw_deg\":"));      Serial.print(live ? fin(f.pitchRaw) : 0.0f, 2);
    Serial.print(F(",\"pitch_lpf_deg\":"));      Serial.print(live ? fin(f.pitchLpf) : 0.0f, 2);
    Serial.print(F(",\"vibration_level\":"));    Serial.print(vibLevel);
  }
  Serial.println('}');
  seqId++;
}

// ======================= COMMANDS =======================
static bool jsonFindLong(const char* s, const char* key, long* out) {
  const char* p = strstr(s, key);
  if (!p) return false;
  p = strchr(p, ':');
  if (!p) return false;
  p++;
  while (*p == ' ') p++;
  char* end;
  long v = strtol(p, &end, 10);
  if (end == p) return false;
  *out = v;
  return true;
}

static void handleJsonCommand(const char* s) {
  if (!strstr(s, "vibration_command")) { Serial.println(F("# ERR unknown message type")); return; }
  long level, dur = VIB_DEFAULT_MS, id = -1;
  if (!jsonFindLong(s, "\"level\"", &level) || level < 0 || level > 3) { Serial.println(F("# ERR bad level")); return; }
  jsonFindLong(s, "\"duration_ms\"", &dur);
  jsonFindLong(s, "\"command_id\"", &id);
  if (dur < 50) dur = 50;
  demoMode = false;
  vibCmdId = id;
  setVibrationLevel((uint8_t)level, (uint32_t)dur);
  Serial.print(F("# ack vibration_command id=")); Serial.print(id);
  Serial.print(F(" level=")); Serial.print(level);
  Serial.print(F(" duration_ms=")); Serial.println(level == 0 ? 0L : dur);
}

static void handleChar(char c) {
  if (c >= '0' && c <= '3') { demoMode = false; setVibrationLevel(c - '0', VIB_MANUAL_MS); }
  else if (c == 'D' || c == 'd') demoMode = true;
  else if (c == 'E' || c == 'e') extendedOutput = true;
  else if (c == 'S' || c == 's') extendedOutput = false;
  else if (c == 'C' || c == 'c') {
    setVibrationLevel(0, 0); updateVibration();
    calibrateIMU();
    lastSampleUs = micros();
  } else if (c == '?') {
    Serial.print(F("# status imu=")); Serial.print(statusStr(imuStatus));
    Serial.print(F(" cal=")); Serial.print(calStatus == CAL_DONE ? 1 : 0);
    Serial.print(F(" q=")); Serial.print(imuQuality, 2);
    Serial.print(F(" demo=")); Serial.print(demoMode ? 1 : 0);
    Serial.print(F(" ext=")); Serial.print(extendedOutput ? 1 : 0);
    Serial.print(F(" vib=")); Serial.println(vibLevel);
  }
}

static void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (cmdInJson) {
      if (c == '\n' || c == '\r') {
        cmdBuf[cmdLen] = 0;
        if (cmdLen > 0) handleJsonCommand(cmdBuf);
        cmdInJson = false; cmdLen = 0;
      } else if (cmdLen < sizeof(cmdBuf) - 1) cmdBuf[cmdLen++] = c;
      else { cmdInJson = false; cmdLen = 0; Serial.println(F("# ERR command too long")); }
    } else if (c == '{') {
      cmdInJson = true; cmdLen = 0; cmdBuf[cmdLen++] = c;
    } else if (c != '\n' && c != '\r' && c != ' ') {
      handleChar(c);
    }
  }
}

// ======================= SETUP / LOOP =======================
void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  analogWrite(MOTOR_PIN, 0);

  Wire.begin();
  Wire.setClock(100000);
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);
#endif
  if (!initIMU()) {
    imuStatus = ST_DISCONNECTED;
    Serial.println(F("# ERROR: MPU6050 not found - check SDA(A4)/SCL(A5)/VCC/GND (will keep retrying)"));
  } else {
    digitalWrite(LED_BUILTIN, HIGH); delay(250); digitalWrite(LED_BUILTIN, LOW);   // self-test blink
    delay(1500);
    calibrateIMU();
  }
  Serial.println(F("# SmartCap IMU + onboard LED | send ? for status"));
  lastSampleUs = micros();
}

void loop() {
  handleSerial();
  updateVibration();
  updateLed();

  uint32_t nowUs = micros();
  if (nowUs - lastSampleUs < SAMPLE_US) return;
  float dt = (nowUs - lastSampleUs) * 1e-6f;
  lastSampleUs = nowUs;
  if (dt < 0.001f) dt = 0.001f;
  if (dt > 0.1f)   dt = 0.1f;
  uint32_t nowMs = millis();

  if (!readIMU()) {
    if (failCount < 255) failCount++;
    updateImuQuality(false);
    if (failCount >= I2C_FAIL_DISCONNECT) {
      imuStatus = ST_DISCONNECTED; imuQuality = 0.0f; wasDisconnected = true;
      if (nowMs - lastHeartbeatMs >= 200) { lastHeartbeatMs = nowMs; logIMUData(); }
      if (nowMs - lastRecoverMs >= 1000) {
        lastRecoverMs = nowMs;
        Wire.begin(); Wire.setClock(100000);
      }
      if (vibLevel && demoMode) setVibrationLevel(0, 0);
    }
    return;
  }
  if (wasDisconnected) {
    wasDisconnected = false;
    initIMU();
    compPitch = lpfPitch = baselinePitch + f.pitch;
    degradedUntilMs = millis() + RECOVERY_DEGRADED_MS;
    nowMs = millis();
  }
  failCount = 0;

  filterIMU(dt);
  calculateMotionFeatures(dt);
  updateHeadDownDuration(dt);
  calculateNoddingScore(dt);
  updateImuQuality(true);

  if (demoMode && calStatus == CAL_DONE) setVibrationLevel(decideDemoLevel(), 0);

  sampleCount++;
  if (sampleCount % OUTPUT_DIV == 0) logIMUData();
}
