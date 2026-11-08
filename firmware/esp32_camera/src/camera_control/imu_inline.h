#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <string.h>

// SmartCap ESP32-S3 IMU port.
// Camera handlers and routes are intentionally unchanged.

// ---------------- Hardware / timing ----------------
static constexpr int IMU_SDA_PIN = 14;
static constexpr int IMU_SCL_PIN = 47;
static constexpr int IMU_I2C_CONTROLLER = 0;
static constexpr uint32_t IMU_I2C_FREQUENCY = 50000; // preserve known-good bus setting
static constexpr uint32_t IMU_SAMPLE_PERIOD_MS = 20; // 50 Hz
static constexpr uint32_t IMU_RETRY_PERIOD_MS = 3000;
static constexpr uint32_t IMU_STALE_MS = 500;
static constexpr uint8_t IMU_MAX_READ_FAILURES = 5;

static constexpr uint8_t MPU_ADDR_LOW = 0x68;
static constexpr uint8_t MPU_ADDR_HIGH = 0x69;
static constexpr float ACCEL_LSB_PER_G = 16384.0f; // +/-2g
static constexpr float GYRO_LSB_PER_DPS = 65.5f;   // +/-500 dps

// ---------------- Calibration / feature thresholds ----------------
static constexpr uint16_t CALIB_SAMPLES = 250;      // 5 seconds at 50 Hz
static constexpr uint8_t CALIB_MAX_ATTEMPTS = 3;
static constexpr float CALIB_MAX_GYRO_STD = 1.0f;  // dps, per axis
static constexpr float CALIB_MAX_ACC_STD = 0.02f;  // g, per axis
static constexpr float CALIB_MAX_G_ERR = 0.15f;    // |mean |a| - 1g|

static constexpr float LPF_ALPHA = 0.25f;
static constexpr float COMP_ALPHA = 0.98f;
static constexpr float VEL_ALPHA = 0.30f;
static constexpr float ACC_ALPHA = 0.20f;

static constexpr float HEAD_DOWN_ON_DEG = 15.0f;
static constexpr float HEAD_DOWN_OFF_DEG = 10.0f;
static constexpr float NOD_ARM_DEG = 6.0f;
static constexpr float NOD_DEPTH_DEG = 12.0f;
static constexpr float NOD_RETURN_DEG = 3.0f;
static constexpr uint32_t NOD_MIN_MS = 150;
static constexpr uint32_t NOD_MAX_MS = 2500;
static constexpr float NOD_MIN_VEL_DPS = 25.0f;
static constexpr uint32_t NOD_DECAY_MS = 1500;
static constexpr float GYRO_SAT_DPS = 450.0f;
static constexpr float RAD2DEG = 57.2957795f;

// Mounting convention from the Arduino reference:
// +X points toward the cap brim, +Z points upward.
// Positive pitch means nose down; positive roll means right ear down.
// Adjust these signs only after the physical direction test.
static constexpr float PITCH_SIGN = 1.0f;
static constexpr float ROLL_SIGN = 1.0f;

// ---------------- Data structures ----------------
struct ImuRawSample {
  uint32_t sequenceId = 0;
  uint32_t timestampMs = 0;
  bool valid = false;
  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;
  float pitch = 0, roll = 0;
  float pitchVelocity = 0, pitchAcceleration = 0;
  float headDownDurationMs = 0;
  float noddingScore = 0;
  float motionEnergy = 0;
  float accelMagnitude = 0;
  float gyroMagnitude = 0;
  float quality = 0;
  uint8_t imuState = 0; // 0 unknown, 1 calibrating, 2 OK, 3 degraded, 4 invalid, 5 disconnected
  uint8_t calibrationState = 0; // 0 not calibrated, 1 calibrating, 2 calibrated, 3 failed
};

static TwoWire imuWire(IMU_I2C_CONTROLLER);
static ImuRawSample imuLatest;
static portMUX_TYPE imuMux = portMUX_INITIALIZER_UNLOCKED;

static bool imuBusStarted = false;
static bool imuReady = false;
static uint8_t imuAddress = 0;
static uint8_t imuConsecutiveReadFailures = 0;
static uint32_t imuLastAttemptMs = 0;
static uint32_t imuLastSampleMs = 0;
static uint32_t imuLastTickMs = 0;
static uint8_t imuLastI2CError = 0;
static uint8_t imuLastBytesReceived = 0;
static char imuDiagnostic[32] = "NOT_STARTED";
static char imuScanResults[96] = "[]";

static uint8_t calibrationState = 0;
static uint8_t imuState = 0;
static float gyroBias[3] = {0, 0, 0};
static float baselinePitch = 0, baselineRoll = 0;
static float compPitch = 0, compRoll = 0;
static float lpfPitch = 0, lpfRoll = 0;
static float pitchVelocity = 0, pitchAcceleration = 0;
static float headDownAccumMs = 0;
static bool headDownActive = false;

enum ImuNodState : uint8_t { NOD_IDLE, NOD_TRACK, NOD_LOCK };
static ImuNodState nodState = NOD_IDLE;
static uint32_t nodStartMs = 0;
static float nodPeakPitch = 0, nodPeakVelocity = 0;
static float noddingScore = 0;

// ---------------- Low-level I2C ----------------
static bool imuWriteRegister(uint8_t reg, uint8_t value) {
  imuWire.beginTransmission(imuAddress);
  imuWire.write(reg);
  imuWire.write(value);
  imuLastI2CError = imuWire.endTransmission();
  return imuLastI2CError == 0;
}

static bool imuReadRegisters(uint8_t reg, uint8_t *data, size_t length) {
  imuWire.beginTransmission(imuAddress);
  imuWire.write(reg);
  imuLastI2CError = imuWire.endTransmission(false);
  if (imuLastI2CError != 0) {
    imuLastBytesReceived = 0;
    return false;
  }

  size_t received = imuWire.requestFrom(imuAddress, (uint8_t)length, (uint8_t)true);
  imuLastBytesReceived = (uint8_t)received;
  if (received != length) {
    while (imuWire.available()) imuWire.read();
    return false;
  }

  for (size_t i = 0; i < length; ++i) {
    if (!imuWire.available()) return false;
    data[i] = (uint8_t)imuWire.read();
  }
  return true;
}

static int16_t imuToInt16(uint8_t high, uint8_t low) {
  return (int16_t)(((uint16_t)high << 8) | low);
}

static void imuScanBus() {
  char devices[96] = "[";
  size_t used = 1;
  bool first = true;
  imuAddress = 0;

  for (uint8_t address = 1; address < 127; ++address) {
    imuWire.beginTransmission(address);
    if (imuWire.endTransmission() == 0) {
      char item[10];
      snprintf(item, sizeof(item), "%s\"0x%02X\"", first ? "" : ",", address);
      size_t n = strlen(item);
      if (used + n + 2 < sizeof(devices)) {
        memcpy(devices + used, item, n);
        used += n;
        devices[used] = '\0';
      }
      first = false;
      if (address == MPU_ADDR_LOW || address == MPU_ADDR_HIGH) imuAddress = address;
    }
  }

  if (used + 2 <= sizeof(devices)) {
    devices[used++] = ']';
    devices[used] = '\0';
  }
  strlcpy(imuScanResults, devices, sizeof(imuScanResults));
}

static bool imuBegin() {
  if (!imuBusStarted) {
    if (!imuWire.begin(IMU_SDA_PIN, IMU_SCL_PIN, IMU_I2C_FREQUENCY)) {
      strlcpy(imuDiagnostic, "I2C_BEGIN_FAILED", sizeof(imuDiagnostic));
      Serial.println("# IMU: I2C initialization failed");
      return false;
    }
    imuWire.setTimeOut(50);
    imuBusStarted = true;
    Serial.println("# IMU: I2C initialized");
  }

  imuScanBus();
  if (imuAddress == 0) {
    strlcpy(imuDiagnostic, "I2C_DEVICE_NOT_FOUND", sizeof(imuDiagnostic));
    Serial.println("# IMU: MPU6050 not found");
    return false;
  }

  uint8_t whoAmI = 0;
  if (!imuReadRegisters(0x75, &whoAmI, 1)) {
    strlcpy(imuDiagnostic, "WHO_AM_I_READ_FAILED", sizeof(imuDiagnostic));
    return false;
  }
  if (whoAmI != 0x68) {
    strlcpy(imuDiagnostic, "WHO_AM_I_MISMATCH", sizeof(imuDiagnostic));
    Serial.printf("# IMU: Unexpected WHO_AM_I 0x%02X\n", whoAmI);
    return false;
  }

  // Wake MPU6050; configure DLPF, +/-500 dps gyro, +/-2g accelerometer.
  if (!imuWriteRegister(0x6B, 0x01)) return false;
  delay(100);
  if (!imuWriteRegister(0x1A, 0x00)) return false;
  if (!imuWriteRegister(0x1B, 0x08)) return false;
  if (!imuWriteRegister(0x1C, 0x00)) return false;

  imuConsecutiveReadFailures = 0;
  imuLastI2CError = 0;
  imuLastBytesReceived = 0;
  strlcpy(imuDiagnostic, "CONNECTED", sizeof(imuDiagnostic));
  Serial.printf("# IMU: MPU6050 ready at 0x%02X (+/-2g, +/-500dps)\n", imuAddress);
  return true;
}

static bool imuReadRaw(ImuRawSample &sample) {
  uint8_t data[14];
  if (!imuReadRegisters(0x3B, data, sizeof(data))) return false;

  int16_t rax = imuToInt16(data[0], data[1]);
  int16_t ray = imuToInt16(data[2], data[3]);
  int16_t raz = imuToInt16(data[4], data[5]);
  int16_t rgx = imuToInt16(data[8], data[9]);
  int16_t rgy = imuToInt16(data[10], data[11]);
  int16_t rgz = imuToInt16(data[12], data[13]);

  if ((rax == 0 && ray == 0 && raz == 0) ||
      (rax == -1 && ray == -1 && raz == -1)) return false;

  sample.ax = rax / ACCEL_LSB_PER_G;
  sample.ay = ray / ACCEL_LSB_PER_G;
  sample.az = raz / ACCEL_LSB_PER_G;
  sample.gx = rgx / GYRO_LSB_PER_DPS;
  sample.gy = rgy / GYRO_LSB_PER_DPS;
  sample.gz = rgz / GYRO_LSB_PER_DPS;
  sample.valid = true;
  return true;
}

static void resetMotionState() {
  pitchVelocity = 0;
  pitchAcceleration = 0;
  headDownAccumMs = 0;
  headDownActive = false;
  nodState = NOD_IDLE;
  nodStartMs = 0;
  nodPeakPitch = 0;
  nodPeakVelocity = 0;
  noddingScore = 0;
}

// ---------------- Calibration ----------------
static bool imuCalibrate() {
  calibrationState = 1;
  imuState = 1;
  resetMotionState();

  bool accepted = false;
  float mean[6], m2[6];

  for (uint8_t attempt = 1; attempt <= CALIB_MAX_ATTEMPTS && !accepted; ++attempt) {
    Serial.printf("# IMU calibration %u/%u: keep the cap still in normal wearing position for 5 seconds\n",
                  attempt, CALIB_MAX_ATTEMPTS);
    memset(mean, 0, sizeof(mean));
    memset(m2, 0, sizeof(m2));

    uint16_t count = 0;
    uint8_t failures = 0;
    uint32_t nextSample = millis();

    while (count < CALIB_SAMPLES && failures <= 50) {
      uint32_t now = millis();
      if ((int32_t)(now - nextSample) < 0) {
        delay(1);
        continue;
      }
      nextSample += IMU_SAMPLE_PERIOD_MS;

      ImuRawSample s;
      if (!imuReadRaw(s)) {
        ++failures;
        continue;
      }

      const float values[6] = {s.ax, s.ay, s.az, s.gx, s.gy, s.gz};
      ++count;
      for (uint8_t i = 0; i < 6; ++i) {
        float delta = values[i] - mean[i];
        mean[i] += delta / count;
        m2[i] += delta * (values[i] - mean[i]);
      }

      if (count % 50 == 0) {
        Serial.printf("# IMU calibration samples: %u/%u\n", count, CALIB_SAMPLES);
      }
    }

    if (count < CALIB_SAMPLES) {
      Serial.println("# IMU calibration failed: not enough valid samples");
      break;
    }

    float sd[6];
    for (uint8_t i = 0; i < 6; ++i) sd[i] = sqrtf(m2[i] / (count - 1));
    float gravityMagnitude = sqrtf(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]);

    Serial.printf("# IMU noise std accel(g): %.5f %.5f %.5f | gyro(dps): %.4f %.4f %.4f | |a|=%.4f g\n",
                  sd[0], sd[1], sd[2], sd[3], sd[4], sd[5], gravityMagnitude);

    bool still =
      sd[0] < CALIB_MAX_ACC_STD && sd[1] < CALIB_MAX_ACC_STD && sd[2] < CALIB_MAX_ACC_STD &&
      sd[3] < CALIB_MAX_GYRO_STD && sd[4] < CALIB_MAX_GYRO_STD && sd[5] < CALIB_MAX_GYRO_STD &&
      fabsf(gravityMagnitude - 1.0f) < CALIB_MAX_G_ERR;

    if (still) {
      gyroBias[0] = mean[3];
      gyroBias[1] = mean[4];
      gyroBias[2] = mean[5];

      baselinePitch = PITCH_SIGN * atan2f(-mean[0],
        sqrtf(mean[1] * mean[1] + mean[2] * mean[2])) * RAD2DEG;
      baselineRoll = ROLL_SIGN * atan2f(mean[1], mean[2]) * RAD2DEG;

      accepted = true;
      Serial.printf("# IMU calibration accepted. gyro bias: %.3f %.3f %.3f dps | baseline pitch/roll: %.2f / %.2f deg\n",
                    gyroBias[0], gyroBias[1], gyroBias[2], baselinePitch, baselineRoll);
    } else {
      Serial.println("# IMU calibration rejected: movement/noise/gravity check failed");
    }
  }

  if (accepted) {
    compPitch = lpfPitch = baselinePitch;
    compRoll = lpfRoll = baselineRoll;
    calibrationState = 2;
    imuState = 2;
    strlcpy(imuDiagnostic, "CALIBRATED", sizeof(imuDiagnostic));
    Serial.println("# IMU CALIBRATED");
  } else {
    calibrationState = 3;
    imuState = 4;
    strlcpy(imuDiagnostic, "CALIBRATION_FAILED", sizeof(imuDiagnostic));
    Serial.println("# IMU NOT CALIBRATED. Keep still and restart/retry.");
  }

  resetMotionState();
  imuLastTickMs = millis();
  return accepted;
}

// ---------------- Feature extraction ----------------
static void imuUpdateFeatures(ImuRawSample &s, float dt) {
  s.gx -= gyroBias[0];
  s.gy -= gyroBias[1];
  s.gz -= gyroBias[2];

  const float pitchAccAngle = PITCH_SIGN *
    atan2f(-s.ax, sqrtf(s.ay * s.ay + s.az * s.az)) * RAD2DEG;
  const float rollAccAngle = ROLL_SIGN * atan2f(s.ay, s.az) * RAD2DEG;

  lpfPitch += LPF_ALPHA * (pitchAccAngle - lpfPitch);
  lpfRoll += LPF_ALPHA * (rollAccAngle - lpfRoll);

  compPitch = COMP_ALPHA * (compPitch + PITCH_SIGN * s.gy * dt) +
              (1.0f - COMP_ALPHA) * pitchAccAngle;
  compRoll = COMP_ALPHA * (compRoll + ROLL_SIGN * s.gx * dt) +
             (1.0f - COMP_ALPHA) * rollAccAngle;

  s.pitch = compPitch - baselinePitch;
  s.roll = compRoll - baselineRoll;

  float previousVelocity = pitchVelocity;
  float measuredVelocity = PITCH_SIGN * s.gy;
  pitchVelocity += VEL_ALPHA * (measuredVelocity - pitchVelocity);
  pitchAcceleration += ACC_ALPHA *
    (((pitchVelocity - previousVelocity) / dt) - pitchAcceleration);

  s.pitchVelocity = pitchVelocity;
  s.pitchAcceleration = pitchAcceleration;
  s.accelMagnitude = sqrtf(s.ax*s.ax + s.ay*s.ay + s.az*s.az);
  s.gyroMagnitude = sqrtf(s.gx*s.gx + s.gy*s.gy + s.gz*s.gz);
  s.motionEnergy = (s.gyroMagnitude * s.gyroMagnitude) / 10000.0f;

  if (s.pitch > HEAD_DOWN_ON_DEG) headDownActive = true;
  else if (s.pitch < HEAD_DOWN_OFF_DEG) headDownActive = false;

  if (headDownActive) headDownAccumMs += dt * 1000.0f;
  else headDownAccumMs = 0;
  s.headDownDurationMs = headDownAccumMs;

  uint32_t now = millis();
  switch (nodState) {
    case NOD_IDLE:
      if (s.pitch > NOD_ARM_DEG) {
        nodState = NOD_TRACK;
        nodStartMs = now;
        nodPeakPitch = s.pitch;
        nodPeakVelocity = s.pitchVelocity;
      }
      break;

    case NOD_TRACK: {
      if (s.pitch > nodPeakPitch) nodPeakPitch = s.pitch;
      if (s.pitchVelocity > nodPeakVelocity) nodPeakVelocity = s.pitchVelocity;
      uint32_t elapsed = now - nodStartMs;

      if (s.pitch < NOD_RETURN_DEG) {
        if (elapsed >= NOD_MIN_MS && elapsed <= NOD_MAX_MS &&
            nodPeakPitch >= NOD_DEPTH_DEG &&
            nodPeakVelocity >= NOD_MIN_VEL_DPS) {
          float depth = constrain(nodPeakPitch / 30.0f, 0.0f, 1.0f);
          float speed = constrain(nodPeakVelocity / 120.0f, 0.0f, 1.0f);
          float score = 0.5f * depth + 0.5f * speed;
          if (score > noddingScore) noddingScore = score;
        }
        nodState = NOD_IDLE;
      } else if (elapsed > NOD_MAX_MS) {
        nodState = NOD_LOCK;
      }
      break;
    }

    case NOD_LOCK:
      if (s.pitch < NOD_RETURN_DEG) nodState = NOD_IDLE;
      break;
  }

  noddingScore *= expf(-(dt * 1000.0f) / NOD_DECAY_MS);
  s.noddingScore = noddingScore;
}

// ---------------- Public functions used by camera_control.ino ----------------
static void imuStart() {
  imuLastAttemptMs = millis();
  imuReady = imuBegin();

  if (imuReady) {
    calibrationState = 1;
    imuState = 1;
    imuCalibrate();
    imuLastSampleMs = millis();
    imuLastTickMs = millis();
  } else {
    calibrationState = 0;
    imuState = 5;
  }
}

static void imuTick() {
  uint32_t now = millis();

  if (!imuReady) {
    if ((uint32_t)(now - imuLastAttemptMs) >= IMU_RETRY_PERIOD_MS) {
      imuLastAttemptMs = now;
      Serial.println("# IMU: attempting reconnect");
      imuReady = imuBegin();
      if (imuReady) {
        imuCalibrate();
        imuLastSampleMs = millis();
        imuLastTickMs = millis();
      }
    }
    return;
  }

  if ((uint32_t)(now - imuLastSampleMs) < IMU_SAMPLE_PERIOD_MS) return;
  imuLastSampleMs = now;

  ImuRawSample sample;
  if (!imuReadRaw(sample)) {
    if (imuConsecutiveReadFailures < 255) ++imuConsecutiveReadFailures;

    if (imuConsecutiveReadFailures >= IMU_MAX_READ_FAILURES) {
      imuReady = false;
      imuLastAttemptMs = now;
      imuState = 5;
      strlcpy(imuDiagnostic, "I2C_READ_FAILED", sizeof(imuDiagnostic));
      Serial.println("# IMU: repeated read failures; reconnect scheduled");
    }
    return;
  }

  imuConsecutiveReadFailures = 0;
  uint32_t sampleNow = millis();
  float dt = (sampleNow - imuLastTickMs) / 1000.0f;
  imuLastTickMs = sampleNow;
  if (dt < 0.001f) dt = 0.001f;
  if (dt > 0.1f) dt = 0.1f;

  sample.timestampMs = sampleNow;
  sample.valid = true;

  if (calibrationState == 2) {
    imuUpdateFeatures(sample, dt);

    float q = 1.0f;
    if (fabsf(sample.accelMagnitude - 1.0f) > 0.6f) q -= 0.15f;
    if (fabsf(sample.gx) > GYRO_SAT_DPS ||
        fabsf(sample.gy) > GYRO_SAT_DPS ||
        fabsf(sample.gz) > GYRO_SAT_DPS) q -= 0.35f;
    if (imuConsecutiveReadFailures > 0) q -= 0.2f;
    sample.quality = constrain(q, 0.0f, 1.0f);
    sample.imuState = sample.quality < 0.5f ? 3 : 2;
  } else {
    sample.quality = 0.0f;
    sample.imuState = 4;
  }
  sample.calibrationState = calibrationState;

  portENTER_CRITICAL(&imuMux);
  sample.sequenceId = imuLatest.sequenceId + 1;
  imuLatest = sample;
  portEXIT_CRITICAL(&imuMux);

  imuState = sample.imuState;
}

// ---------------- JSON endpoint ----------------
static const char *imuStatusText(uint8_t state) {
  switch (state) {
    case 1: return "CALIBRATING";
    case 2: return "OK";
    case 3: return "DEGRADED";
    case 4: return "INVALID";
    case 5: return "DISCONNECTED";
    default: return "UNKNOWN";
  }
}

static const char *imuCalibrationText(uint8_t state) {
  switch (state) {
    case 1: return "CALIBRATING";
    case 2: return "CALIBRATED";
    case 3: return "FAILED";
    default: return "NOT_CALIBRATED";
  }
}

static void imuMakeJson(char *buffer, size_t capacity, const char *deviceId) {
  ImuRawSample sample;
  portENTER_CRITICAL(&imuMux);
  sample = imuLatest;
  portEXIT_CRITICAL(&imuMux);

  uint32_t now = millis();
  uint32_t age = now - sample.timestampMs;
  bool fresh = sample.valid && age <= IMU_STALE_MS;

  uint8_t state = imuState;
  if (!fresh && imuReady) state = 4;
  if (!imuReady) state = 5;

  if (fresh) {
    snprintf(buffer, capacity,
      "{\"type\":\"imu_raw\",\"device_id\":\"%s\","
      "\"sequence_id\":%lu,\"device_timestamp_ms\":%lu,"
      "\"device_clock_domain\":\"esp32_millis\",\"sample_age_ms\":%lu,"
      "\"sample_valid\":true,"
      "\"ax_g\":%.5f,\"ay_g\":%.5f,\"az_g\":%.5f,"
      "\"gx_dps\":%.3f,\"gy_dps\":%.3f,\"gz_dps\":%.3f,"
      "\"pitch_deg\":%.2f,\"roll_deg\":%.2f,"
      "\"pitch_velocity_dps\":%.2f,\"pitch_acceleration_dps2\":%.2f,"
      "\"head_down_duration_ms\":%.0f,\"nodding_score\":%.3f,"
      "\"motion_energy\":%.4f,\"accel_magnitude_g\":%.4f,"
      "\"gyro_magnitude_dps\":%.2f,\"imu_quality\":%.2f,"
      "\"imu_status\":\"%s\",\"calibration_status\":\"%s\","
      "\"i2c_address\":\"0x%02X\",\"i2c_devices\":%s}",
      deviceId, (unsigned long)sample.sequenceId,
      (unsigned long)sample.timestampMs, (unsigned long)age,
      (double)sample.ax, (double)sample.ay, (double)sample.az,
      (double)sample.gx, (double)sample.gy, (double)sample.gz,
      (double)sample.pitch, (double)sample.roll,
      (double)sample.pitchVelocity, (double)sample.pitchAcceleration,
      (double)sample.headDownDurationMs, (double)sample.noddingScore,
      (double)sample.motionEnergy, (double)sample.accelMagnitude,
      (double)sample.gyroMagnitude, (double)sample.quality,
      imuStatusText(state), imuCalibrationText(sample.calibrationState),
      imuAddress, imuScanResults);
  } else {
    snprintf(buffer, capacity,
      "{\"type\":\"imu_raw\",\"device_id\":\"%s\","
      "\"sequence_id\":%lu,\"device_timestamp_ms\":%lu,"
      "\"device_clock_domain\":\"esp32_millis\",\"sample_age_ms\":null,"
      "\"sample_valid\":false,"
      "\"ax_g\":null,\"ay_g\":null,\"az_g\":null,"
      "\"gx_dps\":null,\"gy_dps\":null,\"gz_dps\":null,"
      "\"pitch_deg\":null,\"roll_deg\":null,"
      "\"pitch_velocity_dps\":null,\"pitch_acceleration_dps2\":null,"
      "\"head_down_duration_ms\":null,\"nodding_score\":null,"
      "\"motion_energy\":null,\"accel_magnitude_g\":null,"
      "\"gyro_magnitude_dps\":null,\"imu_quality\":0.0,"
      "\"imu_status\":\"%s\",\"calibration_status\":\"%s\","
      "\"i2c_address\":\"0x%02X\",\"i2c_devices\":%s}",
      deviceId, (unsigned long)sample.sequenceId,
      (unsigned long)sample.timestampMs,
      imuStatusText(state), imuCalibrationText(calibrationState),
      imuAddress, imuScanResults);
  }
}
