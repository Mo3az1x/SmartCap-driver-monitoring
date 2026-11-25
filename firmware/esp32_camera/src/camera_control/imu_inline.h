#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <string.h>
#include <math.h>

// ---------------- IMU Configuration ----------------

static constexpr int IMU_SDA_PIN = 14;
static constexpr int IMU_SCL_PIN = 47;

// Use I2C controller 1 to keep the IMU bus separate
// from the camera SCCB bus.
static constexpr int IMU_I2C_CONTROLLER = 0;

static constexpr uint32_t IMU_I2C_FREQUENCY = 50000;

static constexpr uint32_t IMU_SAMPLE_PERIOD_MS = 20;
static constexpr uint32_t IMU_RETRY_PERIOD_MS = 3000;
static constexpr uint32_t IMU_STALE_MS = 500;

static constexpr uint8_t IMU_MAX_READ_FAILURES = 5;

static constexpr float IMU_ACCEL_LSB_PER_G = 8192.0f;
static constexpr float IMU_GYRO_LSB_PER_DPS = 65.5f;

static constexpr uint8_t MPU_ADDR_LOW = 0x68;
static constexpr uint8_t MPU_ADDR_HIGH = 0x69;

static TwoWire imuWire(IMU_I2C_CONTROLLER);

// ---------------- Data Structures ----------------

struct ImuRawSample {
  uint32_t sequenceId = 0;
  uint32_t timestampMs = 0;
  bool valid = false;

  float ax = 0;
  float ay = 0;
  float az = 0;

  float gx = 0;
  float gy = 0;
  float gz = 0;

  float tiltAngleDeg = 0;
  bool orientationReady = false;
};

static ImuRawSample imuLatest;
static portMUX_TYPE imuMux = portMUX_INITIALIZER_UNLOCKED;

// ---------------- State Variables ----------------

static bool imuBusStarted = false;
static bool imuReady = false;

static uint8_t imuAddress = 0;
static uint8_t imuConsecutiveReadFailures = 0;

static char imuDiagnostic[32] = "NOT_STARTED";
static char imuScanResults[96] = "[]";

static uint32_t imuLastAttemptMs = 0;
static uint32_t imuLastSampleMs = 0;

static uint8_t imuLastI2CError = 0;
static uint8_t imuLastBytesReceived = 0;

// ---------------- Gyro Calibration ----------------

static constexpr uint16_t IMU_CAL_SAMPLES = 250;
static constexpr float IMU_CAL_MAX_GYRO_STD = 2.0f;
static constexpr uint32_t IMU_CAL_TIMEOUT_MS = 15000;

enum ImuCalState : uint8_t {
  CAL_NONE,
  CAL_RUNNING,
  CAL_DONE,
  CAL_FAILED
};

static volatile bool imuCalRequested = false;
static ImuCalState imuCalState = CAL_NONE;
static uint16_t imuCalCount = 0;
static uint32_t imuCalStartMs = 0;

static double imuCalSum[3] = {0, 0, 0};
static double imuCalSumSq[3] = {0, 0, 0};

static float imuGyroBias[3] = {0, 0, 0};
static float imuCalStd = 0;

// ---------------- Orientation Baseline ----------------

static constexpr uint16_t IMU_ORIENTATION_SAMPLES = 50;

static float imuAccelReference[3] = {0, 0, 0};
static double imuAccelSum[3] = {0, 0, 0};

static uint16_t imuOrientationCount = 0;
static bool imuOrientationReady = false;

static float imuCurrentTiltDeg = 0;

// ---------------- Gyro Calibration Functions ----------------

static void imuCalBegin() {

  imuCalCount = 0;
  imuCalStartMs = millis();

  for (int i = 0; i < 3; i++) {
    imuCalSum[i] = 0;
    imuCalSumSq[i] = 0;
  }

  imuCalState = CAL_RUNNING;

  Serial.println(
    "# IMU: Calibration started, keep the cap still"
  );
}

static void imuCalFeed(float gx, float gy, float gz) {

  float v[3] = {gx, gy, gz};

  for (int i = 0; i < 3; i++) {
    imuCalSum[i] += v[i];
    imuCalSumSq[i] += (double)v[i] * v[i];
  }

  if (++imuCalCount < IMU_CAL_SAMPLES) {
    return;
  }

  float mean[3];
  float worst = 0;

  for (int i = 0; i < 3; i++) {

    double m = imuCalSum[i] / imuCalCount;

    double var =
      imuCalSumSq[i] / imuCalCount - m * m;

    float sd = var > 0 ? (float)sqrt(var) : 0.0f;

    if (sd > worst) {
      worst = sd;
    }

    mean[i] = (float)m;
  }

  imuCalStd = worst;

  if (worst <= IMU_CAL_MAX_GYRO_STD) {

    for (int i = 0; i < 3; i++) {
      imuGyroBias[i] = mean[i];
    }

    imuCalState = CAL_DONE;

    Serial.printf(
      "# IMU: Calibration DONE, bias=%.2f %.2f %.2f dps, std=%.2f dps\n",
      (double)mean[0],
      (double)mean[1],
      (double)mean[2],
      (double)worst
    );

  } else {

    imuCalState = CAL_FAILED;

    Serial.printf(
      "# IMU: Calibration FAILED (moved or noisy), std=%.2f dps > %.2f\n",
      (double)worst,
      (double)IMU_CAL_MAX_GYRO_STD
    );
  }
}

static const char *imuCalibrationName() {

  switch (imuCalState) {

    case CAL_RUNNING:
      return "CALIBRATING";

    case CAL_DONE:
      return "CALIBRATED";

    default:
      return "NOT_CALIBRATED";
  }
}

// ---------------- Orientation Calculation ----------------

static void imuUpdateOrientation(ImuRawSample &sample) {

  float magnitude = sqrtf(
    sample.ax * sample.ax +
    sample.ay * sample.ay +
    sample.az * sample.az
  );

  // Ignore acceleration values outside a reasonable gravity range.
  if (magnitude < 0.75f || magnitude > 1.25f) {

    sample.orientationReady = imuOrientationReady;
    sample.tiltAngleDeg = imuCurrentTiltDeg;

    return;
  }

  float ax = sample.ax / magnitude;
  float ay = sample.ay / magnitude;
  float az = sample.az / magnitude;

  float gyroMagnitude = sqrtf(
    sample.gx * sample.gx +
    sample.gy * sample.gy +
    sample.gz * sample.gz
  );

  // Capture the initial orientation while the sensor is still.
  if (!imuOrientationReady) {

    if (gyroMagnitude < 5.0f) {

      imuAccelSum[0] += ax;
      imuAccelSum[1] += ay;
      imuAccelSum[2] += az;

      imuOrientationCount++;

      if (imuOrientationCount >= IMU_ORIENTATION_SAMPLES) {

        float rx =
          imuAccelSum[0] / imuOrientationCount;

        float ry =
          imuAccelSum[1] / imuOrientationCount;

        float rz =
          imuAccelSum[2] / imuOrientationCount;

        float refMagnitude = sqrtf(
          rx * rx +
          ry * ry +
          rz * rz
        );

        if (refMagnitude > 0.1f) {

          imuAccelReference[0] = rx / refMagnitude;
          imuAccelReference[1] = ry / refMagnitude;
          imuAccelReference[2] = rz / refMagnitude;

          imuOrientationReady = true;

          Serial.println(
            "# IMU: Initial orientation captured"
          );
        }
      }

    } else {

      // Restart capture if the sensor moves during initialization.
      imuOrientationCount = 0;

      imuAccelSum[0] = 0;
      imuAccelSum[1] = 0;
      imuAccelSum[2] = 0;
    }
  }

  if (imuOrientationReady) {

    float dot =
      ax * imuAccelReference[0] +
      ay * imuAccelReference[1] +
      az * imuAccelReference[2];

    dot = constrain(dot, -1.0f, 1.0f);

    imuCurrentTiltDeg =
      acosf(dot) * 180.0f / PI;
  }

  sample.tiltAngleDeg = imuCurrentTiltDeg;
  sample.orientationReady = imuOrientationReady;
}

// ---------------- Low-Level I2C ----------------

static bool imuWriteRegister(uint8_t reg, uint8_t value) {

  imuWire.beginTransmission(imuAddress);

  imuWire.write(reg);
  imuWire.write(value);

  imuLastI2CError = imuWire.endTransmission();

  if (imuLastI2CError != 0) {

    Serial.printf(
      "# IMU: I2C write failed, register=0x%02X error=%u\n",
      reg,
      imuLastI2CError
    );

    return false;
  }

  return true;
}

static bool imuReadRegisters(
  uint8_t reg,
  uint8_t *data,
  size_t length
) {

  imuWire.beginTransmission(imuAddress);

  imuWire.write(reg);

  imuLastI2CError = imuWire.endTransmission(false);

  if (imuLastI2CError != 0) {

    imuLastBytesReceived = 0;

    return false;
  }

  size_t received = imuWire.requestFrom(
    imuAddress,
    (uint8_t)length,
    (uint8_t)true
  );

  imuLastBytesReceived = (uint8_t)received;

  if (received != length) {

    while (imuWire.available()) {
      imuWire.read();
    }

    return false;
  }

  for (size_t i = 0; i < length; i++) {

    if (!imuWire.available()) {
      return false;
    }

    data[i] = imuWire.read();
  }

  return true;
}

static int16_t imuToInt16(uint8_t high, uint8_t low) {

  return (int16_t)(
    ((uint16_t)high << 8) | low
  );
}

// ---------------- I2C Scanner ----------------

static void imuScanBus() {

  char devices[96] = "[";

  size_t used = 1;

  bool first = true;

  imuAddress = 0;

  Serial.println("# IMU: Starting I2C scan...");

  for (uint8_t address = 1; address < 127; address++) {

    imuWire.beginTransmission(address);

    uint8_t error = imuWire.endTransmission();

    if (error == 0) {

      char item[10];

      snprintf(
        item,
        sizeof(item),
        "%s\"0x%02X\"",
        first ? "" : ",",
        address
      );

      size_t itemLength = strlen(item);

      if (used + itemLength + 2 < sizeof(devices)) {

        memcpy(
          devices + used,
          item,
          itemLength
        );

        used += itemLength;

        devices[used] = '\0';
      }

      first = false;

      Serial.printf(
        "# IMU: I2C device found at 0x%02X\n",
        address
      );

      if (
        address == MPU_ADDR_LOW ||
        address == MPU_ADDR_HIGH
      ) {
        imuAddress = address;
      }
    }
  }

  if (used + 2 <= sizeof(devices)) {

    devices[used++] = ']';

    devices[used] = '\0';
  }

  strlcpy(
    imuScanResults,
    devices,
    sizeof(imuScanResults)
  );

  Serial.printf(
    "# IMU: I2C scan result: %s\n",
    imuScanResults
  );

  if (imuAddress != 0) {

    Serial.printf(
      "# IMU: MPU6050 candidate address: 0x%02X\n",
      imuAddress
    );
  } else {

    Serial.println(
      "# IMU: No MPU6050 address found (0x68/0x69)"
    );
  }
}

// ---------------- Initialization ----------------

static bool imuBegin() {

  if (!imuBusStarted) {

    Serial.printf(
      "# IMU: Starting I2C controller %d, SDA=%d, SCL=%d, frequency=%lu Hz\n",
      IMU_I2C_CONTROLLER,
      IMU_SDA_PIN,
      IMU_SCL_PIN,
      (unsigned long)IMU_I2C_FREQUENCY
    );

    if (
      !imuWire.begin(
        IMU_SDA_PIN,
        IMU_SCL_PIN,
        IMU_I2C_FREQUENCY
      )
    ) {

      strlcpy(
        imuDiagnostic,
        "I2C_BEGIN_FAILED",
        sizeof(imuDiagnostic)
      );

      Serial.println(
        "# IMU ERROR: I2C begin failed"
      );

      return false;
    }

    imuWire.setTimeOut(50);

    imuBusStarted = true;

    Serial.println(
      "# IMU: I2C bus started successfully"
    );
  }

  imuScanBus();

  if (imuAddress == 0) {

    strlcpy(
      imuDiagnostic,
      "I2C_DEVICE_NOT_FOUND",
      sizeof(imuDiagnostic)
    );

    return false;
  }

  uint8_t whoAmI = 0;

  if (!imuReadRegisters(0x75, &whoAmI, 1)) {

    strlcpy(
      imuDiagnostic,
      "WHO_AM_I_READ_FAILED",
      sizeof(imuDiagnostic)
    );

    Serial.printf(
      "# IMU ERROR: WHO_AM_I register read failed, I2C error=%u bytes=%u\n",
      imuLastI2CError,
      imuLastBytesReceived
    );

    return false;
  }

  Serial.printf(
    "# IMU: WHO_AM_I = 0x%02X\n",
    whoAmI
  );

  if (whoAmI != 0x68) {

    strlcpy(
      imuDiagnostic,
      "WHO_AM_I_FAILED",
      sizeof(imuDiagnostic)
    );

    Serial.println(
      "# IMU ERROR: Unexpected WHO_AM_I value"
    );

    return false;
  }

  if (!imuWriteRegister(0x6B, 0x01)) {

    strlcpy(
      imuDiagnostic,
      "WAKEUP_FAILED",
      sizeof(imuDiagnostic)
    );

    return false;
  }

  delay(100);

  if (!imuWriteRegister(0x1C, 0x08)) {

    strlcpy(
      imuDiagnostic,
      "ACCEL_CONFIG_FAILED",
      sizeof(imuDiagnostic)
    );

    return false;
  }

  if (!imuWriteRegister(0x1B, 0x08)) {

    strlcpy(
      imuDiagnostic,
      "GYRO_CONFIG_FAILED",
      sizeof(imuDiagnostic)
    );

    return false;
  }

  imuConsecutiveReadFailures = 0;

  strlcpy(
    imuDiagnostic,
    "CONNECTED",
    sizeof(imuDiagnostic)
  );

  Serial.println(
    "# IMU: MPU6050 initialized successfully"
  );

  return true;
}

// ---------------- Public Functions ----------------

static void imuStart() {

  imuLastAttemptMs = millis();

  imuReady = imuBegin();

  if (!imuReady) {

    Serial.printf(
      "# IMU: Initialization failed: %s\n",
      imuDiagnostic
    );
  }
}

static void imuTick() {

  uint32_t now = millis();

  if (
    imuCalState == CAL_RUNNING &&
    (uint32_t)(now - imuCalStartMs) > IMU_CAL_TIMEOUT_MS
  ) {

    imuCalState = CAL_FAILED;

    Serial.println("# IMU: Calibration timed out");
  }

  if (!imuReady) {

    if (
      (uint32_t)(now - imuLastAttemptMs) >=
      IMU_RETRY_PERIOD_MS
    ) {

      imuLastAttemptMs = now;

      Serial.println(
        "# IMU: Retrying initialization..."
      );

      imuReady = imuBegin();

      if (!imuReady) {

        Serial.printf(
          "# IMU: Retry failed: %s\n",
          imuDiagnostic
        );
      }
    }

    return;
  }

  if (
    (uint32_t)(now - imuLastSampleMs) <
    IMU_SAMPLE_PERIOD_MS
  ) {
    return;
  }

  imuLastSampleMs = now;

  uint8_t data[14];

  if (!imuReadRegisters(0x3B, data, sizeof(data))) {

    imuConsecutiveReadFailures++;

    if (
      imuConsecutiveReadFailures >=
      IMU_MAX_READ_FAILURES
    ) {

      imuReady = false;

      imuLastAttemptMs = now;

      strlcpy(
        imuDiagnostic,
        "I2C_READ_FAILED",
        sizeof(imuDiagnostic)
      );

      Serial.println(
        "# IMU ERROR: Too many consecutive read failures"
      );
    }

    return;
  }

  imuConsecutiveReadFailures = 0;

  int16_t ax = imuToInt16(data[0], data[1]);
  int16_t ay = imuToInt16(data[2], data[3]);
  int16_t az = imuToInt16(data[4], data[5]);

  int16_t gx = imuToInt16(data[8], data[9]);
  int16_t gy = imuToInt16(data[10], data[11]);
  int16_t gz = imuToInt16(data[12], data[13]);

  ImuRawSample sample;

  sample.ax = ax / IMU_ACCEL_LSB_PER_G;
  sample.ay = ay / IMU_ACCEL_LSB_PER_G;
  sample.az = az / IMU_ACCEL_LSB_PER_G;

  sample.gx = gx / IMU_GYRO_LSB_PER_DPS;
  sample.gy = gy / IMU_GYRO_LSB_PER_DPS;
  sample.gz = gz / IMU_GYRO_LSB_PER_DPS;

  if (imuCalRequested) {

    imuCalRequested = false;

    imuCalBegin();
  }

  if (imuCalState == CAL_RUNNING) {

    imuCalFeed(
      sample.gx,
      sample.gy,
      sample.gz
    );
  }

  sample.gx -= imuGyroBias[0];
  sample.gy -= imuGyroBias[1];
  sample.gz -= imuGyroBias[2];

  // Calculate tilt relative to the initial orientation.
  imuUpdateOrientation(sample);

  sample.timestampMs = millis();
  sample.valid = true;

  portENTER_CRITICAL(&imuMux);

  sample.sequenceId = imuLatest.sequenceId + 1;

  imuLatest = sample;

  portEXIT_CRITICAL(&imuMux);
}

// ---------------- JSON Endpoint ----------------

static void imuMakeJson(
  char *buffer,
  size_t capacity,
  const char *deviceId
) {

  ImuRawSample s;

  portENTER_CRITICAL(&imuMux);

  s = imuLatest;

  portEXIT_CRITICAL(&imuMux);

  uint32_t age = millis() - s.timestampMs;

  bool fresh =
    s.valid &&
    age <= IMU_STALE_MS;

  const char *status =
    !fresh
      ? (imuReady ? "UNKNOWN" : "DISCONNECTED")
      : (imuCalState == CAL_RUNNING ? "CALIBRATING" : "OK");

  float quality = fresh ? 1.0f : 0.0f;

  char values[384];

  if (fresh) {

    snprintf(
      values,
      sizeof(values),
      "\"sample_age_ms\":%lu,\"sample_valid\":true,"
      "\"ax_g\":%.5f,\"ay_g\":%.5f,\"az_g\":%.5f,"
      "\"gx_dps\":%.3f,\"gy_dps\":%.3f,\"gz_dps\":%.3f,"
      "\"tilt_angle_deg\":%.2f,\"orientation_ready\":%s",
      (unsigned long)age,
      (double)s.ax,
      (double)s.ay,
      (double)s.az,
      (double)s.gx,
      (double)s.gy,
      (double)s.gz,
      (double)s.tiltAngleDeg,
      s.orientationReady ? "true" : "false"
    );

  } else {

    snprintf(
      values,
      sizeof(values),
      "\"sample_age_ms\":null,\"sample_valid\":false,"
      "\"ax_g\":null,\"ay_g\":null,\"az_g\":null,"
      "\"gx_dps\":null,\"gy_dps\":null,\"gz_dps\":null,"
      "\"tilt_angle_deg\":null,\"orientation_ready\":false"
    );
  }

  snprintf(
    buffer,
    capacity,
    "{\"type\":\"imu_raw\",\"device_id\":\"%s\",\"sequence_id\":%lu,"
    "\"device_timestamp_ms\":%lu,\"device_clock_domain\":\"esp32_millis\",%s,"
    "\"imu_status\":\"%s\",\"imu_quality\":%.2f,\"imu_diagnostic\":\"%s\","
    "\"i2c_address\":\"0x%02X\",\"i2c_devices\":%s,"
    "\"calibration_status\":\"%s\",\"gyro_calibration_std_dps\":%.3f}",
    deviceId,
    (unsigned long)s.sequenceId,
    (unsigned long)s.timestampMs,
    values,
    status,
    (double)quality,
    imuDiagnostic,
    imuAddress,
    imuScanResults,
    imuCalibrationName(),
    (double)imuCalStd
  );
}

// ---------------- Tilt Data Getter ----------------

static bool imuGetTiltData(
  float *angle,
  bool *orientationReady
) {

  ImuRawSample s;

  portENTER_CRITICAL(&imuMux);

  s = imuLatest;

  portEXIT_CRITICAL(&imuMux);

  uint32_t age = millis() - s.timestampMs;

  bool fresh =
    s.valid &&
    age <= IMU_STALE_MS;

  if (angle) {
    *angle = s.tiltAngleDeg;
  }

  if (orientationReady) {
    *orientationReady = s.orientationReady;
  }

  return fresh;
}