
#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <string.h>

// ---------------- IMU Configuration ----------------

static constexpr int IMU_SDA_PIN = 14;
static constexpr int IMU_SCL_PIN = 47;
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

// Diagnostic information
static uint8_t imuLastI2CError = 0;
static uint8_t imuLastBytesReceived = 0;

// ---------------- Low-Level I2C ----------------

static bool imuWriteRegister(uint8_t reg, uint8_t value) {
  imuWire.beginTransmission(imuAddress);

  imuWire.write(reg);
  imuWire.write(value);

  imuLastI2CError = imuWire.endTransmission();

  return imuLastI2CError == 0;
}

static bool imuReadRegisters(
  uint8_t reg,
  uint8_t *data,
  size_t length
) {
  // Select register.
  imuWire.beginTransmission(imuAddress);
  imuWire.write(reg);

  imuLastI2CError = imuWire.endTransmission(false);

  if (imuLastI2CError != 0) {
    imuLastBytesReceived = 0;
    return false;
  }

  // Request data.
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

  // Read all bytes.
  for (size_t i = 0; i < length; i++) {
    if (!imuWire.available()) {
      return false;
    }

    data[i] = imuWire.read();
  }

  return true;
}

static int16_t imuToInt16(uint8_t high, uint8_t low) {
  return (int16_t)(((uint16_t)high << 8) | low);
}

// ---------------- I2C Scanner ----------------

static void imuScanBus() {
  char devices[96] = "[";
  size_t used = 1;

  bool first = true;

  imuAddress = 0;

  for (uint8_t address = 1; address < 127; address++) {
    imuWire.beginTransmission(address);

    if (imuWire.endTransmission() == 0) {
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
        memcpy(devices + used, item, itemLength);

        used += itemLength;
        devices[used] = '\0';
      }

      first = false;

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
}

// ---------------- Initialization ----------------

static bool imuBegin() {

  // Start I2C bus only once.
  if (!imuBusStarted) {

    if (!imuWire.begin(
      IMU_SDA_PIN,
      IMU_SCL_PIN,
      IMU_I2C_FREQUENCY
    )) {
      strlcpy(
        imuDiagnostic,
        "I2C_BEGIN_FAILED",
        sizeof(imuDiagnostic)
      );

      Serial.println("# IMU: I2C initialization failed");

      return false;
    }

    imuWire.setTimeOut(50);

    imuBusStarted = true;

    Serial.println("# IMU: I2C initialized");
  }

  // Scan devices.
  imuScanBus();

  if (imuAddress == 0) {
    strlcpy(
      imuDiagnostic,
      "I2C_DEVICE_NOT_FOUND",
      sizeof(imuDiagnostic)
    );

    Serial.println("# IMU: Device not found");

    return false;
  }

  Serial.printf(
    "# IMU: Device detected at 0x%02X\n",
    imuAddress
  );

  // Check WHO_AM_I register.
  uint8_t whoAmI = 0;

  if (!imuReadRegisters(0x75, &whoAmI, 1)) {
    strlcpy(
      imuDiagnostic,
      "WHO_AM_I_READ_FAILED",
      sizeof(imuDiagnostic)
    );

    Serial.println("# IMU: WHO_AM_I read failed");

    return false;
  }

  if (whoAmI != 0x68) {
    strlcpy(
      imuDiagnostic,
      "WHO_AM_I_MISMATCH",
      sizeof(imuDiagnostic)
    );

    Serial.printf(
      "# IMU: Unexpected WHO_AM_I: 0x%02X\n",
      whoAmI
    );

    return false;
  }

  // Wake up MPU6050.
  if (!imuWriteRegister(0x6B, 0x00)) {
    strlcpy(
      imuDiagnostic,
      "CONFIG_FAILED",
      sizeof(imuDiagnostic)
    );

    return false;
  }

  delay(100);

  // Accelerometer: +/-4g.
  if (!imuWriteRegister(0x1C, 0x08)) {
    strlcpy(
      imuDiagnostic,
      "ACCEL_CONFIG_FAILED",
      sizeof(imuDiagnostic)
    );

    return false;
  }

  // Gyroscope: +/-500 degrees/second.
  if (!imuWriteRegister(0x1B, 0x08)) {
    strlcpy(
      imuDiagnostic,
      "GYRO_CONFIG_FAILED",
      sizeof(imuDiagnostic)
    );

    return false;
  }

  // Successful initialization.
  imuConsecutiveReadFailures = 0;
  imuLastI2CError = 0;
  imuLastBytesReceived = 0;

  strlcpy(
    imuDiagnostic,
    "CONNECTED",
    sizeof(imuDiagnostic)
  );

  Serial.println("# IMU: Initialization successful");

  return true;
}

// ---------------- Public Functions ----------------

static void imuStart() {

  imuLastAttemptMs = millis();

  imuReady = imuBegin();
}

static void imuTick() {

  uint32_t now = millis();

  // Attempt recovery if sensor is not ready.
  if (!imuReady) {

    if (
      (uint32_t)(now - imuLastAttemptMs) >=
      IMU_RETRY_PERIOD_MS
    ) {
      imuLastAttemptMs = now;

      Serial.println("# IMU: Attempting recovery");

      imuReady = imuBegin();
    }

    return;
  }

  // Sampling interval: 20ms (50Hz).
  if (
    (uint32_t)(now - imuLastSampleMs) <
    IMU_SAMPLE_PERIOD_MS
  ) {
    return;
  }

  imuLastSampleMs = now;

  // Read registers 0x3B through 0x48.
  uint8_t data[14];

  if (!imuReadRegisters(
    0x3B,
    data,
    sizeof(data)
  )) {

    imuConsecutiveReadFailures++;

    Serial.printf(
      "# IMU: Read failed %u/%u | I2C error=%u | bytes=%u\n",
      imuConsecutiveReadFailures,
      IMU_MAX_READ_FAILURES,
      imuLastI2CError,
      imuLastBytesReceived
    );

    // Do not stop after a single failed read.
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

      Serial.println("# IMU: Too many failures, recovery required");
    }

    return;
  }

  // Successful read: reset failure counter.
  imuConsecutiveReadFailures = 0;

  // Parse accelerometer.
  int16_t ax = imuToInt16(data[0], data[1]);
  int16_t ay = imuToInt16(data[2], data[3]);
  int16_t az = imuToInt16(data[4], data[5]);

  // Parse gyroscope.
  int16_t gx = imuToInt16(data[8], data[9]);
  int16_t gy = imuToInt16(data[10], data[11]);
  int16_t gz = imuToInt16(data[12], data[13]);

  // Convert raw values.
  ImuRawSample sample;

  sample.ax = ax / IMU_ACCEL_LSB_PER_G;
  sample.ay = ay / IMU_ACCEL_LSB_PER_G;
  sample.az = az / IMU_ACCEL_LSB_PER_G;

  sample.gx = gx / IMU_GYRO_LSB_PER_DPS;
  sample.gy = gy / IMU_GYRO_LSB_PER_DPS;
  sample.gz = gz / IMU_GYRO_LSB_PER_DPS;

  sample.timestampMs = millis();
  sample.valid = true;

  // Update shared sample safely.
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

  ImuRawSample sample;

  portENTER_CRITICAL(&imuMux);

  sample = imuLatest;

  portEXIT_CRITICAL(&imuMux);

  uint32_t age = millis() - sample.timestampMs;

  bool fresh =
    sample.valid &&
    age <= IMU_STALE_MS;

  const char *status = fresh
    ? "UNCALIBRATED"
    : imuDiagnostic;

  if (fresh) {

    snprintf(
      buffer,
      capacity,

      "{\"type\":\"imu_raw\","
      "\"device_id\":\"%s\","
      "\"sequence_id\":%lu,"
      "\"device_timestamp_ms\":%lu,"
      "\"device_clock_domain\":\"esp32_millis\","
      "\"sample_age_ms\":%lu,"
      "\"sample_valid\":true,"
      "\"ax_g\":%.5f,"
      "\"ay_g\":%.5f,"
      "\"az_g\":%.5f,"
      "\"gx_dps\":%.3f,"
      "\"gy_dps\":%.3f,"
      "\"gz_dps\":%.3f,"
      "\"imu_status\":\"%s\","
      "\"i2c_address\":\"0x%02X\","
      "\"i2c_devices\":%s,"
      "\"calibration_status\":\"NOT_IMPLEMENTED\"}",

      deviceId,
      (unsigned long)sample.sequenceId,
      (unsigned long)sample.timestampMs,
      (unsigned long)age,

      (double)sample.ax,
      (double)sample.ay,
      (double)sample.az,

      (double)sample.gx,
      (double)sample.gy,
      (double)sample.gz,

      status,
      imuAddress,
      imuScanResults
    );

  } else {

    snprintf(
      buffer,
      capacity,

      "{\"type\":\"imu_raw\","
      "\"device_id\":\"%s\","
      "\"sequence_id\":%lu,"
      "\"device_timestamp_ms\":%lu,"
      "\"device_clock_domain\":\"esp32_millis\","
      "\"sample_age_ms\":null,"
      "\"sample_valid\":false,"
      "\"ax_g\":null,"
      "\"ay_g\":null,"
      "\"az_g\":null,"
      "\"gx_dps\":null,"
      "\"gy_dps\":null,"
      "\"gz_dps\":null,"
      "\"imu_status\":\"%s\","
      "\"i2c_address\":\"0x%02X\","
      "\"i2c_devices\":%s,"
      "\"calibration_status\":\"NOT_IMPLEMENTED\"}",

      deviceId,
      (unsigned long)sample.sequenceId,
      (unsigned long)sample.timestampMs,

      status,
      imuAddress,
      imuScanResults
    );
  }
}
