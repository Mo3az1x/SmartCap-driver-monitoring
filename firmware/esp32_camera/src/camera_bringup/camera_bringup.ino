#include "esp_camera.h"
#include "../../include/camera_config.h"

void setup() {
  Serial.begin(115200);
  delay(2000);

  if (psramFound()) {
    Serial.printf("PSRAM OK: %u bytes\n", (unsigned)ESP.getPsramSize());
  } else {
    Serial.println("PSRAM NOT found - set PSRAM to OPI PSRAM in Tools");
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = CAM_XCLK_HZ;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = CAM_FRAME_SIZE;
  config.jpeg_quality = CAM_JPEG_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init FAILED: 0x%x\n", err);
    return;
  }
  Serial.println("Camera init OK");
}

void loop() {
  static unsigned long windowStart = millis();
  static int frames = 0;
  static size_t bytes = 0;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed");
    delay(100);
    return;
  }
  frames++;
  bytes += fb->len;
  esp_camera_fb_return(fb);

  unsigned long now = millis();
  if (now - windowStart >= 2000) {
    float sec = (now - windowStart) / 1000.0;
    Serial.printf("FPS: %.1f | avg frame: %u bytes\n", frames / sec, (unsigned)(bytes / frames));
    frames = 0;
    bytes = 0;
    windowStart = now;
  }
}