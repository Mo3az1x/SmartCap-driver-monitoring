
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/camera_config.h"

// =====================================================
// SmartCap - ESP32-S3-CAM N16R8
// Live stream + digital zoom + image controls
// =====================================================

const char* AP_SSID = "SmartCap-CAM";
const char* AP_PASSWORD = "smartcap123";

static httpd_handle_t control_server = NULL;
static httpd_handle_t stream_server = NULL;

static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=smartcapframe";

static const char* STREAM_BOUNDARY = "\r\n--smartcapframe\r\n";

static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n\r\n";

// Use the defaults already defined in camera_config.h.
const framesize_t DEFAULT_FRAME_SIZE = CAM_FRAME_SIZE;
const int DEFAULT_QUALITY = CAM_JPEG_QUALITY;

// -----------------------------------------------------
// Resolution helper
// -----------------------------------------------------

bool getFrameDimensions(framesize_t frameSize, int& width, int& height) {
  switch (frameSize) {
    case FRAMESIZE_QVGA:
      width = 320;
      height = 240;
      return true;

    case FRAMESIZE_VGA:
      width = 640;
      height = 480;
      return true;

    case FRAMESIZE_SVGA:
      width = 800;
      height = 600;
      return true;

    default:
      width = 0;
      height = 0;
      return false;
  }
}

// -----------------------------------------------------
// Initialize camera using camera_config.h pin names
// -----------------------------------------------------

bool initCamera() {
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  // Camera data pins: D0 = Y2 ... D7 = Y9
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
  config.frame_size = DEFAULT_FRAME_SIZE;
  config.jpeg_quality = DEFAULT_QUALITY;

  if (psramFound()) {
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
    Serial.println("PSRAM detected.");
  } else {
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.frame_size = FRAMESIZE_QVGA;
    Serial.println("Warning: PSRAM not detected; using QVGA.");
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor == NULL) {
    Serial.println("Could not get camera sensor.");
    return false;
  }

  sensor->set_framesize(sensor, config.frame_size);
  sensor->set_quality(sensor, DEFAULT_QUALITY);
  sensor->set_brightness(sensor, 0);
  sensor->set_contrast(sensor, 0);
  sensor->set_saturation(sensor, 0);
  sensor->set_hmirror(sensor, 0);
  sensor->set_vflip(sensor, 0);

  Serial.println("Camera initialized successfully.");
  return true;
}

// -----------------------------------------------------
// Web control page
// -----------------------------------------------------

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartCap Camera</title>

<style>
* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 16px;
  background: #101820;
  color: #f3f6f9;
  font-family: Arial, sans-serif;
}

main {
  max-width: 1150px;
  margin: auto;
}

h1 { margin-bottom: 6px; }
h2 { font-size: 18px; }

.subtitle, .note { color: #aab8c5; }
.subtitle { margin-bottom: 18px; }

.layout {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 16px;
}

.panel {
  min-width: 0;
  padding: 16px;
  background: #1b2836;
  border: 1px solid #344456;
  border-radius: 14px;
}

.preview {
  width: 100%;
  aspect-ratio: 4 / 3;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  background: #000;
  border-radius: 10px;
}

#cameraImage {
  width: 100%;
  height: 100%;
  object-fit: contain;
  transform: scale(1);
  transform-origin: center;
}

.control { margin: 19px 0; }

.control label {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}

input[type=range], select { width: 100%; }
input[type=range] { accent-color: #36c99a; }

select, button {
  padding: 10px;
  color: white;
  background: #26394b;
  border: 1px solid #465b70;
  border-radius: 8px;
}

button { cursor: pointer; }
button:hover { background: #354d63; }

.primary {
  background: #087f67;
  border-color: #087f67;
}

.row {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-top: 13px;
}

.row button { flex: 1; }
.row input { flex: 1; min-width: 0; }

.value {
  min-width: 48px;
  color: #6ee7b7;
  text-align: right;
  font-weight: bold;
}

.note {
  font-size: 12px;
  line-height: 1.5;
}

#status {
  margin-top: 12px;
  font-size: 13px;
  color: #b8c7d6;
  line-height: 1.7;
  overflow-wrap: anywhere;
}

@media (max-width: 760px) {
  .layout { grid-template-columns: 1fr; }
  body { padding: 10px; }
}
</style>
</head>

<body>
<main>
  <h1>SmartCap Camera</h1>
  <div class="subtitle">Live monitoring and camera settings</div>

  <div class="layout">

    <section class="panel">
      <h2>Live Preview</h2>

      <div class="preview">
        <img id="cameraImage" alt="Camera stream">
      </div>

      <div class="row">
        <button type="button" onclick="changeZoom(-0.25)">−</button>
        <input id="zoom" type="range" min="0.5" max="3"
               step="0.25" value="1" oninput="applyZoom()">
        <button type="button" onclick="changeZoom(0.25)">+</button>
        <span id="zoomValue" class="value">1.00x</span>
      </div>

      <div class="row">
        <button type="button" onclick="resetZoom()">Reset Zoom</button>
        <button type="button" class="primary" onclick="reloadStream()">
          Reconnect Stream
        </button>
      </div>

      <p class="note">
        Digital zoom changes the displayed image only. It does not
        change the lens or increase the camera's optical zoom.
      </p>

      <div id="status">Connecting to camera...</div>
    </section>

    <section class="panel">
      <h2>Camera Settings</h2>

      <div class="control">
        <label for="resolution">Resolution</label>
        <select id="resolution" onchange="sendControl('framesize', this.value)">
          <option value="5">QVGA - 320 x 240</option>
          <option value="8" selected>VGA - 640 x 480</option>
          <option value="9">SVGA - 800 x 600</option>
        </select>
      </div>

      <div class="control">
        <label for="quality">
          JPEG Quality <span id="qualityValue" class="value">12</span>
        </label>
        <input id="quality" type="range" min="10" max="40" value="12"
               oninput="showValue('quality','qualityValue')"
               onchange="sendControl('quality',this.value)">
        <p class="note">Lower number means higher JPEG quality.</p>
      </div>

      <div class="control">
        <label for="brightness">
          Brightness <span id="brightnessValue" class="value">0</span>
        </label>
        <input id="brightness" type="range" min="-2" max="2" step="1" value="0"
               oninput="showValue('brightness','brightnessValue')"
               onchange="sendControl('brightness',this.value)">
      </div>

      <div class="control">
        <label for="contrast">
          Contrast <span id="contrastValue" class="value">0</span>
        </label>
        <input id="contrast" type="range" min="-2" max="2" step="1" value="0"
               oninput="showValue('contrast','contrastValue')"
               onchange="sendControl('contrast',this.value)">
      </div>

      <div class="control">
        <label for="saturation">
          Saturation <span id="saturationValue" class="value">0</span>
        </label>
        <input id="saturation" type="range" min="-2" max="2" step="1" value="0"
               oninput="showValue('saturation','saturationValue')"
               onchange="sendControl('saturation',this.value)">
      </div>

      <div class="row">
        <button type="button" onclick="toggleFlip('hmirror')">
          Horizontal Flip
        </button>
        <button type="button" onclick="toggleFlip('vflip')">
          Vertical Flip
        </button>
      </div>

      <div class="row">
        <button type="button" class="primary" onclick="resetSettings()">
          Reset All Settings
        </button>
      </div>
    </section>

  </div>
</main>

<script>
const image = document.getElementById('cameraImage');
const zoomSlider = document.getElementById('zoom');

let hmirror = 0;
let vflip = 0;

function reloadStream() {
  image.src = 'http://' + location.hostname +
              ':81/stream?t=' + Date.now();
}

function applyZoom() {
  const zoom = Number(zoomSlider.value);
  image.style.transform = 'scale(' + zoom + ')';
  document.getElementById('zoomValue').textContent =
    zoom.toFixed(2) + 'x';
}

function changeZoom(amount) {
  let zoom = Number(zoomSlider.value) + amount;
  zoom = Math.max(0.5, Math.min(3, zoom));
  zoomSlider.value = zoom;
  applyZoom();
}

function resetZoom() {
  zoomSlider.value = 1;
  applyZoom();
}

function showValue(inputId, labelId) {
  document.getElementById(labelId).textContent =
    document.getElementById(inputId).value;
}

async function sendControl(variable, value) {
  try {
    const response = await fetch(
      '/control?var=' + encodeURIComponent(variable) +
      '&val=' + encodeURIComponent(value)
    );

    if (!response.ok) {
      alert('Setting failed: ' + await response.text());
    }

    await updateStatus();
  } catch (error) {
    document.getElementById('status').textContent =
      'Control error: ' + error.message;
  }
}

async function toggleFlip(variable) {
  if (variable === 'hmirror') {
    hmirror = 1 - hmirror;
    await sendControl(variable, hmirror);
  } else {
    vflip = 1 - vflip;
    await sendControl(variable, vflip);
  }
}

async function resetSettings() {
  if (!confirm('Restore default camera settings?')) return;

  try {
    const response = await fetch('/reset');

    if (!response.ok) {
      alert('Reset failed: ' + await response.text());
      return;
    }

    resetZoom();
    await updateStatus();
  } catch (error) {
    alert('Reset error: ' + error.message);
  }
}

async function updateStatus() {
  try {
    const response = await fetch('/status');

    if (!response.ok) throw new Error('Status unavailable');

    const data = await response.json();

    document.getElementById('status').textContent =
      'Camera connected | Resolution: ' + data.width + ' x ' +
      data.height + ' | JPEG quality: ' + data.quality;

    document.getElementById('resolution').value =
      String(data.framesize);

    ['quality', 'brightness', 'contrast', 'saturation'].forEach(id => {
      document.getElementById(id).value = data[id];
      showValue(id, id + 'Value');
    });

    hmirror = data.hmirror;
    vflip = data.vflip;

  } catch (error) {
    document.getElementById('status').textContent =
      'Unable to read camera status.';
  }
}

reloadStream();
updateStatus();
setInterval(updateStatus, 5000);
</script>
</body>
</html>
)rawliteral";

// -----------------------------------------------------
// Page handler
// -----------------------------------------------------

static esp_err_t indexHandler(httpd_req_t* req) {
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

// -----------------------------------------------------
// Apply a camera setting
// -----------------------------------------------------

static esp_err_t controlHandler(httpd_req_t* req) {
  char query[100];
  char variable[32] = {0};
  char valueText[16] = {0};

  if (httpd_req_get_url_query_len(req) >= sizeof(query)) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Query too long");
    return ESP_FAIL;
  }

  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
      httpd_query_key_value(query, "var", variable, sizeof(variable)) != ESP_OK ||
      httpd_query_key_value(query, "val", valueText, sizeof(valueText)) != ESP_OK) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing var or val");
    return ESP_FAIL;
  }

  char* end = NULL;
  long value = strtol(valueText, &end, 10);

  if (end == valueText || *end != '\0') {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid value");
    return ESP_FAIL;
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor == NULL) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                        "Sensor unavailable");
    return ESP_FAIL;
  }

  int result = -1;

  if (strcmp(variable, "framesize") == 0) {
    if (value == FRAMESIZE_QVGA ||
        value == FRAMESIZE_VGA ||
        value == FRAMESIZE_SVGA) {
      result = sensor->set_framesize(sensor, (framesize_t)value);
    } else {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid resolution");
      return ESP_FAIL;
    }

  } else if (strcmp(variable, "quality") == 0) {
    if (value < 10 || value > 40) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Quality must be 10-40");
      return ESP_FAIL;
    }
    result = sensor->set_quality(sensor, value);

  } else if (strcmp(variable, "brightness") == 0) {
    if (value < -2 || value > 2) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Brightness must be -2 to 2");
      return ESP_FAIL;
    }
    result = sensor->set_brightness(sensor, value);

  } else if (strcmp(variable, "contrast") == 0) {
    if (value < -2 || value > 2) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Contrast must be -2 to 2");
      return ESP_FAIL;
    }
    result = sensor->set_contrast(sensor, value);

  } else if (strcmp(variable, "saturation") == 0) {
    if (value < -2 || value > 2) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Saturation must be -2 to 2");
      return ESP_FAIL;
    }
    result = sensor->set_saturation(sensor, value);

  } else if (strcmp(variable, "hmirror") == 0) {
    if (value < 0 || value > 1) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Value must be 0 or 1");
      return ESP_FAIL;
    }
    result = sensor->set_hmirror(sensor, value);

  } else if (strcmp(variable, "vflip") == 0) {
    if (value < 0 || value > 1) {
      httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Value must be 0 or 1");
      return ESP_FAIL;
    }
    result = sensor->set_vflip(sensor, value);

  } else {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Unknown setting");
    return ESP_FAIL;
  }

  if (result != 0) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                        "Sensor rejected setting");
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_sendstr(req, "OK");
}

// -----------------------------------------------------
// Reset sensor settings
// -----------------------------------------------------

static esp_err_t resetHandler(httpd_req_t* req) {
  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor == NULL) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                        "Sensor unavailable");
    return ESP_FAIL;
  }

  int result = 0;

  result |= sensor->set_framesize(sensor, DEFAULT_FRAME_SIZE);
  result |= sensor->set_quality(sensor, DEFAULT_QUALITY);
  result |= sensor->set_brightness(sensor, 0);
  result |= sensor->set_contrast(sensor, 0);
  result |= sensor->set_saturation(sensor, 0);
  result |= sensor->set_hmirror(sensor, 0);
  result |= sensor->set_vflip(sensor, 0);

  if (result != 0) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                        "Some settings could not be reset");
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_sendstr(req, "Settings reset");
}

// -----------------------------------------------------
// Current sensor status
// Access status fields directly; no sensor_status_t alias
// -----------------------------------------------------

static esp_err_t statusHandler(httpd_req_t* req) {
  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor == NULL) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                        "Sensor unavailable");
    return ESP_FAIL;
  }

  int width = 0;
  int height = 0;

  getFrameDimensions(sensor->status.framesize, width, height);

  char json[256];

  snprintf(json, sizeof(json),
    "{\"framesize\":%d,\"quality\":%d,"
    "\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
    "\"hmirror\":%d,\"vflip\":%d,\"width\":%d,\"height\":%d}",
    (int)sensor->status.framesize,
    sensor->status.quality,
    sensor->status.brightness,
    sensor->status.contrast,
    sensor->status.saturation,
    sensor->status.hmirror,
    sensor->status.vflip,
    width,
    height
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

// -----------------------------------------------------
// MJPEG stream handler
// -----------------------------------------------------

static esp_err_t streamHandler(httpd_req_t* req) {
  esp_err_t result = ESP_OK;
  char partHeader[96];

  httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    camera_fb_t* fb = esp_camera_fb_get();

    if (fb == NULL) {
      Serial.println("Camera frame capture failed");
      result = ESP_FAIL;
      break;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      esp_camera_fb_return(fb);
      result = ESP_FAIL;
      break;
    }

    result = httpd_resp_send_chunk(
      req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));

    if (result == ESP_OK) {
      size_t headerLength = snprintf(
        partHeader, sizeof(partHeader), STREAM_PART, fb->len);

      result = httpd_resp_send_chunk(req, partHeader, headerLength);
    }

    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
        req, (const char*)fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);

    if (result != ESP_OK) {
      break;
    }
  }

  return result;
}

// -----------------------------------------------------
// Start control and stream HTTP servers
// -----------------------------------------------------

void startServers() {
  httpd_config_t controlConfig = HTTPD_DEFAULT_CONFIG();

  controlConfig.server_port = 80;
  controlConfig.ctrl_port = 32768;
  controlConfig.max_uri_handlers = 8;

  if (httpd_start(&control_server, &controlConfig) == ESP_OK) {
    httpd_uri_t indexUri = {
      .uri = "/",
      .method = HTTP_GET,
      .handler = indexHandler,
      .user_ctx = NULL
    };

    httpd_uri_t controlUri = {
      .uri = "/control",
      .method = HTTP_GET,
      .handler = controlHandler,
      .user_ctx = NULL
    };

    httpd_uri_t statusUri = {
      .uri = "/status",
      .method = HTTP_GET,
      .handler = statusHandler,
      .user_ctx = NULL
    };

    httpd_uri_t resetUri = {
      .uri = "/reset",
      .method = HTTP_GET,
      .handler = resetHandler,
      .user_ctx = NULL
    };

    httpd_register_uri_handler(control_server, &indexUri);
    httpd_register_uri_handler(control_server, &controlUri);
    httpd_register_uri_handler(control_server, &statusUri);
    httpd_register_uri_handler(control_server, &resetUri);

    Serial.println("Control server started on port 80");
  } else {
    Serial.println("Failed to start control server");
  }

  httpd_config_t streamConfig = HTTPD_DEFAULT_CONFIG();

  streamConfig.server_port = 81;
  streamConfig.ctrl_port = 32769;
  streamConfig.max_uri_handlers = 2;

  if (httpd_start(&stream_server, &streamConfig) == ESP_OK) {
    httpd_uri_t streamUri = {
      .uri = "/stream",
      .method = HTTP_GET,
      .handler = streamHandler,
      .user_ctx = NULL
    };

    httpd_register_uri_handler(stream_server, &streamUri);

    Serial.println("Stream server started on port 81");
  } else {
    Serial.println("Failed to start stream server");
  }
}

// -----------------------------------------------------
// Setup
// -----------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("Starting SmartCap Camera...");

  if (!initCamera()) {
    Serial.println("Camera initialization failed.");
    return;
  }

  WiFi.mode(WIFI_AP);

  if (!WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    Serial.println("Failed to start Wi-Fi access point.");
    return;
  }

  delay(500);

  IPAddress ip = WiFi.softAPIP();

  Serial.println();
  Serial.println("Wi-Fi AP started");
  Serial.print("Network: ");
  Serial.println(AP_SSID);
  Serial.print("Password: ");
  Serial.println(AP_PASSWORD);
  Serial.print("Control page: http://");
  Serial.println(ip);
  Serial.print("Stream URL: http://");
  Serial.print(ip);
  Serial.println(":81/stream");

  startServers();
}

// -----------------------------------------------------
// Loop
// -----------------------------------------------------

void loop() {
  delay(1000);
}
