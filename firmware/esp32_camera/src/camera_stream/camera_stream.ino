// SmartCap camera node - lean version (protocol only, no settings UI)
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "../../include/camera_config.h"
#include "../../include/camera_protocol.h"

const char* AP_SSID = "SmartCap-CAM";
const char* AP_PASS = "smartcap123";   // test only, 8+ characters

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART =
  "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Frame-Id: %lu\r\nX-Timestamp-Ms: %lu\r\n\r\n";

httpd_handle_t server_httpd = NULL;
bool cameraOk = false;
uint32_t frameCounter = 0;
volatile uint32_t lastFrameTs = 0;
volatile uint16_t lastW = 0, lastH = 0;
volatile float streamFps = 0;
volatile bool streamBusy = false;

static inline uint32_t nextFrameId() {
  return __atomic_add_fetch(&frameCounter, 1, __ATOMIC_RELAXED);
}

// ---------------- Minimal page ----------------
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartCap Camera Node</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;margin:10px}
img{max-width:100%;background:#000;border-radius:6px}
pre{background:#000;color:#9f9;padding:8px;font-size:12px;overflow:auto}
a{color:#8ab4ff}
button{background:#2563eb;color:#fff;border:0;padding:8px 12px;border-radius:5px;cursor:pointer}
</style></head><body>
<h3>SmartCap Camera Node</h3>
<button onclick="toggleVideo()">Show / hide video</button>
<p><img id="v"></p>
<pre id="m">waiting for /meta ...</pre>
<p><a href="/stream">/stream</a> | <a href="/capture">/capture</a> | <a href="/meta">/meta</a></p>
<p>Only one client can read /stream at a time. Hide the video before a script connects.</p>
<script>
let on=false;
function toggleVideo(){
 const im=document.getElementById('v');
 if(on){im.removeAttribute('src');on=false;}
 else{setTimeout(()=>{im.src='/stream?t='+Date.now();},1500);on=true;}
}
setInterval(()=>fetch('/meta').then(r=>r.text()).then(t=>{document.getElementById('m').textContent=t;}),1000);
</script></body></html>
)rawliteral";

// ---------------- Handlers ----------------
static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { httpd_resp_send_500(req); return ESP_FAIL; }

  uint32_t ts  = millis();
  uint32_t fid = nextFrameId();
  lastFrameTs = ts;
  lastW = fb->width;
  lastH = fb->height;

  char idbuf[16], tsbuf[16];
  snprintf(idbuf, sizeof(idbuf), "%lu", (unsigned long)fid);
  snprintf(tsbuf, sizeof(tsbuf), "%lu", (unsigned long)ts);

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "X-Frame-Id", idbuf);
  httpd_resp_set_hdr(req, "X-Timestamp-Ms", tsbuf);
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// Interface spec v1, section 8: camera_frame metadata
// received_timestamp_ms is added by the laptop, not by the camera.
static esp_err_t meta_handler(httpd_req_t *req) {
  sensor_t *s = esp_camera_sensor_get();
  const char* status;
  float quality;
  float fps = streamFps;
  cameraHealth(cameraOk, frameCounter, lastFrameTs, fps, &status, &quality);

  char buf[384];
  snprintf(buf, sizeof(buf),
    "{\"type\":\"camera_frame\",\"device_id\":\"%s\",\"frame_id\":%lu,"
    "\"device_timestamp_ms\":%lu,\"width\":%u,\"height\":%u,\"jpeg_quality\":%u,"
    "\"actual_fps\":%.1f,\"camera_status\":\"%s\",\"camera_quality\":%.2f}",
    DEVICE_ID, (unsigned long)frameCounter, (unsigned long)lastFrameTs,
    (unsigned)lastW, (unsigned)lastH, s ? (unsigned)s->status.quality : 0u,
    (double)fps, status, (double)quality);

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, buf, HTTPD_RESP_USE_STRLEN);
}

// /control?var=framesize&val=8   (5=QVGA, 8=VGA, 9=SVGA, 10=XGA)
// /control?var=quality&val=12   /control?var=hmirror&val=1   /control?var=vflip&val=1
static esp_err_t control_handler(httpd_req_t *req) {
  char query[128], variable[32], value[32];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
      httpd_query_key_value(query, "var", variable, sizeof(variable)) != ESP_OK ||
      httpd_query_key_value(query, "val", value, sizeof(value)) != ESP_OK) {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }
  sensor_t *s = esp_camera_sensor_get();
  if (!s) { httpd_resp_send_500(req); return ESP_FAIL; }

  int val = atoi(value);
  int res = -1;
  if      (!strcmp(variable, "framesize")) res = s->set_framesize(s, (framesize_t)val);
  else if (!strcmp(variable, "quality"))   res = s->set_quality(s, val);
  else if (!strcmp(variable, "hmirror"))   res = s->set_hmirror(s, val);
  else if (!strcmp(variable, "vflip"))     res = s->set_vflip(s, val);

  Serial.printf("SET %s=%d -> %s\n", variable, val, res == 0 ? "ok" : "failed");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, res == 0 ? "OK" : "FAIL", HTTPD_RESP_USE_STRLEN);
}

// ---------------- Stream (runs in its own task, so the server stays free) ----------------
static void stream_task(void *arg) {
  httpd_req_t *req = (httpd_req_t *)arg;
  httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  char part_buf[160];
  uint32_t windowStart = millis();
  int frames = 0;
  esp_err_t res = ESP_OK;

  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) break;

    uint32_t ts  = millis();
    uint32_t fid = nextFrameId();
    lastFrameTs = ts;
    lastW = fb->width;
    lastH = fb->height;

    size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART,
                           (unsigned)fb->len, (unsigned long)fid, (unsigned long)ts);
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;

    frames++;
    uint32_t now = millis();
    if (now - windowStart >= 1000) {
      streamFps = frames * 1000.0f / (now - windowStart);
      Serial.printf("Stream FPS: %.1f\n", streamFps);
      frames = 0;
      windowStart = now;
    }
    vTaskDelay(1);   // let Wi-Fi and the HTTP server run
  }

  streamFps = 0;
  streamBusy = false;
  httpd_req_async_handler_complete(req);
  vTaskDelete(NULL);
}

static esp_err_t stream_handler(httpd_req_t *req) {
  if (streamBusy) {
    httpd_resp_set_status(req, "503 Service Unavailable");
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_send(req, "Stream busy: one client at a time", HTTPD_RESP_USE_STRLEN);
  }
  httpd_req_t *async_req = NULL;
  if (httpd_req_async_handler_begin(req, &async_req) != ESP_OK) return ESP_FAIL;
  streamBusy = true;
  if (xTaskCreate(stream_task, "stream", 8192, async_req, 5, NULL) != pdPASS) {
    streamBusy = false;
    httpd_req_async_handler_complete(async_req);
    return ESP_FAIL;
  }
  return ESP_OK;
}

static esp_err_t not_found_handler(httpd_req_t *req, httpd_err_code_t err) {
  httpd_resp_set_status(req, "404 Not Found");
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_send(req, "Not found. Valid: /  /stream  /capture  /meta  /control?var=..&val=..",
                  HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

void startServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 8;
  config.lru_purge_enable = true;

  httpd_uri_t index_uri   = { "/",        HTTP_GET, index_handler,   NULL };
  httpd_uri_t capture_uri = { "/capture", HTTP_GET, capture_handler, NULL };
  httpd_uri_t meta_uri    = { "/meta",    HTTP_GET, meta_handler,    NULL };
  httpd_uri_t control_uri = { "/control", HTTP_GET, control_handler, NULL };
  httpd_uri_t stream_uri  = { "/stream",  HTTP_GET, stream_handler,  NULL };

  if (httpd_start(&server_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(server_httpd, &index_uri);
    httpd_register_uri_handler(server_httpd, &capture_uri);
    httpd_register_uri_handler(server_httpd, &meta_uri);
    httpd_register_uri_handler(server_httpd, &control_uri);
    httpd_register_uri_handler(server_httpd, &stream_uri);
    httpd_register_err_handler(server_httpd, HTTPD_404_NOT_FOUND, not_found_handler);
  }
}

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

  cameraOk = (esp_camera_init(&config) == ESP_OK);
  Serial.println(cameraOk ? "Camera init OK" : "Camera init FAILED (check ribbon cable and PSRAM setting)");

  // The AP and server start even if the camera failed, so /meta can report DISCONNECTED
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.setSleep(false);
  Serial.printf("AP started: %s\n", AP_SSID);
  Serial.print("Open: http://");
  Serial.println(WiFi.softAPIP());

  startServer();
}

void loop() {
  delay(1000);
}