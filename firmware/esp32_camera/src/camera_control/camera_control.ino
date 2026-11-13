// SmartCap: camera tuning + MPU6050 raw Wi-Fi measurements.

#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"

#include "../../include/camera_config.h"
#include "../../include/camera_protocol.h"
#include "imu_inline.h"

const char *AP_SSID = "SmartCap-CAM";
const char *AP_PASS = "smartcap123";

#define PART_BOUNDARY "123456789000000000000987654321"

static const char *STREAM_CONTENT_TYPE =
  "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;

static const char *STREAM_BOUNDARY =
  "\r\n--" PART_BOUNDARY "\r\n";

static const char *STREAM_PART =
  "Content-Type: image/jpeg\r\n"
  "Content-Length: %u\r\n"
  "X-Frame-Id: %lu\r\n"
  "X-Timestamp-Ms: %lu\r\n\r\n";

static httpd_handle_t server_httpd = NULL;
static bool cameraOk = false;

static uint32_t frameCounter = 0;
static uint32_t lastFrameTs = 0;
static uint16_t lastW = 0;
static uint16_t lastH = 0;
static float streamFps = 0;

static bool streamBusy = false;
static portMUX_TYPE cameraMux = portMUX_INITIALIZER_UNLOCKED;

struct CameraSnapshot {
  uint32_t frames;
  uint32_t timestamp;
  uint16_t width;
  uint16_t height;
  float fps;
};

static CameraSnapshot cameraSnapshot() {
  CameraSnapshot out;
  portENTER_CRITICAL(&cameraMux);
  out.frames = frameCounter;
  out.timestamp = lastFrameTs;
  out.width = lastW;
  out.height = lastH;
  out.fps = streamFps;
  portEXIT_CRITICAL(&cameraMux);
  return out;
}

static uint32_t recordFrame(camera_fb_t *fb, uint32_t ts) {
  uint32_t id;
  portENTER_CRITICAL(&cameraMux);
  id = ++frameCounter;
  lastFrameTs = ts;
  lastW = fb->width;
  lastH = fb->height;
  portEXIT_CRITICAL(&cameraMux);
  return id;
}

static void setStreamFps(float fps) {
  portENTER_CRITICAL(&cameraMux);
  streamFps = fps;
  portEXIT_CRITICAL(&cameraMux);
}

// ---------------- Web page ----------------

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartCap Camera + IMU</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;margin:12px}
.wrap{display:flex;flex-wrap:wrap;gap:18px}
.left{flex:1 1 480px}
.right{flex:1 1 320px;max-width:440px}
img{width:100%;background:#000;border-radius:6px}
button{background:#2563eb;color:white;border:0;border-radius:5px;
padding:9px;margin:4px;cursor:pointer}
.row{display:flex;align-items:center;gap:8px;margin:8px 0}
.row label{width:135px;font-size:14px}
.row input[type=range],select{flex:1}
.row span{width:36px;text-align:right}
select{background:#222;color:#eee;padding:5px}
pre,textarea{background:#000;color:#9f9;padding:10px;
overflow:auto;box-sizing:border-box}
textarea{width:100%;height:150px}
h4{color:#8ab4ff;margin-bottom:8px}
.note{color:#bbb;font-size:13px}
</style>
</head>
<body>
<h3>SmartCap Camera + IMU</h3>

<div class="wrap">
<div class="left">
<img id="stream">
<div>Stream FPS: <span id="fps">-</span></div>

<button onclick="toggleVideo()">Show / hide video</button>
<button onclick="window.open('/capture')">Snapshot</button>
<button onclick="window.open('/meta')">Camera metadata</button>

<p class="note">
Only one /stream client. Hide browser video before starting vision_worker.
</p>

<h4>MPU6050</h4>
<button onclick="fetch('/calibrate')">Calibrate IMU (keep still 5 s)</button>
<pre id="imu">Waiting...</pre>

<h4>Camera settings log</h4>
<button onclick="logNow()">Log now</button>
<label><input type="checkbox" id="auto" onchange="toggleAuto()">
Auto log every 2s</label>
<button onclick="copyLog()">Copy log</button>
<button onclick="document.getElementById('log').value=''">Clear</button>
<textarea id="log" readonly></textarea>
</div>

<div class="right" id="ctrl"></div>
</div>

<script>
const SIZES=[
 [5,'QVGA 320x240'],[6,'CIF 400x296'],[8,'VGA 640x480'],
 [9,'SVGA 800x600'],[10,'XGA 1024x768'],[11,'HD 1280x720'],
 [12,'SXGA 1280x1024'],[13,'UXGA 1600x1200']
];
const GAIN=[[0,'2x'],[1,'4x'],[2,'8x'],[3,'16x'],
 [4,'32x'],[5,'64x'],[6,'128x']];
const WB=[[0,'Auto'],[1,'Sunny'],[2,'Cloudy'],[3,'Office'],[4,'Home']];
const FX=[[0,'None'],[1,'Negative'],[2,'Grayscale'],
 [3,'Red tint'],[4,'Green tint'],[5,'Blue tint'],[6,'Sepia']];

const C=[
 ['Size & compression'],
 ['framesize','Resolution','select',SIZES],
 ['quality','JPEG quality','range',10,63],
 ['Image'],
 ['brightness','Brightness','range',-2,2],
 ['contrast','Contrast','range',-2,2],
 ['saturation','Saturation','range',-2,2],
 ['special_effect','Effect','select',FX],
 ['Exposure & gain'],
 ['aec','Auto exposure','check'],
 ['aec2','AEC DSP','check'],
 ['ae_level','AE level','range',-2,2],
 ['aec_value','Exposure value','range',0,1200],
 ['agc','Auto gain','check'],
 ['agc_gain','Gain value','range',0,30],
 ['gainceiling','Gain ceiling','select',GAIN],
 ['White balance'],
 ['awb','Auto WB','check'],
 ['awb_gain','AWB gain','check'],
 ['wb_mode','WB mode','select',WB],
 ['Orientation'],
 ['hmirror','H-mirror','check'],
 ['vflip','V-flip','check'],
 ['Correction'],
 ['lenc','Lens correction','check'],
 ['bpc','Bad pixel','check'],
 ['wpc','White pixel','check'],
 ['raw_gma','Gamma','check'],
 ['dcw','Downsize EN','check'],
 ['colorbar','Color bar','check']
];

let videoOn=false;
let autoTimer=null;

async function send(k,v){
 try{
  const r=await fetch('/control?var='+encodeURIComponent(k)+'&val='+v);
  if(!r.ok) throw new Error(await r.text());
  await load();
 }catch(e){console.error(e);}
}

function build(){
 const root=document.getElementById('ctrl');
 C.forEach(c=>{
  if(c.length===1){
   const title=document.createElement('h4');
   title.textContent=c[0];
   root.appendChild(title);
   return;
  }

  const [k,label,type]=c;
  const row=document.createElement('div');
  row.className='row';

  const lab=document.createElement('label');
  lab.textContent=label;
  row.appendChild(lab);

  const el=document.createElement(type==='select'?'select':'input');
  el.id=k;

  if(type==='range'){
   el.type='range';el.min=c[3];el.max=c[4];
  }else if(type==='check'){
   el.type='checkbox';
  }else{
   c[3].forEach(([value,text])=>{
    const option=document.createElement('option');
    option.value=value;option.textContent=text;
    el.appendChild(option);
   });
  }

  row.appendChild(el);

  if(type==='range'){
   const value=document.createElement('span');
   value.id='v_'+k;
   row.appendChild(value);
   el.oninput=()=>value.textContent=el.value;
  }

  el.onchange=()=>send(k,type==='check'?(el.checked?1:0):el.value);
  root.appendChild(row);
 });
}

async function getStatus(){
 const r=await fetch('/status',{cache:'no-store'});
 if(!r.ok) throw new Error('Camera status unavailable');
 return r.json();
}

async function load(){
 try{
  const j=await getStatus();
  C.forEach(c=>{
   if(c.length===1)return;
   const [k,,type]=c;
   const el=document.getElementById(k);
   if(!(k in j))return;
   if(type==='check')el.checked=!!j[k];
   else el.value=j[k];
   if(type==='range')document.getElementById('v_'+k).textContent=j[k];
  });
  document.getElementById('fps').textContent=Number(j.fps).toFixed(1);
 }catch(e){document.getElementById('fps').textContent='unavailable';}
}

async function pollImu(){
 try{
  const r=await fetch('/imu',{cache:'no-store'});
  if(!r.ok)throw new Error('IMU request failed');
  const j=await r.json();
  document.getElementById('imu').textContent=JSON.stringify(j,null,2);
 }catch(e){
  document.getElementById('imu').textContent='IMU request failed';
 }finally{
  setTimeout(pollImu,500);
 }
}

function toggleVideo(){
 const im=document.getElementById('stream');
 if(videoOn){im.removeAttribute('src');videoOn=false;}
 else{im.src='/stream?t='+Date.now();videoOn=true;}
}

const LOG_KEYS=[
 'framesize','quality','fps','brightness','contrast','saturation',
 'aec','aec_value','agc','agc_gain','hmirror','vflip'
];

async function logNow(){
 try{
  const j=await getStatus();
  const t=document.getElementById('log');
  if(!t.value)t.value=['Time',...LOG_KEYS].join('\t')+'\n';
  t.value+=[new Date().toISOString(),...LOG_KEYS.map(k=>j[k])].join('\t')+'\n';
  t.scrollTop=t.scrollHeight;
 }catch(e){console.error(e);}
}

function toggleAuto(){
 clearInterval(autoTimer);
 if(document.getElementById('auto').checked){
  logNow();autoTimer=setInterval(logNow,2000);
 }
}

function copyLog(){
 const t=document.getElementById('log');
 t.select();document.execCommand('copy');
}

build();
load();
setInterval(load,2000);
pollImu();
</script>
</body>
</html>
)rawliteral";

// ---------------- HTTP handlers ----------------

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t imu_handler(httpd_req_t *req) {
  char buf[768];
  imuMakeJson(buf, sizeof(buf), DEVICE_ID);

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  return httpd_resp_send(req, buf, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t calibrate_handler(httpd_req_t *req) {
  imuCalRequested = true;
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, "Calibration requested: keep the cap still for 5 seconds", HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req) {
  if (!cameraOk) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  uint32_t ts = millis();
  char tsbuf[16];
  snprintf(tsbuf, sizeof(tsbuf), "%lu", (unsigned long)ts);

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "X-Timestamp-Ms", tsbuf);
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  esp_err_t result = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return result;
}

static esp_err_t meta_handler(httpd_req_t *req) {
  sensor_t *sensor = cameraOk ? esp_camera_sensor_get() : nullptr;
  CameraSnapshot cam = cameraSnapshot();

  const char *status;
  float quality;
  cameraHealth(cameraOk, cam.frames, cam.timestamp, cam.fps, &status, &quality);

  char buf[512];
  snprintf(buf, sizeof(buf),
    "{\"type\":\"camera_frame\","
    "\"device_id\":\"%s\","
    "\"frame_id\":%lu,"
    "\"device_timestamp_ms\":%lu,"
    "\"width\":%u,\"height\":%u,"
    "\"jpeg_quality\":%u,"
    "\"actual_fps\":%.1f,"
    "\"camera_status\":\"%s\","
    "\"camera_quality\":%.2f}",
    DEVICE_ID,
    (unsigned long)cam.frames,
    (unsigned long)cam.timestamp,
    (unsigned)cam.width,
    (unsigned)cam.height,
    sensor ? (unsigned)sensor->status.quality : 0u,
    (double)cam.fps,
    status,
    (double)quality
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  return httpd_resp_send(req, buf, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t status_handler(httpd_req_t *req) {
  sensor_t *s = cameraOk ? esp_camera_sensor_get() : nullptr;
  if (!s) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  CameraSnapshot cam = cameraSnapshot();
  char buf[768];

  snprintf(buf, sizeof(buf),
    "{\"framesize\":%u,\"quality\":%u,"
    "\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
    "\"special_effect\":%u,\"wb_mode\":%u,"
    "\"awb\":%u,\"awb_gain\":%u,\"aec\":%u,\"aec2\":%u,"
    "\"ae_level\":%d,\"aec_value\":%u,"
    "\"agc\":%u,\"agc_gain\":%u,\"gainceiling\":%u,"
    "\"bpc\":%u,\"wpc\":%u,\"raw_gma\":%u,\"lenc\":%u,"
    "\"hmirror\":%u,\"vflip\":%u,\"dcw\":%u,\"colorbar\":%u,"
    "\"fps\":%.1f}",
    (unsigned)s->status.framesize, (unsigned)s->status.quality,
    (int)s->status.brightness, (int)s->status.contrast, (int)s->status.saturation,
    (unsigned)s->status.special_effect, (unsigned)s->status.wb_mode,
    (unsigned)s->status.awb, (unsigned)s->status.awb_gain, (unsigned)s->status.aec,
    (unsigned)s->status.aec2, (int)s->status.ae_level, (unsigned)s->status.aec_value,
    (unsigned)s->status.agc, (unsigned)s->status.agc_gain, (unsigned)s->status.gainceiling,
    (unsigned)s->status.bpc, (unsigned)s->status.wpc, (unsigned)s->status.raw_gma,
    (unsigned)s->status.lenc, (unsigned)s->status.hmirror, (unsigned)s->status.vflip,
    (unsigned)s->status.dcw, (unsigned)s->status.colorbar, (double)cam.fps
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, buf, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t control_handler(httpd_req_t *req) {
  char query[128];
  char variable[32];
  char value[32];

  if (
    httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
    httpd_query_key_value(query, "var", variable, sizeof(variable)) != ESP_OK ||
    httpd_query_key_value(query, "val", value, sizeof(value)) != ESP_OK
  ) {
    httpd_resp_set_status(req, "400 Bad Request");
    return httpd_resp_send(req, "Expected var and val", HTTPD_RESP_USE_STRLEN);
  }

  sensor_t *s = cameraOk ? esp_camera_sensor_get() : nullptr;
  if (!s) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  char *end = nullptr;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < -32768 || parsed > 32767) {
    httpd_resp_set_status(req, "400 Bad Request");
    return httpd_resp_send(req, "Invalid integer", HTTPD_RESP_USE_STRLEN);
  }

  int val = (int)parsed;
  int result = -1;

  if (!strcmp(variable, "framesize")) {
    if (val >= 0 && val <= (int)FRAMESIZE_QXGA) result = s->set_framesize(s, (framesize_t)val);
  } else if (!strcmp(variable, "quality")) {
    if (val >= 0 && val <= 63) result = s->set_quality(s, val);
  } else if (!strcmp(variable, "brightness")) { result = s->set_brightness(s, val); }
  else if (!strcmp(variable, "contrast")) { result = s->set_contrast(s, val); }
  else if (!strcmp(variable, "saturation")) { result = s->set_saturation(s, val); }
  else if (!strcmp(variable, "special_effect")) { result = s->set_special_effect(s, val); }
  else if (!strcmp(variable, "aec")) { result = s->set_exposure_ctrl(s, val); }
  else if (!strcmp(variable, "aec2")) { result = s->set_aec2(s, val); }
  else if (!strcmp(variable, "ae_level")) { result = s->set_ae_level(s, val); }
  else if (!strcmp(variable, "aec_value")) { result = s->set_aec_value(s, val); }
  else if (!strcmp(variable, "agc")) { result = s->set_gain_ctrl(s, val); }
  else if (!strcmp(variable, "agc_gain")) { result = s->set_agc_gain(s, val); }
  else if (!strcmp(variable, "gainceiling")) { result = s->set_gainceiling(s, (gainceiling_t)val); }
  else if (!strcmp(variable, "awb")) { result = s->set_whitebal(s, val); }
  else if (!strcmp(variable, "awb_gain")) { result = s->set_awb_gain(s, val); }
  else if (!strcmp(variable, "wb_mode")) { result = s->set_wb_mode(s, val); }
  else if (!strcmp(variable, "hmirror")) { result = s->set_hmirror(s, val); }
  else if (!strcmp(variable, "vflip")) { result = s->set_vflip(s, val); }
  else if (!strcmp(variable, "lenc")) { result = s->set_lenc(s, val); }
  else if (!strcmp(variable, "bpc")) { result = s->set_bpc(s, val); }
  else if (!strcmp(variable, "wpc")) { result = s->set_wpc(s, val); }
  else if (!strcmp(variable, "raw_gma")) { result = s->set_raw_gma(s, val); }
  else if (!strcmp(variable, "dcw")) { result = s->set_dcw(s, val); }
  else if (!strcmp(variable, "colorbar")) { result = s->set_colorbar(s, val); }

  Serial.printf("# SET %s=%d -> %s\n", variable, val, result == 0 ? "OK" : "FAIL");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  if (result != 0) httpd_resp_set_status(req, "400 Bad Request");
  return httpd_resp_send(req, result == 0 ? "OK" : "FAIL", HTTPD_RESP_USE_STRLEN);
}

// ---------------- MJPEG stream ----------------

static void stream_task(void *arg) {
  httpd_req_t *req = (httpd_req_t *)arg;

  httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  char part[160];
  uint32_t windowStart = millis();
  uint32_t frames = 0;

  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) break;

    uint32_t ts = millis();
    uint32_t fid = recordFrame(fb, ts);

    int len = snprintf(part, sizeof(part), STREAM_PART, (unsigned)fb->len, (unsigned long)fid, (unsigned long)ts);
    esp_err_t result = ESP_FAIL;

    if (len > 0 && (size_t)len < sizeof(part)) {
      result = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
      if (result == ESP_OK) result = httpd_resp_send_chunk(req, part, len);
      if (result == ESP_OK) result = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);
    if (result != ESP_OK) break;

    frames++;
    uint32_t now = millis();
    uint32_t elapsed = now - windowStart;

    if (elapsed >= 1000) {
      float fps = frames * 1000.0f / elapsed;
      setStreamFps(fps);
      Serial.printf("# Stream FPS: %.1f\n", fps);
      frames = 0;
      windowStart = now;
    }

    vTaskDelay(1);
  }

  setStreamFps(0);
  httpd_req_async_handler_complete(req);

  portENTER_CRITICAL(&cameraMux);
  streamBusy = false;
  portEXIT_CRITICAL(&cameraMux);
  vTaskDelete(NULL);
}

static esp_err_t stream_handler(httpd_req_t *req) {
  if (!cameraOk) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  bool busy;
  portENTER_CRITICAL(&cameraMux);
  busy = streamBusy;
  if (!busy) streamBusy = true;
  portEXIT_CRITICAL(&cameraMux);

  if (busy) {
    httpd_resp_set_status(req, "503 Service Unavailable");
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_send(req, "Stream busy: one client at a time", HTTPD_RESP_USE_STRLEN);
  }

  httpd_req_t *asyncReq = nullptr;
  if (httpd_req_async_handler_begin(req, &asyncReq) != ESP_OK) {
    portENTER_CRITICAL(&cameraMux);
    streamBusy = false;
    portEXIT_CRITICAL(&cameraMux);
    return ESP_FAIL;
  }

  if (xTaskCreate(stream_task, "camera_stream", 8192, asyncReq, 5, nullptr) != pdPASS) {
    httpd_resp_send_500(asyncReq);
    httpd_req_async_handler_complete(asyncReq);
    portENTER_CRITICAL(&cameraMux);
    streamBusy = false;
    portEXIT_CRITICAL(&cameraMux);
    return ESP_FAIL;
  }

  return ESP_OK;
}

static esp_err_t not_found_handler(httpd_req_t *req, httpd_err_code_t error) {
  (void)error;
  httpd_resp_set_status(req, "404 Not Found");
  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_send(req, "Valid paths: / /stream /capture /meta /status /control /imu /calibrate", HTTPD_RESP_USE_STRLEN);
}

static void registerGet(const char *path, esp_err_t (*handler)(httpd_req_t *)) {
  httpd_uri_t uri = {};
  uri.uri = path;
  uri.method = HTTP_GET;
  uri.handler = handler;

  esp_err_t result = httpd_register_uri_handler(server_httpd, &uri);
  if (result != ESP_OK) {
    Serial.printf("# Failed to register %s: %s\n", path, esp_err_to_name(result));
  }
}

void startServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 10;
  config.lru_purge_enable = true;

  esp_err_t result = httpd_start(&server_httpd, &config);
  if (result != ESP_OK) {
    Serial.printf("# HTTP start failed: %s\n", esp_err_to_name(result));
    return;
  }

  registerGet("/", index_handler);
  registerGet("/capture", capture_handler);
  registerGet("/meta", meta_handler);
  registerGet("/status", status_handler);
  registerGet("/control", control_handler);
  registerGet("/stream", stream_handler);
  registerGet("/imu", imu_handler);
  registerGet("/calibrate", calibrate_handler);

  httpd_register_err_handler(server_httpd, HTTPD_404_NOT_FOUND, not_found_handler);
}

// ---------------- Setup ----------------

void setup() {
  Serial.begin(115200);
  delay(1500);

  if (psramFound()) {
    Serial.printf("# PSRAM OK: %u bytes\n", (unsigned)ESP.getPsramSize());
  } else {
    Serial.println("# PSRAM missing: camera configuration needs PSRAM");
  }

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

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

  if (psramFound()) {
    esp_err_t result = esp_camera_init(&config);
    cameraOk = result == ESP_OK;
    Serial.printf("# Camera init: %s\n", esp_err_to_name(result));
  }

  WiFi.mode(WIFI_AP);
  bool apOk = WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.setSleep(false);

  Serial.printf("# AP: %s\n", apOk ? "OK" : "FAILED");
  Serial.print("# Address: http://");
  Serial.println(WiFi.softAPIP());

  startServer();
  imuStart();
}

void loop() {
  imuTick();
  delay(1);
}