// SmartCap camera node - tuning version (full settings page + same protocol endpoints)
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

// ---------------- Web page ----------------
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartCap Camera</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:10px}
.wrap{display:flex;flex-wrap:wrap;gap:16px}
.left{flex:1 1 480px}.right{flex:1 1 320px;max-width:420px}
img{width:100%;border-radius:6px;background:#000}
.row{display:flex;justify-content:space-between;align-items:center;margin:6px 0;gap:8px}
.row label{flex:0 0 130px;font-size:14px}
.row input[type=range]{flex:1}.row span{width:44px;text-align:right;font-size:13px}
select{flex:1;background:#222;color:#eee;border:1px solid #444;padding:3px}
button{background:#2563eb;color:#fff;border:0;padding:8px 12px;border-radius:5px;margin:4px 4px 4px 0;cursor:pointer}
textarea{width:100%;height:160px;background:#000;color:#9f9;border:1px solid #333;font-size:12px;white-space:pre}
h4{margin:14px 0 4px;color:#8ab4ff}
#fps{font-size:20px;color:#9f9}
.logbar{margin:6px 0}
.logbar label{margin-left:8px;font-size:14px}
</style></head><body>
<div class="wrap">
 <div class="left">
  <img id="stream">
  <div>Stream FPS: <span id="fps">-</span></div>
  <div class="logbar">
   <button onclick="toggleVideo()">Show / hide video</button>
   <button onclick="window.open('/capture')">Snapshot</button>
   <button onclick="window.open('/meta')">Meta</button>
  </div>
  <div class="logbar">
   <button onclick="logNow()">Log now</button>
   <label><input type="checkbox" id="auto" onchange="toggleAuto()"> Auto log every 2s</label>
  </div>
  <div class="logbar">
   <button onclick="copyLog()">Copy log</button>
   <button onclick="clearLog()">Clear log</button>
  </div>
  <textarea id="log" readonly></textarea>
 </div>
 <div class="right" id="ctrl"></div>
</div>
<script>
const SIZES=[[5,'QVGA 320x240'],[6,'CIF 400x296'],[8,'VGA 640x480'],[9,'SVGA 800x600'],[10,'XGA 1024x768'],[11,'HD 1280x720'],[12,'SXGA 1280x1024'],[13,'UXGA 1600x1200']];
const GAIN=[[0,'2x'],[1,'4x'],[2,'8x'],[3,'16x'],[4,'32x'],[5,'64x'],[6,'128x']];
const WB=[[0,'Auto'],[1,'Sunny'],[2,'Cloudy'],[3,'Office'],[4,'Home']];
const FX=[[0,'None'],[1,'Negative'],[2,'Grayscale'],[3,'Red tint'],[4,'Green tint'],[5,'Blue tint'],[6,'Sepia']];
const C=[
['Size & compression'],
['framesize','Resolution','select',SIZES],
['quality','JPEG quality (low=better)','range',10,63],
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
['colorbar','Color bar (test)','check']
];
const HEADER=['Time','Resolution','Quality','FPS','Brightness','Contrast','Saturation','AEC','AEC_value','AGC','Gain','Hmirror','Vflip'].join('\t');
let S={};
let autoTimer=null;
let videoOn=true;
const ctrl=document.getElementById('ctrl');

function send(k,v){fetch('/control?var='+k+'&val='+v).then(()=>setTimeout(load,300));}

function build(){
 ctrl.innerHTML='';
 C.forEach(c=>{
  if(c.length==1){ctrl.insertAdjacentHTML('beforeend','<h4>'+c[0]+'</h4>');return;}
  const [k,l,t]=c;const row=document.createElement('div');row.className='row';
  let h='<label>'+l+'</label>';
  if(t=='range')h+='<input type="range" id="'+k+'" min="'+c[3]+'" max="'+c[4]+'"><span id="v_'+k+'"></span>';
  if(t=='check')h+='<input type="checkbox" id="'+k+'">';
  if(t=='select'){h+='<select id="'+k+'">'+c[3].map(o=>'<option value="'+o[0]+'">'+o[1]+'</option>').join('')+'</select>';}
  row.innerHTML=h;ctrl.appendChild(row);
  const el=document.getElementById(k);
  if(t=='range')el.oninput=()=>document.getElementById('v_'+k).textContent=el.value;
  el.onchange=()=>send(k,t=='check'?(el.checked?1:0):el.value);
 });
}

function load(){
 fetch('/status').then(r=>r.json()).then(j=>{
  S=j;
  C.forEach(c=>{
   if(c.length==1)return;const k=c[0],t=c[2],el=document.getElementById(k);if(!el||!(k in j))return;
   if(t=='check')el.checked=!!j[k];else el.value=j[k];
   if(t=='range'){const s=document.getElementById('v_'+k);if(s)s.textContent=j[k];}
  });
  document.getElementById('fps').textContent=j.fps.toFixed(1);
 });
}

function sizeName(v){const o=SIZES.find(x=>x[0]==v);return o?o[1].split(' ')[0]:v;}

function addRow(){
 const t=document.getElementById('log');
 if(t.value==='')t.value=HEADER+'\n';
 const row=[new Date().toLocaleTimeString(),sizeName(S.framesize),S.quality,S.fps.toFixed(1),
  S.brightness,S.contrast,S.saturation,S.aec,S.aec_value,S.agc,S.agc_gain,S.hmirror,S.vflip].join('\t');
 t.value+=row+'\n';
 t.scrollTop=t.scrollHeight;
}

function logNow(){
 fetch('/status').then(r=>r.json()).then(j=>{S=j;addRow();});
}

function toggleAuto(){
 if(document.getElementById('auto').checked){
  logNow();
  autoTimer=setInterval(logNow,2000);
 }else{
  clearInterval(autoTimer);autoTimer=null;
 }
}

function copyLog(){
 const t=document.getElementById('log');
 t.select();
 document.execCommand('copy');
}

function clearLog(){document.getElementById('log').value='';}

function toggleVideo(){
 const im=document.getElementById('stream');
 if(videoOn){im.removeAttribute('src');videoOn=false;}
 else{setTimeout(()=>{im.src='/stream?t='+Date.now();},1500);videoOn=true;}
}

document.getElementById('stream').src='/stream';
build();load();setInterval(load,2000);
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

// Full settings state for the tuning page (not part of the interface spec)
static esp_err_t status_handler(httpd_req_t *req) {
  sensor_t *s = esp_camera_sensor_get();
  if (!s) { httpd_resp_send_500(req); return ESP_FAIL; }
  char buf[640];
  snprintf(buf, sizeof(buf),
    "{\"framesize\":%u,\"quality\":%u,\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
    "\"special_effect\":%u,\"wb_mode\":%u,\"awb\":%u,\"awb_gain\":%u,\"aec\":%u,\"aec2\":%u,"
    "\"ae_level\":%d,\"aec_value\":%u,\"agc\":%u,\"agc_gain\":%u,\"gainceiling\":%u,"
    "\"bpc\":%u,\"wpc\":%u,\"raw_gma\":%u,\"lenc\":%u,\"hmirror\":%u,\"vflip\":%u,"
    "\"dcw\":%u,\"colorbar\":%u,\"fps\":%.1f}",
    s->status.framesize, s->status.quality, s->status.brightness, s->status.contrast,
    s->status.saturation, s->status.special_effect, s->status.wb_mode, s->status.awb,
    s->status.awb_gain, s->status.aec, s->status.aec2, s->status.ae_level,
    s->status.aec_value, s->status.agc, s->status.agc_gain, s->status.gainceiling,
    s->status.bpc, s->status.wpc, s->status.raw_gma, s->status.lenc, s->status.hmirror,
    s->status.vflip, s->status.dcw, s->status.colorbar, (double)streamFps);
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, buf, HTTPD_RESP_USE_STRLEN);
}

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
  int res = 0;

  if      (!strcmp(variable, "framesize"))      res = s->set_framesize(s, (framesize_t)val);
  else if (!strcmp(variable, "quality"))        res = s->set_quality(s, val);
  else if (!strcmp(variable, "brightness"))     res = s->set_brightness(s, val);
  else if (!strcmp(variable, "contrast"))       res = s->set_contrast(s, val);
  else if (!strcmp(variable, "saturation"))     res = s->set_saturation(s, val);
  else if (!strcmp(variable, "special_effect")) res = s->set_special_effect(s, val);
  else if (!strcmp(variable, "aec"))            res = s->set_exposure_ctrl(s, val);
  else if (!strcmp(variable, "aec2"))           res = s->set_aec2(s, val);
  else if (!strcmp(variable, "ae_level"))       res = s->set_ae_level(s, val);
  else if (!strcmp(variable, "aec_value"))      res = s->set_aec_value(s, val);
  else if (!strcmp(variable, "agc"))            res = s->set_gain_ctrl(s, val);
  else if (!strcmp(variable, "agc_gain"))       res = s->set_agc_gain(s, val);
  else if (!strcmp(variable, "gainceiling"))    res = s->set_gainceiling(s, (gainceiling_t)val);
  else if (!strcmp(variable, "awb"))            res = s->set_whitebal(s, val);
  else if (!strcmp(variable, "awb_gain"))       res = s->set_awb_gain(s, val);
  else if (!strcmp(variable, "wb_mode"))        res = s->set_wb_mode(s, val);
  else if (!strcmp(variable, "hmirror"))        res = s->set_hmirror(s, val);
  else if (!strcmp(variable, "vflip"))          res = s->set_vflip(s, val);
  else if (!strcmp(variable, "lenc"))           res = s->set_lenc(s, val);
  else if (!strcmp(variable, "bpc"))            res = s->set_bpc(s, val);
  else if (!strcmp(variable, "wpc"))            res = s->set_wpc(s, val);
  else if (!strcmp(variable, "raw_gma"))        res = s->set_raw_gma(s, val);
  else if (!strcmp(variable, "dcw"))            res = s->set_dcw(s, val);
  else if (!strcmp(variable, "colorbar"))       res = s->set_colorbar(s, val);
  else res = -1;

  Serial.printf("SET %s=%d -> %s\n", variable, val, res == 0 ? "ok" : "not supported");
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
  httpd_resp_send(req, "Not found. Valid: /  /stream  /capture  /meta  /status  /control?var=..&val=..",
                  HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

void startServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 10;
  config.lru_purge_enable = true;

  httpd_uri_t index_uri   = { "/",        HTTP_GET, index_handler,   NULL };
  httpd_uri_t capture_uri = { "/capture", HTTP_GET, capture_handler, NULL };
  httpd_uri_t meta_uri    = { "/meta",    HTTP_GET, meta_handler,    NULL };
  httpd_uri_t status_uri  = { "/status",  HTTP_GET, status_handler,  NULL };
  httpd_uri_t control_uri = { "/control", HTTP_GET, control_handler, NULL };
  httpd_uri_t stream_uri  = { "/stream",  HTTP_GET, stream_handler,  NULL };

  if (httpd_start(&server_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(server_httpd, &index_uri);
    httpd_register_uri_handler(server_httpd, &capture_uri);
    httpd_register_uri_handler(server_httpd, &meta_uri);
    httpd_register_uri_handler(server_httpd, &status_uri);
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