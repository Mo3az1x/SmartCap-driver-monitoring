// SmartCap Interface Spec v1, section 8 (camera_frame)
#define DEVICE_ID       "smartcap_01"
#define TARGET_FPS      10.0f   // camera_quality = measured FPS / TARGET_FPS (agree on this with Team 3)
#define CAM_TIMEOUT_MS  2000    // no frame for this long -> DISCONNECTED (measure it and adjust)

// Status values come from spec section 5: OK, DEGRADED, INVALID, DISCONNECTED, UNKNOWN
static inline void cameraHealth(bool cameraOk, uint32_t framesServed, uint32_t lastFrameTs,
                                float fps, const char **status, float *quality) {
  *quality = 0;
  if (!cameraOk)            { *status = "DISCONNECTED"; return; }  // camera init failed
  if (framesServed == 0)    { *status = "UNKNOWN";      return; }  // no frame yet
  if (millis() - lastFrameTs > CAM_TIMEOUT_MS) { *status = "DISCONNECTED"; return; }
  if (fps <= 0)             { *status = "UNKNOWN";      return; }  // FPS not measured yet
  float q = fps / TARGET_FPS;
  if (q > 1) q = 1;
  *quality = q;
  if (q >= 0.8f)      *status = "OK";
  else if (q >= 0.4f) *status = "DEGRADED";
  else                *status = "INVALID";
}