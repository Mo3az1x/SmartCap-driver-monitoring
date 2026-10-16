# P0 Hardware Checklist

Status of every row: **NOT RUN** in this repository session.
Software tests passing do **not** mean these items passed.

Use firmware profile `haptic` (`230400` baud, ~25 Hz JSON) unless you flash `smartcap_imu_led` (`115200`, ~10 Hz).

Canonical software path:

```text
python gateway/receiver/run_session.py --port COMx --camera-url http://192.168.4.1/stream
```

`vision_worker.py` is the only MJPEG client. Do not also open `/stream` in a browser.

| # | Check | Result (NOT RUN / PASS / FAIL) | Notes |
|---|---|---|---|
| 1 | Flash `smartcap_imu_haptic.ino`, Serial JSON at 230400 | NOT RUN | |
| 2 | `run_session.py --port ... --no-vision` shows IMU `OK` after calibration | NOT RUN | Close Arduino Serial Monitor first |
| 3 | ESP32 Soft-AP `SmartCap-CAM`, laptop joins, `/stream` on port 80 | NOT RUN | Confirm IP from ESP32 Serial |
| 4 | `vision_worker.py --camera-url http://<ip>/stream --show` landmarks on ESP32 frames | NOT RUN | One client only |
| 5 | Combined `run_session.py` IMU + vision, both CSVs written | NOT RUN | |
| 6 | Unplug IMU → sensing `UNKNOWN`, not ATTENTIVE | NOT RUN | |
| 7 | Stop camera / leave AP → `CAMERA_DISCONNECTED` / UNKNOWN | NOT RUN | |
| 8 | Manual `vibration_command` parsed on Arduino (level 1–3 then 0) | NOT RUN | Hardware motor optional |
| 9 | Measure IMU Hz, stream FPS, useful timeout values | NOT RUN | Replace CLI defaults only after measurement |

Clock reminder: `device_timestamp_ms` (Arduino/ESP32 `millis`) and `received_timestamp_ms` (laptop Unix ms) are different clocks. Do not subtract them and call the difference capture latency.
