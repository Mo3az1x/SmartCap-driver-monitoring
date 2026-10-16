# Gateway Receiver

First integration milestone: **camera stream + IMU stream + manual vibration command**,
following `protocol/SMARTCAP_INTERFACE_SPEC_v1.md`.

This folder has two independent programs:

| Program | Job |
|---|---|
| `gateway_receiver.py` | Reads the IMU (Arduino over USB), watches the ESP32 camera stream, sends vibration commands, logs metadata |
| `vision_worker.py` | Reads the camera (ESP32 stream or a laptop webcam) and extracts visual driver-state features with MediaPipe |

Neither of them decides that the driver is drowsy. They only produce measurements and statuses.

## Setup

```bash
cd gateway/receiver
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

`requirements.txt` pins `mediapipe==0.10.21` and `numpy<2`. Do not upgrade mediapipe without testing
`vision_worker.py` again (it uses the MediaPipe Tasks API: `FaceLandmarker`).

## Run the gateway

```bash
# IMU (Arduino over USB) + ESP32 camera
python gateway_receiver.py --port COM5 --camera-url http://192.168.4.1/stream --show

# IMU only
python gateway_receiver.py --port COM5 --no-camera

# No hardware (simulated IMU)
python gateway_receiver.py --sim-imu --no-camera --duration 10
```

Connect the laptop to the `SmartCap-CAM` Wi-Fi network to reach the camera
(the laptop has no internet while connected). The stream is on **port 80**
(`http://192.168.4.1/stream`, no `:81`). Check the real camera IP in the ESP32 Serial output.

Close the Arduino Serial Monitor first, otherwise the COM port is busy.

### Console / preview keys

`v <level 0-3> [duration_ms]`, or just `0` `1` `2` `3`, `q` to quit.
With `--show`, keys `0-3` vibrate and `q` quits.

## Run the vision worker

```bash
# ESP32 camera
python vision_worker.py --camera-url http://192.168.4.1/stream --show

# Laptop webcam (test without hardware)
python vision_worker.py --camera-url 0 --show
```

First run needs internet once: the Face Landmarker model (`face_landmarker.task`, about 4 MB) is downloaded
automatically into `gateway/receiver/models/` if it is not found (searched in `models/` next to the script,
in the parent `models/`, next to the script and in the current folder). Download it before joining the
`SmartCap-CAM` network, or pass `--model PATH`. The model is ignored by git.

### Keys (with `--show`)

`q` or `ESC` quit, `c` redo the EAR calibration.

### What it computes

| Feature | Meaning |
|---|---|
| EAR | eye aspect ratio (eye opening / eye width), smoothed |
| EAR calibration | first 60 face frames with eyes open set a personal threshold = 72 % of the open-eye EAR (clamped 0.12 to 0.26) |
| closure duration | seconds the eyes have been below the threshold; `prolonged_eye_closure` at 1.5 s |
| PERCLOS (30 s) | fraction of the last 30 s with eyes closed, time-weighted (not frame-count based) |
| blinks | closures of 0.05 to 0.5 s; `blink_rate_per_min`; closures longer than 0.5 s go to `long_closure_count` |
| MAR | mouth aspect ratio; a yawn = MAR above 0.50 for at least 1.2 s, counted when the mouth closes |
| `head_yaw_ratio` | about 0 when facing the camera, above 0.35 in absolute value = head turned |
| `face_width_ratio` | face size in the image (small = far or low resolution) |
| `vision_quality` | 0 to 1 from face size, brightness, head turn, processing fps |

### Vision status values

| Status | Meaning |
|---|---|
| `OK` | quality 0.70 or more |
| `DEGRADED` | quality below 0.70 |
| `INVALID` | quality below 0.30 |
| `CALIBRATING` | EAR calibration running |
| `NO_FACE` | no face in the frame. Eye state is `UNKNOWN`: this is **not** "eyes closed" and **not** "attentive" |
| `WAITING_CAMERA` | camera never delivered a frame yet |
| `CAMERA_DISCONNECTED` | no new frame for more than `--camera-timeout-s` |

### Useful options

| Option | Default | Meaning |
|---|---|---|
| `--ear-closed-threshold` | 0.20 | fixed EAR threshold (used until calibration replaces it) |
| `--no-calibration` | off | keep the fixed threshold |
| `--ear-ratio` | 0.72 | calibrated threshold = open-eye EAR x this value |
| `--mar-yawn-threshold` | 0.50 | MAR limit for a yawn |
| `--yawn-min-s` | 1.2 | minimum mouth-open time for a yawn |
| `--prolonged-closure-s` | 1.5 | prolonged eye closure limit |
| `--perclos-window-s` | 30 | PERCLOS window |
| `--camera-timeout-s` | 2.0 | camera considered lost after this |
| `--flip` | off | mirror the image |
| `--model` | auto | path to `face_landmarker.task` |
| `--latest-json` | auto | file with the latest features (`''` disables it) |
| `--outdir` | `output/` next to the script | where logs are written |

All thresholds are starting values and must be tuned with real recordings from the ESP32 camera.

## Do not open the camera stream twice

The ESP32 stream accepts **one client at a time** (a second client gets `503 Stream busy`).
So these cannot run together on the same camera:

- `gateway_receiver.py` with the camera enabled
- `vision_worker.py`
- a browser tab showing the camera page

Use `gateway_receiver.py --no-camera` while `vision_worker.py` is running, or close one of them.

## What it records (`output/`)

| File | Content |
|---|---|
| `gateway_imu_*.csv` | every valid IMU message + `received_timestamp_ms` |
| `gateway_camera_meta_*.csv` | per-frame metadata only (no images are stored) |
| `gateway_commands_*.csv` | every vibration command sent |
| `vision_features_*.csv` | one row per processed frame (26 columns: EAR, MAR, PERCLOS, blinks, yawns, status, quality, ...) |
| `vision_latest.json` | the most recent vision features, rewritten about 10 times per second (for other programs to read) |

`gateway/receiver/output/` and `gateway/receiver/models/` are in `.gitignore`.
Camera images are never saved.

## Behaviour

- Packet loss = gaps in `sequence_id`; a sequence going backwards counts as an Arduino reset.
- A source with no data inside its timeout becomes `DISCONNECTED` and the gateway state is `UNKNOWN`.
  It never reports `ATTENTIVE`. Defaults (`--imu-timeout-ms 500`, `--camera-timeout-ms 2000`) are starting
  values and must be measured experimentally.
- The MJPEG stream carries no device timestamp, so camera metadata has laptop arrival time only.
  Device-to-laptop latency cannot be measured until the clocks are synchronized.
- `actual_fps` is inflated for the first second (OpenCV delivers buffered frames in a burst).
- `vision_worker.py` keeps only the newest camera frame, so processing delay does not grow over time,
  and it reconnects by itself if the stream drops.

## Tested / not tested

Tested here: simulated IMU, packet-loss counting, commands, a fake MJPEG server, camera-down handling.
Vision worker: feature logic (EAR, MAR, blinks, PERCLOS, yawns, calibration, quality) with synthetic landmarks,
the frame grabber and main loop on a synthetic video, and a run on a laptop webcam.

**Not tested with real hardware:** Arduino serial port, ESP32 stream (gateway and vision worker),
vibration command parsing on the Arduino. Vision thresholds are not tuned on ESP32 images yet.
