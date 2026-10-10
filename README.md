# SmartCap

**SmartCap** is a wearable multimodal driver-monitoring research prototype that combines camera-based facial behavior features and IMU-based head-motion features. The system uses edge processing to estimate drowsiness-related and distraction-related driver states, produce a risk score, and later support haptic feedback and prototype vehicle CAN integration.

> This is a graduation-project research prototype. It is not a medical diagnostic system, an automotive-certified safety product, or a replacement for production driver-monitoring systems.

---

## Project Goal

SmartCap investigates the following research question:

> Can multimodal fusion of facial visual features and head-motion features, combined with temporal modeling, improve robustness and reduce false alarms compared with vision-only and IMU-only driver-monitoring baselines?

The system focuses on **drowsiness-related behavioral cues** and **distraction-related behavior**, including:

- Prolonged eye closure
- Repeated slow blinking
- PERCLOS-like eye-closure ratio
- Yawning-related mouth opening
- Head-down duration
- Head nodding patterns
- Rapid or abnormal head movement
- Missing or degraded sensor data

The project does **not** claim to medically diagnose fatigue, identify drivers, recognize phone usage reliably, or control a real vehicle.

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                       SMARTCAP NODE                       │
│                                                          │
│  ESP32-S3 Camera                MPU6050 IMU              │
│         │                            │                   │
│         └───────────┬────────────────┘                   │
│                     │                                    │
│         Camera frames / head-motion features             │
└─────────────────────┼────────────────────────────────────┘
                      │
             Wi-Fi / USB Serial
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  EDGE GATEWAY                             │
│              Laptop during development                    │
│          Raspberry Pi in the later prototype              │
│                                                          │
│  Vision Features                                          │
│  - Face landmarks                                         │
│  - EAR                                                    │
│  - MAR                                                    │
│  - Eye closure duration                                   │
│  - PERCLOS-like score                                     │
│                                                          │
│  IMU Features                                             │
│  - Pitch / roll                                           │
│  - Pitch velocity                                         │
│  - Head-down duration                                     │
│  - Nodding score                                          │
│  - Motion energy                                          │
│                                                          │
│  Sensor Quality + Time Alignment                          │
│          ↓                                               │
│  Temporal Fusion + Risk Assessment                        │
│          ↓                                               │
│  Driver State + Confidence + Diagnostic Status            │
└─────────────────────┼────────────────────────────────────┘
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
  Future Haptic Alert      Future Prototype CAN Message
```

---

## Main Components

| Component | Role | Current Status |
|---|---|---|
| ESP32-S3 N16R8 camera node | Captures camera frames and sends them through local Wi-Fi | In progress |
| MPU6050 | Captures head-motion features | Working |
| Arduino Uno/Nano | Temporary IMU development controller | Working |
| Laptop | Development edge gateway for vision and IMU processing | Working |
| Raspberry Pi | Future edge gateway target | Planned |
| Haptic vibration motor | Future local warning actuator | Pending hardware |
| MCP2515 + CAN transceiver | Future prototype CAN integration | Planned |

---

## Current Development Status

### Team 1 — IMU and Haptic

The Team 1 firmware currently provides:

```text
MPU6050
    ↓
Calibration
    ↓
Low-pass and complementary filtering
    ↓
Motion feature extraction
    ↓
JSON messages through USB Serial
```

Current IMU features include:

```text
ax_g, ay_g, az_g
gx_dps, gy_dps, gz_dps
pitch_deg
roll_deg
pitch_velocity_dps
head_down_duration_ms
nodding_score
motion_energy
imu_status
calibration_status
imu_quality
```

The IMU sends data at approximately 25 Hz after internal sensing at 50 Hz.

> The MPU6050 does not independently detect drowsiness. It provides head-motion features for later fusion with visual features.

### Team 2 — ESP32 Camera and Communication

Team 2 is configuring an ESP32-S3 camera node without requiring SD-card recording.

Current objective:

```text
ESP32-S3 Camera
    ↓
Wi-Fi
    ↓
JPEG snapshot or MJPEG stream
    ↓
Laptop / Raspberry Pi vision pipeline
```

The camera node must provide:

```text
Camera IP address
Web UI URL
Snapshot endpoint
Stream endpoint
Resolution
JPEG quality
Measured FPS
Measured latency
```

### Team 3 — AI and Computer Vision

The vision baseline currently supports:

```text
Laptop Webcam
    ↓
MediaPipe Face Landmarker
    ↓
Face landmarks
    ↓
EAR + MAR
    ↓
Eye closure timing
    ↓
PERCLOS-like score
    ↓
Prolonged eye-closure and yawn warnings
```

Current vision files:

```text
edge_ai/vision/src/
├── webcam_test.py
├── face_landmarks.py
├── ear_mar.py
├── eye_closure_timer.py
└── drowsiness_detector.py
```

> Current warnings indicate behavioral cues such as prolonged eye closure. They are not final medical drowsiness diagnoses.

---

## Current Milestone

### Synchronized Multimodal Sensing Prototype

The current milestone is to demonstrate:

```text
ESP32-S3 Camera Stream
        ↓ Wi-Fi
Laptop Edge Gateway
        ↑ USB Serial
Arduino + MPU6050
        ↓
Vision Features + IMU Features
        ↓
Live Sensor Status and Feature Display
```

The milestone is complete when the system can:

1. Receive camera frames from ESP32-S3.
2. Detect facial landmarks on ESP32 camera frames.
3. Extract EAR, MAR, eye-closure duration, and PERCLOS-like values.
4. Receive IMU JSON messages from Arduino through USB Serial.
5. Display pitch, roll, motion energy, head-down duration, and nodding score.
6. Detect sensor failures and output `UNKNOWN`, not `ATTENTIVE`.
7. Log synchronized or timestamped sensor data for later fusion experiments.

The current milestone does **not** require:

```text
A final drowsiness classifier
GRU / LSTM / TCN
Automatic haptic vibration
Raspberry Pi deployment
CAN integration
Battery enclosure
Cloud dashboard
Phone detection
Real-road driving tests
```

---

## Repository Structure

```text
smartcap-driver-monitoring/
├── firmware/
│   ├── imu_haptic/
│   └── esp32_camera/
│
├── edge_ai/
│   ├── vision/
│   ├── temporal_model/
│   └── fusion/
│
├── gateway/
├── datasets/
├── experiments/
├── docs/
├── protocol/
├── tests/
├── README.md
└── .gitignore
```

### Important Paths

| Path | Purpose |
|---|---|
| `protocol/SMARTCAP_INTERFACE_SPEC_v1.md` | Shared communication and data-interface contract |
| `firmware/imu_haptic/` | Team 1 Arduino and MPU6050 firmware |
| `firmware/esp32_camera/` | Team 2 ESP32-S3 camera firmware |
| `edge_ai/vision/` | Team 3 computer vision pipeline |
| `edge_ai/fusion/` | IMU reading, time alignment, sensor fusion, and risk logic |
| `datasets/schemas/` | Dataset format documentation |
| `datasets/sample_data/` | Small anonymized sample data only |
| `experiments/logs/` | Experiment documentation |
| `experiments/results/` | Tables, metrics, and generated plots |
| `docs/wiring/` | Hardware wiring diagrams |
| `docs/calibration/` | Calibration procedures |

---

## Team Responsibilities

| Team | Main Responsibility |
|---|---|
| Team 1 | MPU6050 calibration, filtering, motion features, serial JSON output, future haptic output |
| Team 2 | ESP32-S3 camera configuration, Wi-Fi communication, stream/snapshot endpoint, camera testing |
| Team 3 | Face landmarks, EAR/MAR, eye-closure timing, temporal vision baseline, later sensor fusion |
| Wave 2 | Raspberry Pi gateway, CAN prototype, power system, dashboard, deployment testing |

---

## Data Interface

All teams must follow:

```text
protocol/SMARTCAP_INTERFACE_SPEC_v1.md
```

Core principles:

```text
- Use defined field names and units.
- Use device_timestamp_ms and sequence_id.
- Add received_timestamp_ms at the gateway.
- Use status and quality fields.
- Missing or unreliable sensing data must produce UNKNOWN.
- Do not change interfaces without team agreement.
```

Example IMU message:

```json
{
  "type": "imu",
  "device_id": "smartcap_01",
  "sequence_id": 4065,
  "device_timestamp_ms": 169245,
  "pitch_deg": -12.4,
  "roll_deg": 2.8,
  "pitch_velocity_dps": 18.6,
  "head_down_duration_ms": 0,
  "nodding_score": 0.24,
  "motion_energy": 0.12,
  "imu_status": "OK",
  "calibration_status": "CALIBRATED",
  "imu_quality": 1.00
}
```

---

## Dataset and Experiment Rules

### Data Collection

- Use anonymous participant codes such as `P01`, `P02`, and `P03`.
- Obtain written consent before recording any participant.
- Do not collect real sleep deprivation data.
- Record simulated drowsiness-related behaviors only.
- Conduct experiments in a parked vehicle, lab setup, or simulator.
- Do not conduct drowsiness experiments while driving on public roads.
- Use subject-independent train/validation/test splits later in the project.

### GitHub Rules

Do not upload:

```text
Raw participant videos
Private raw datasets
Personal identifiers
Large recordings
Virtual environments
Secrets, Wi-Fi passwords, or API keys
```

Upload only:

```text
Code
Documentation
Small anonymized sample data
Dataset schemas
Experiment logs
Analysis summaries
Metrics tables
Plots
```

---

## Safety and Scope

SmartCap is designed with safety-oriented considerations:

```text
- Sensor status and quality monitoring
- UNKNOWN state for missing or unreliable data
- Timestamp and sequence ID tracking
- Local processing preference
- No cloud video upload in the intended final design
- Prototype CAN messaging only
```

The project may be described as:

```text
ISO 26262-inspired
```

It must not be described as:

```text
ISO 26262 certified
Automotive safety certified
Medical diagnostic system
Guaranteed accident-prevention system
```

The prototype CAN integration will report driver-state information only. It will not control:

```text
Braking
Steering
Throttle
Vehicle engine
