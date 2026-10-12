# SmartCap IMU Dataset Schema

## Unit conventions

| Field | Unit | Description |
|---|---|---|
| ax_g, ay_g, az_g | g | Accelerometer axes |
| gx_dps, gy_dps, gz_dps | degrees/s | Gyroscope axes |
| pitch_deg | degrees | Relative head pitch after calibration |
| roll_deg | degrees | Relative head roll after calibration |
| pitch_velocity_dps | degrees/s | Smoothed pitch angular velocity |
| head_down_duration_ms | ms | Time held in head-down state |
| nodding_score | 0–1 | Decaying motion-pattern confidence |
| motion_energy | normalized | Relative motion intensity |
| imu_quality | 0–1 | Signal-health quality score |

## Labels

```text
still
normal
look_down
nod
rapid_turn
```

## File naming

```text
PXX_label_optional-tag_YYYYMMDD_HHMMSS.csv
```

Example:

```text
P01_nod_trial1_20261010_224500.csv
```

## Required metadata

```text
participant_id
label
device_timestamp_ms
received_timestamp_ms
hardware_version
firmware_version
calibration_status
```
