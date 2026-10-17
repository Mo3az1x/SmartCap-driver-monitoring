# Team 1 - IMU (MPU6050) + LED/Haptic

Arduino Uno firmware that reads an MPU6050, computes head-motion features and lights the
onboard LED ("L", pin 13) when the head tilts down.

## Folder layout

```
firmware/imu_haptic/
  src/
    smartcap_imu_led/smartcap_imu_led.ino   <- main sketch (IMU + LED)
    imu_calibration/imu_calibration.ino
    haptic_test/haptic_test.ino
  include/
  data/
```

Open `smartcap_imu_led.ino` in the Arduino IDE (the folder name must match the file name).

## Wiring

| MPU6050 (GY-521) | Arduino Uno |
|---|---|
| VCC | 5V |
| GND | GND |
| SDA | A4 |
| SCL | A5 |
| AD0 | GND |

No external LED is needed: the onboard LED (pin 13) is used.

## Serial

115200 baud, line ending "Newline". One JSON line per message (10 Hz). Lines starting with `#` are comments.

| Key | Action |
|---|---|
| `0` `1` `2` `3` | test LED/vibration level for 3 s |
| `C` | recalibrate (keep still 5 s) |
| `E` / `S` | extended / spec-only output |
| `D` | local demo mode ON |
| `?` | print status |

## Start-up behaviour

1. LED flashes once (sensor found).
2. About 5 s of slow blinking = calibration. Keep the head still in the normal driving posture.
3. 3 fast blinks = calibration done.
4. LED is on while the head is tilted down; off when the head is back up.

## Conventions

- Chip X axis points forward (towards the brim), Z points up.
- `pitch_deg > 0` means head DOWN, `roll_deg > 0` means right ear down.
- If the sign is reversed on your mounting, change `PITCH_SIGN` / `ROLL_SIGN` in the sketch.
- All angles are relative to the baseline measured during calibration.

## Thresholds (in the sketch)

| Constant | Value | Meaning |
|---|---|---|
| `HEAD_DOWN_ON_DEG` | 15 | head is "down" above this pitch |
| `HEAD_DOWN_OFF_DEG` | 10 | head is "up" again below this pitch (hysteresis) |
| `NOD_DEPTH_DEG` | 12 | minimum depth of a nod |
| `NOD_MIN_VEL_DPS` | 25 | minimum downward speed of a nod |
| `NOD_MIN_MS` / `NOD_MAX_MS` | 150 / 2500 | duration window of a nod |
| `COMP_ALPHA` | 0.98 | gyro weight in the complementary filter |

Demo vibration/LED levels: level 1 after 0.6 s head-down, level 2 after 1.5 s, level 3 after 3 s
(or 3 nods in 10 s). The IMU alone does not detect drowsiness; it only produces motion features.
