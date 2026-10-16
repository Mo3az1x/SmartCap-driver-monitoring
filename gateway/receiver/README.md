ر\# Gateway Receiver



First integration milestone: \*\*camera stream + IMU stream + manual vibration command\*\*,

following `protocol/SMARTCAP\_INTERFACE\_SPEC\_v1.md`.



\## Run



```bash

pip install -r requirements.txt



\# IMU (Arduino over USB) + ESP32 camera

python gateway\_receiver.py --port COM5 --camera-url http://192.168.4.1/stream --show



\# IMU only

python gateway\_receiver.py --port COM5 --no-camera



\# No hardware (simulated IMU)

python gateway\_receiver.py --sim-imu --no-camera --duration 10

```



Connect the laptop to the `SmartCap-CAM` Wi-Fi network to reach the camera

(the laptop has no internet while connected). Check the real camera IP in the ESP32 Serial output.



Close the Arduino Serial Monitor first, otherwise the COM port is busy.



\## Console / preview keys



`v <level 0-3> \[duration\_ms]`, or just `0` `1` `2` `3`, `q` to quit.

With `--show`, keys `0-3` vibrate and `q` quits.



\## What it records (`output/`)



| File | Content |

|---|---|

| `gateway\_imu\_\*.csv` | every valid IMU message + `received\_timestamp\_ms` |

| `gateway\_camera\_meta\_\*.csv` | per-frame metadata only (no images are stored) |

| `gateway\_commands\_\*.csv` | every vibration command sent |



Add `gateway/receiver/output/` to `.gitignore`.



\## Behaviour



\- Packet loss = gaps in `sequence\_id`; a sequence going backwards counts as an Arduino reset.

\- A source with no data inside its timeout becomes `DISCONNECTED` and the gateway state is `UNKNOWN`.

&#x20; It never reports `ATTENTIVE`. Defaults (`--imu-timeout-ms 500`, `--camera-timeout-ms 2000`) are starting

&#x20; values and must be measured experimentally.

\- The MJPEG stream carries no device timestamp, so camera metadata has laptop arrival time only.

&#x20; Device-to-laptop latency cannot be measured until the clocks are synchronized.

\- `actual\_fps` is inflated for the first second (OpenCV delivers buffered frames in a burst).



\## Tested / not tested



Tested here: simulated IMU, packet-loss counting, commands, a fake MJPEG server, camera-down handling.

\*\*Not tested with real hardware\*\* (Arduino serial port, ESP32 stream, vibration command parsing on the Arduino).

