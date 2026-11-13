
#!/usr/bin/env python3
"""
SmartCap Gateway - HTTP IMU + ESP32 MJPEG Camera

Architecture:
    ESP32 /imu    -> HTTP IMU Worker -> IMU CSV
    ESP32 /stream -> Camera Worker   -> Camera CSV

Features:
    - Reads MPU6050 raw data from ESP32 HTTP endpoint.
    - Reads ESP32 MJPEG camera stream using OpenCV.
    - Tracks sequence gaps, duplicates, resets and invalid samples.
    - Adds laptop receive timestamps.
    - Detects IMU and camera timeouts.
    - Logs metadata only; camera frames are not saved.
    - Supports simulated IMU testing without hardware.

Examples:
    python gateway_receiver.py
    python gateway_receiver.py --show
    python gateway_receiver.py --no-camera
    python gateway_receiver.py --sim-imu --no-camera --duration 10

Console:
    q + Enter = quit
"""

import argparse
import csv
import json
import math
import sys
import threading
import time
import urllib.request
import urllib.error

from collections import deque
from datetime import datetime
from pathlib import Path

<<<<<<< Updated upstream

DEFAULT_IMU_URL = "http://192.168.4.1/imu"
DEFAULT_CAMERA_URL = "http://192.168.4.1/stream"
=======
try:
    import serial
except ImportError:  # pyserial might not be installed in the current environment
    serial = None

DEFAULT_CAMERA_URL = "http://192.168.4.1:81/stream"   # ESP32 soft-AP default IP; check the ESP32 Serial output
DEFAULT_BAUD = 230400                                  # must match the Arduino sketch (SERIAL_BAUD)
>>>>>>> Stashed changes

IMU_REQUIRED = [
    "type",
    "device_id",
    "sequence_id",
    "device_timestamp_ms",
    "device_clock_domain",
    "sample_age_ms",
    "sample_valid",
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
    "imu_status",
    "calibration_status",
]

NUMERIC_FIELDS = [
    "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps",
]


# ============================================================
# Utilities
# ============================================================

def now_ms():
    return int(time.time() * 1000)


def rate_hz(times):
    if len(times) < 2 or times[-1] <= times[0]:
        return 0.0

    return (len(times) - 1) / (times[-1] - times[0])


def finite_number(value):
    try:
        number = float(value)
        return math.isfinite(number)
    except (ValueError, TypeError, OverflowError):
        return False


# ============================================================
# CSV Logger
# ============================================================

class CsvLog:
    """Thread-safe append-only CSV logger."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.file = None
        self.writer = None
        self.rows = 0

    def write(self, row):
        with self.lock:
            if self.writer is None:
                self.file = open(
                    self.path,
                    "w",
                    newline="",
                    encoding="utf-8"
                )

                self.writer = csv.DictWriter(
                    self.file,
                    fieldnames=list(row.keys()),
                    extrasaction="ignore",
                    restval=""
                )

                self.writer.writeheader()

            self.writer.writerow(row)
            self.rows += 1

            if self.rows % 10 == 0:
                self.file.flush()

    def close(self):
        with self.lock:
            if self.file:
                self.file.flush()
                self.file.close()
                self.file = None


# ============================================================
# Shared State
# ============================================================

class Shared:

    def __init__(self):
        self.lock = threading.Lock()
        self.quit = threading.Event()

        self.imu = {
            "status": "WAITING",
            "last_rx_ms": None,
            "last_http_ms": None,
            "last_seq": None,
            "received": 0,
            "lost": 0,
            "duplicates": 0,
            "invalid_samples": 0,
            "bad_lines": 0,
            "resets": 0,
            "max_gap_ms": 0,
            "rx_times": deque(maxlen=50),
            "last": None,
        }

        self.cam = {
            "status": "WAITING",
            "last_rx_ms": None,
            "frames": 0,
            "reconnects": 0,
            "size": None,
            "rx_times": deque(maxlen=30),
            "frame": None,
        }


# ============================================================
# IMU HTTP Worker
# ============================================================

class HttpImuWorker(threading.Thread):

    def __init__(self, shared, args, log):
        super().__init__(daemon=True)

        self.s = shared
        self.args = args
        self.log = log

    def handle_message(self, msg):

        if not isinstance(msg, dict):
            self._bad_message("JSON is not an object")
            return

        missing = [
            field for field in IMU_REQUIRED
            if field not in msg
        ]

        if msg.get("type") != "imu_raw" or missing:
            self._bad_message(
                f"Wrong type or missing fields: {missing}"
            )
            return

        try:
            seq = int(msg["sequence_id"])
            device_ts = int(msg["device_timestamp_ms"])
            age = int(msg["sample_age_ms"])

            if seq < 0 or device_ts < 0:
                raise ValueError("Negative sequence or timestamp")

            values = {
                field: float(msg[field])
                for field in NUMERIC_FIELDS
            }

            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError("Non-finite sensor value")

            if not isinstance(msg["sample_valid"], bool):
                raise ValueError("sample_valid must be boolean")

        except (ValueError, TypeError, OverflowError) as error:
            self._bad_message(str(error))
            return

        rx = now_ms()

        with self.s.lock:

            st = self.s.imu
            st["last_http_ms"] = rx

            previous_seq = st["last_seq"]

            if previous_seq is not None:

                if seq == previous_seq:
                    st["duplicates"] += 1
                    return

                if seq < previous_seq:
                    st["resets"] += 1

                elif seq > previous_seq + 1:
                    st["lost"] += seq - previous_seq - 1

            st["last_seq"] = seq

            if not msg["sample_valid"] or age < 0 or age > 500:
                st["invalid_samples"] += 1
                return

            if st["last_rx_ms"] is not None:
                gap = rx - st["last_rx_ms"]
                st["max_gap_ms"] = max(st["max_gap_ms"], gap)

            msg["received_timestamp_ms"] = rx

            st["last_rx_ms"] = rx
            st["last"] = msg
            st["received"] += 1
            st["rx_times"].append(rx / 1000.0)
            st["status"] = "OK"

        self.log.write(msg)

    def _bad_message(self, reason):
        with self.s.lock:
            self.s.imu["bad_lines"] += 1

        if self.args.verbose:
            print(f"[IMU] Invalid message: {reason}")

    def run(self):

        print(f"[IMU] HTTP endpoint: {self.args.imu_url}")

        interval = self.args.imu_poll_ms / 1000.0

        while not self.s.quit.is_set():

            start = time.monotonic()

            try:
                request = urllib.request.Request(
                    self.args.imu_url,
                    headers={"Accept": "application/json"}
                )

                with urllib.request.urlopen(
                    request,
                    timeout=self.args.http_timeout
                ) as response:

                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP status {response.status}"
                        )

                    raw = response.read().decode("utf-8")

                msg = json.loads(raw)
                self.handle_message(msg)

            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                RuntimeError
            ) as error:

                with self.s.lock:
                    self.s.imu["status"] = "DISCONNECTED"

                if self.args.verbose:
                    print(f"[IMU] HTTP error: {error}")

            except (json.JSONDecodeError, UnicodeDecodeError) as error:

                self._bad_message(str(error))

            elapsed = time.monotonic() - start
            remaining = max(0.0, interval - elapsed)

            self.s.quit.wait(remaining)


# ============================================================
# Simulated IMU Worker
# ============================================================

class SimulatedImuWorker(threading.Thread):

    def __init__(self, shared, args, log):
        super().__init__(daemon=True)

        self.s = shared
        self.args = args
        self.log = log

    def run(self):

        print("[IMU] Running simulated IMU")

        seq = 0
        t0 = time.monotonic()

        while not self.s.quit.is_set():

            seq += 1

            if seq % 100 == 0:
                seq += 1

            t = time.monotonic() - t0

            msg = {
                "type": "imu_raw",
                "device_id": "smartcap_sim",
                "sequence_id": seq,
                "device_timestamp_ms": int(t * 1000),
                "device_clock_domain": "simulated_clock",
                "sample_age_ms": 1,
                "sample_valid": True,
                "ax_g": round(0.05 * math.sin(t), 5),
                "ay_g": round(0.05 * math.cos(t), 5),
                "az_g": -0.99658,
                "gx_dps": round(2.0 * math.sin(t), 3),
                "gy_dps": round(1.0 * math.cos(t), 3),
                "gz_dps": 0.5,
                "imu_status": "SIMULATED",
                "i2c_address": "0x68",
                "i2c_devices": ["0x68"],
                "calibration_status": "NOT_IMPLEMENTED",
            }

            # Simulate the same raw-message structure.
            HttpImuWorker(
                self.s, self.args, self.log
            ).handle_message(msg)

            self.s.quit.wait(0.02)


# ============================================================
# Camera Worker
# ============================================================

class CameraWorker(threading.Thread):

    def __init__(self, shared, args, log):
        super().__init__(daemon=True)

        self.s = shared
        self.args = args
        self.log = log

        self.frame_id = 0

    def run(self):

        import cv2

        print(f"[CAMERA] Stream: {self.args.camera_url}")

        st = self.s.cam

        while not self.s.quit.is_set():

            cap = cv2.VideoCapture(self.args.camera_url)

            if not cap.isOpened():

                with self.s.lock:
                    st["status"] = "DISCONNECTED"
                    st["reconnects"] += 1

                if self.args.verbose:
                    print("[CAMERA] Cannot open stream. Retrying...")

                cap.release()
                self.s.quit.wait(2)
                continue

            while not self.s.quit.is_set():

                ok, frame = cap.read()

                if not ok:
                    if self.args.verbose:
                        print("[CAMERA] Frame read failed")
                    break

                rx = now_ms()
                self.frame_id += 1

                height, width = frame.shape[:2]

                with self.s.lock:

                    st["frames"] += 1
                    st["last_rx_ms"] = rx
                    st["rx_times"].append(rx / 1000.0)
                    st["size"] = (width, height)
                    st["status"] = "OK"

                    fps = rate_hz(st["rx_times"])

                    if self.args.show:
                        st["frame"] = frame

                self.log.write({
                    "type": "camera_frame",
                    "frame_id": self.frame_id,
                    "received_timestamp_ms": rx,
                    "width": width,
                    "height": height,
                    "actual_fps": round(fps, 2),
                    "camera_status": "OK",
                })

            cap.release()

            with self.s.lock:
                st["status"] = "DISCONNECTED"
                st["reconnects"] += 1

            self.s.quit.wait(1)


# ============================================================
# Status and Timeout Handling
# ============================================================

def source_status(st, timeout_ms, enabled=True):

    if not enabled:
        return "OFF"

    if st["last_rx_ms"] is None:
        return st["status"]

    if now_ms() - st["last_rx_ms"] <= timeout_ms:
        return "OK"

    return "DISCONNECTED"


def print_status(shared, args, start_time):

    with shared.lock:
        imu = dict(shared.imu)
        cam = dict(shared.cam)

        imu_hz = rate_hz(imu["rx_times"])
        cam_fps = rate_hz(cam["rx_times"])

    imu_state = source_status(
        imu, args.imu_timeout_ms
    )

    cam_state = source_status(
        cam,
        args.camera_timeout_ms,
        enabled=not args.no_camera
    )

    last = imu["last"] or {}

    total = imu["received"] + imu["lost"]

    loss_percent = (
        100.0 * imu["lost"] / total
        if total else 0.0
    )

    problems = [
        name
        for name, state in (
            ("IMU", imu_state),
            ("CAMERA", cam_state)
        )
        if state in ("DISCONNECTED", "WAITING")
    ]

    gateway = (
        "UNKNOWN (" + ", ".join(problems) + " unavailable)"
        if problems else "SENSING_OK"
    )

    imu_text = (
        f"IMU {imu_state} "
        f"rx={imu_hz:.1f}Hz "
        f"received={imu['received']} "
        f"lost={imu['lost']} ({loss_percent:.1f}%) "
        f"duplicates={imu['duplicates']} "
        f"invalid={imu['invalid_samples']} "
        f"bad={imu['bad_lines']} "
        f"maxgap={imu['max_gap_ms']}ms "
        f"resets={imu['resets']} "
        f"seq={last.get('sequence_id', '-')} "
        f"age={last.get('sample_age_ms', '-')}ms "
        f"status={last.get('imu_status', '-')}"
    )

    cam_text = (
        "CAM OFF"
        if args.no_camera
        else (
            f"CAM {cam_state} "
            f"frames={cam['frames']} "
            f"fps={cam_fps:.1f} "
            f"size={cam['size']} "
            f"reconnects={cam['reconnects']}"
        )
    )

    elapsed = time.monotonic() - start_time

    print(
        f"[{elapsed:5.0f}s] "
        f"{imu_text} | {cam_text} | gateway={gateway}"
    )

    if problems:
        print(
            "  WARNING: A sensing source is unavailable. "
            "Driver state must remain UNKNOWN."
        )


# ============================================================
# Console
# ============================================================

def quit_console(shared):

    print("[CONSOLE] Press q + Enter to quit.")

    while not shared.quit.is_set():

        try:
            command = input().strip().lower()

        except EOFError:
            shared.quit.set()
            return

        if command == "q":
            shared.quit.set()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="SmartCap HTTP IMU + Camera Gateway"
    )

    parser.add_argument(
        "--imu-url",
        default=DEFAULT_IMU_URL,
        help="ESP32 IMU HTTP endpoint"
    )

    parser.add_argument(
        "--camera-url",
        default=DEFAULT_CAMERA_URL,
        help="ESP32 MJPEG camera endpoint"
    )

    parser.add_argument(
        "--imu-poll-ms",
        type=int,
        default=20,
        help="IMU HTTP polling interval"
    )

    parser.add_argument(
        "--http-timeout",
        type=float,
        default=1.0,
        help="HTTP request timeout in seconds"
    )

    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Run IMU only"
    )

    parser.add_argument(
        "--sim-imu",
        action="store_true",
        help="Use simulated IMU instead of ESP32 HTTP"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show camera preview"
    )

    parser.add_argument(
        "--imu-timeout-ms",
        type=int,
        default=500
    )

    parser.add_argument(
        "--camera-timeout-ms",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Auto-stop after N seconds; 0 means run until q"
    )

    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "output")
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    args = parser.parse_args()

    if args.imu_poll_ms < 10:
        parser.error("--imu-poll-ms must be at least 10")

    if args.http_timeout <= 0:
        parser.error("--http-timeout must be positive")

    if args.imu_timeout_ms <= 0 or args.camera_timeout_ms <= 0:
        parser.error("Timeouts must be positive")

    if args.show and args.no_camera:
        parser.error("--show cannot be used with --no-camera")

    try:
        import cv2
    except ImportError:
        if not args.no_camera:
            sys.exit(
                "OpenCV is missing. Install it using: "
                "pip install opencv-python"
            )

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    shared = Shared()

    imu_log = CsvLog(
        out / f"gateway_imu_{stamp}.csv"
    )

    cam_log = CsvLog(
        out / f"gateway_camera_meta_{stamp}.csv"
    )

    workers = []

    if args.sim_imu:
        imu_worker = SimulatedImuWorker(
            shared, args, imu_log
        )
    else:
        imu_worker = HttpImuWorker(
            shared, args, imu_log
        )

    workers.append(imu_worker)

    if not args.no_camera:
        camera_worker = CameraWorker(
            shared, args, cam_log
        )
        workers.append(camera_worker)

    start_time = time.monotonic()

    print("=" * 65)
    print("SMARTCAP GATEWAY")
    print("=" * 65)
    print(f"IMU mode: {'SIMULATED' if args.sim_imu else 'HTTP'}")

    if not args.sim_imu:
        print(f"IMU URL: {args.imu_url}")

    if not args.no_camera:
        print(f"Camera URL: {args.camera_url}")
    else:
        print("Camera: DISABLED")

    print(f"IMU polling interval: {args.imu_poll_ms} ms")
    print(f"Output directory: {out}")
    print("=" * 65)

    for worker in workers:
        worker.start()

    console = threading.Thread(
        target=quit_console,
        args=(shared,),
        daemon=True
    )
    console.start()

    next_status = time.monotonic() + 1.0

    try:

        while not shared.quit.is_set():

            if (
                args.duration > 0
                and time.monotonic() - start_time >= args.duration
            ):
                break

            if args.show and not args.no_camera:

                with shared.lock:
                    frame = shared.cam["frame"]

                if frame is not None:
                    cv2.imshow(
                        "SmartCap Gateway",
                        frame
                    )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            else:
                shared.quit.wait(0.05)

            if time.monotonic() >= next_status:
                print_status(shared, args, start_time)
                next_status = time.monotonic() + 1.0

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt")

    finally:

        shared.quit.set()

        for worker in workers:
            worker.join(timeout=2.0)

        imu_log.close()
        cam_log.close()

        if args.show:
            cv2.destroyAllWindows()

        print("\n" + "=" * 65)
        print("SMARTCAP GATEWAY STOPPED")
        print(f"IMU CSV rows: {imu_log.rows}")
        print(f"Camera CSV rows: {cam_log.rows}")
        print(f"Logs saved in: {out}")
        print("=" * 65)


if __name__ == "__main__":
    main()
