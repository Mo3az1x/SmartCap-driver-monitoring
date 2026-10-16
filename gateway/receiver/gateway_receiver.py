#!/usr/bin/env python3
"""
SmartCap gateway receiver - first integration milestone.

    Camera stream + IMU stream + manual vibration command

What it does (SMARTCAP_INTERFACE_SPEC_v1):
  * Reads IMU JSON lines from the Arduino over USB Serial (spec section 7).
  * Reads the ESP32-S3 MJPEG stream and builds camera_frame metadata (section 8).
  * Stamps every message with received_timestamp_ms (laptop = reference clock, section 3).
  * Detects packet loss (sequence_id gaps) and source timeouts (section 12).
  * Sends vibration_command JSON to the Arduino (section 11), manually from the console.

It does NOT detect drowsiness. It never reports ATTENTIVE; when a source is down the
gateway state is UNKNOWN (spec section 10: missing data must not become ATTENTIVE).

Camera frames are NOT saved (spec section 14: raw video should not be stored).
Only metadata is logged.

Examples:
    python gateway_receiver.py --port COM5
    python gateway_receiver.py --port COM5 --camera-url http://192.168.4.1:81/stream --show
    python gateway_receiver.py --no-camera --port COM5
    python gateway_receiver.py --sim-imu --no-camera --duration 10     # no hardware needed

Console commands:   v <level 0-3> [duration_ms]   |   0 1 2 3   |   q
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

<<<<<<< Updated upstream
DEFAULT_CAMERA_URL = "http://192.168.4.1:81/stream"   # ESP32 soft-AP default IP; check the ESP32 Serial output
DEFAULT_BAUD = 230400                                  # must match the Arduino sketch (SERIAL_BAUD)
=======
from imu_protocol import (
    DEFAULT_PROFILE,
    IMU_REQUIRED_GATEWAY,
    get_serial_profile,
    validate_imu_message,
)
from source_health import sensing_availability, source_fresh
>>>>>>> Stashed changes

DEFAULT_CAMERA_URL = "http://192.168.4.1/stream"   # ESP32 soft-AP default IP; check the ESP32 Serial output
DEFAULT_BAUD = 230400                                  # haptic sketch default; use --profile led for 115200

IMU_REQUIRED = list(IMU_REQUIRED_GATEWAY)
VIB_LEVEL_NAMES = {0: "STOP", 1: "WARNING", 2: "HIGH_RISK", 3: "CRITICAL"}
VIB_REASONS = {
    "HIGH_RISK_DROWSINESS", "PROLONGED_EYE_CLOSURE", "REPEATED_NODDING",
    "COMMUNICATION_FAILURE", "SENSOR_FAILURE", "MANUAL_TEST",
}


def now_ms():
    return int(time.time() * 1000)


def rate_hz(times):
    if len(times) < 2 or times[-1] <= times[0]:
        return 0.0
    return (len(times) - 1) / (times[-1] - times[0])      # times are in seconds


class CsvLog:
    """Append-only CSV writer; header is taken from the first row."""

    def __init__(self, path, fieldnames=None):
        self.path = path
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None
        self._rows = 0

    def write(self, row):
        if self._writer is None:
            self.fieldnames = self.fieldnames or list(row.keys())
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file, fieldnames=self.fieldnames, extrasaction="ignore", restval=""
            )
            self._writer.writeheader()
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % 25 == 0:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.quit = threading.Event()
        self.imu = dict(status="WAITING", last_rx_ms=None, last_rx_mono=None, last_seq=None,
                        received=0, lost=0, resets=0, bad_lines=0, max_gap_ms=0,
                        rx_times=deque(maxlen=50), last=None)
        self.cam = dict(status="WAITING", last_rx_ms=None, last_rx_mono=None, frames=0,
                        reconnects=0, size=None, rx_times=deque(maxlen=30), frame=None)


def find_port():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    keys = ("arduino", "ch340", "usb serial", "usb-serial", "cp210")
    for p in ports:
        if any(k in (p.description or "").lower() for k in keys):
            return p.device
    return ports[0].device if ports else None


class ImuWorker(threading.Thread):
    def __init__(self, shared, args, log):
        super().__init__(daemon=True)
        self.s, self.args, self.log = shared, args, log
        self.conn = None
        self.write_lock = threading.Lock()
        self.command_id = 0
        self.cmd_log = None
        self.latest_path = getattr(args, "imu_latest", None)

    # ---------- commands (spec section 11) ----------
    def send_vibration(self, level, duration_ms=500, reason="MANUAL_TEST"):
        if level not in VIB_LEVEL_NAMES:
            print(f"Invalid level {level}; use 0-3.")
            return
        if reason not in VIB_REASONS:
            print(f"Invalid reason {reason}.")
            return
        self.command_id += 1
        cmd = {
            "type": "vibration_command", "command_id": self.command_id,
            "timestamp_ms": now_ms(), "level": level,
            "duration_ms": int(duration_ms), "reason": reason,
        }
        line = json.dumps(cmd, separators=(",", ":"))   # compact: the Arduino line buffer is small
        if self.cmd_log:
            self.cmd_log.write(dict(cmd, sent_ok=bool(self.conn) or self.args.sim_imu))
        if self.args.sim_imu:
            print(f">> [sim] vibration level {level} ({VIB_LEVEL_NAMES[level]}) for {duration_ms} ms")
            return
        with self.write_lock:
            try:
                self.conn.write((line + "\n").encode("utf-8"))
                self.conn.flush()
                print(f">> sent: level {level} ({VIB_LEVEL_NAMES[level]}), {duration_ms} ms, id={self.command_id}")
            except Exception as error:  # serial not open / disconnected
                print(f">> could not send command: {error}")

    # ---------- message handling ----------
    def handle_message(self, msg):
        st = self.s.imu
        ok, err = validate_imu_message(msg)
        if not ok:
            with self.s.lock:
                st["bad_lines"] += 1
            print(f"Ignored IMU message ({err})")
            return
        rx = now_ms()
        rx_mono = time.monotonic()
        msg["received_timestamp_ms"] = rx
        msg["received_clock_domain"] = "unix_epoch_ms"
        msg["device_clock_domain"] = "arduino_millis"
        session_id = getattr(self.args, "session_id", None)
        if session_id:
            msg["session_id"] = session_id
        seq = msg["sequence_id"]
        with self.s.lock:
            if st["last_seq"] is not None:
                if seq > st["last_seq"] + 1:
                    st["lost"] += seq - st["last_seq"] - 1
                elif seq <= st["last_seq"]:
                    st["resets"] += 1          # Arduino restarted: sequence went back
            if st["last_rx_ms"] is not None:
                st["max_gap_ms"] = max(st["max_gap_ms"], rx - st["last_rx_ms"])
            st["last_seq"], st["last_rx_ms"], st["last"] = seq, rx, msg
            st["last_rx_mono"] = rx_mono
            st["received"] += 1
            st["rx_times"].append(rx_mono)
            st["status"] = "OK"
        self.log.write(msg)
        if self.latest_path:
            try:
                tmp = str(self.latest_path) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(msg, fh)
                os.replace(tmp, self.latest_path)
            except OSError:
                pass

    # ---------- run loops ----------
    def run(self):
        (self._run_sim if self.args.sim_imu else self._run_serial)()

    def _run_sim(self):
        seq, t0 = 0, time.time()
        while not self.s.quit.is_set():
            seq += 1
            if seq % 100 == 0:
                seq += 1                       # simulate one lost packet
            t = time.time() - t0
            self.handle_message({
                "type": "imu", "device_id": "sim", "sequence_id": seq,
                "device_timestamp_ms": int(t * 1000), "pitch_deg": round(8 * (1 + 0.5 * (t % 4)), 2),
                "roll_deg": 0.0, "pitch_velocity_dps": 0.0, "head_down_duration_ms": 0,
                "nodding_score": 0.0, "motion_energy": 0.1, "imu_status": "OK",
                "calibration_status": "CALIBRATED", "imu_quality": 0.9,
            })
            time.sleep(0.04)                   # 25 Hz like the real sketch

    def _run_serial(self):
        import serial
        port = self.args.port or find_port()
        if not port:
            print("No serial port found. Use --port COM5 (or /dev/ttyUSB0).")
        while not self.s.quit.is_set():
            if self.conn is None:
                try:
                    self.conn = serial.Serial(port, self.args.baud, timeout=1)
                    time.sleep(2)              # Arduino resets when the port opens
                    self.conn.reset_input_buffer()
                    print(f"IMU serial open: {port} @ {self.args.baud}")
                except Exception as error:
                    with self.s.lock:
                        self.s.imu["status"] = "DISCONNECTED"
                    print(f"Cannot open serial port {port}: {error} (retrying)")
                    self.s.quit.wait(2)
                    continue
            try:
                raw = self.conn.readline().decode("utf-8", errors="replace").strip()
            except Exception as error:
                print(f"IMU serial lost: {error}")
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
                continue
            if not raw:
                continue
            if raw.startswith("#"):
                if self.args.verbose:
                    print(f"Arduino: {raw}")
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                with self.s.lock:
                    self.s.imu["bad_lines"] += 1
                continue
            if isinstance(msg, dict):
                self.handle_message(msg)


class CameraWorker(threading.Thread):
    def __init__(self, shared, args, log):
        super().__init__(daemon=True)
        self.s, self.args, self.log = shared, args, log
        self.frame_id = 0

    def run(self):
        import cv2
        st = self.s.cam
        while not self.s.quit.is_set():
            cap = cv2.VideoCapture(self.args.camera_url)
            if not cap.isOpened():
                with self.s.lock:
                    st["status"] = "DISCONNECTED"
                    st["reconnects"] += 1
                self.s.quit.wait(2)
                continue
            while not self.s.quit.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                rx = now_ms()
                rx_mono = time.monotonic()
                self.frame_id += 1
                h, w = frame.shape[:2]
                with self.s.lock:
                    st["frames"] += 1
                    st["last_rx_ms"] = rx
                    st["last_rx_mono"] = rx_mono
                    st["rx_times"].append(rx_mono)
                    st["size"] = (w, h)
                    st["status"] = "OK"
                    fps = rate_hz(st["rx_times"])
                    if self.args.show:
                        st["frame"] = frame
                # OpenCV VideoCapture does not expose ESP32 part headers.
                # Prefer vision_worker.py (canonical MJPEG client) for device timestamps.
                self.log.write({
                    "type": "camera_frame", "frame_id": self.frame_id,
                    "received_timestamp_ms": rx, "width": w, "height": h,
                    "actual_fps": round(fps, 2), "camera_status": "OK",
                    "metadata_status": "MISSING",
                    "camera_clock_domain": "unknown",
                    "received_clock_domain": "unix_epoch_ms",
                    "device_timestamp_ms": "",
                    "device_frame_id": "",
                })
            cap.release()
            with self.s.lock:
                st["status"] = "DISCONNECTED"
                st["reconnects"] += 1
            self.s.quit.wait(1)


def command_console(imu_worker, shared):
    help_text = "Commands: v <level 0-3> [duration_ms] | 0 1 2 3 | q (quit)"
    print(help_text)
    while not shared.quit.is_set():
        try:
            line = input().strip().lower()
        except EOFError:
            return
        if not line:
            continue
        parts = line.split()
        if parts[0] == "q":
            shared.quit.set()
        elif parts[0] in ("0", "1", "2", "3"):
            imu_worker.send_vibration(int(parts[0]))
        elif parts[0] == "v" and len(parts) >= 2 and parts[1].isdigit():
            duration = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 500
            imu_worker.send_vibration(int(parts[1]), duration)
        else:
            print(help_text)


def source_status(st, timeout_ms, enabled=True):
    """Spec section 12: no data within the timeout -> DISCONNECTED (monotonic clock)."""
    if not enabled:
        return "OFF"
    return source_fresh(
        st.get("last_rx_mono"),
        timeout_ms / 1000.0,
        never_received_status=st.get("status") or "WAITING",
    )


def print_status(shared, args, t0):
    with shared.lock:
        imu, cam = dict(shared.imu), dict(shared.cam)
        imu_hz, cam_fps = rate_hz(imu["rx_times"]), rate_hz(cam["rx_times"])
    imu_state = source_status(imu, args.imu_timeout_ms)
    cam_state = source_status(cam, args.camera_timeout_ms, enabled=not args.no_camera)
    last = imu["last"] or {}
    total = imu["received"] + imu["lost"]
    loss = 100.0 * imu["lost"] / total if total else 0.0
    gateway = sensing_availability(imu_state, cam_state, camera_enabled=not args.no_camera)

    imu_text = (f"IMU {imu_state} rx={imu_hz:.1f}Hz lost={imu['lost']} ({loss:.1f}%) "
                f"maxgap={imu['max_gap_ms']}ms resets={imu['resets']} "
                f"pitch={last.get('pitch_deg', '-')} q={last.get('imu_quality', '-')} "
                f"dev={last.get('imu_status', '-')}/{last.get('calibration_status', '-')}")
    cam_text = ("CAM OFF (use vision_worker as the MJPEG client)" if args.no_camera else
                f"CAM {cam_state} frames={cam['frames']} fps={cam_fps:.1f} size={cam['size']} reconnects={cam['reconnects']}")
    print(f"[{time.monotonic() - t0:5.0f}s] {imu_text} | {cam_text} | sensing={gateway}")
    if "DISCONNECTED" in (imu_state, cam_state):
        print("  WARNING: communication timeout - driver_state must be UNKNOWN (sensing is not ATTENTIVE)")


def main():
    p = argparse.ArgumentParser(description="SmartCap gateway receiver", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--port", default=None, help="IMU serial port, e.g. COM5 (auto-detect if omitted)")
    p.add_argument("--profile", default=DEFAULT_PROFILE, choices=("haptic", "led"),
                   help="firmware serial profile (haptic=230400/25Hz, led=115200/10Hz)")
    p.add_argument("--baud", type=int, default=None, help="override profile baud")
    p.add_argument("--camera-url", default=DEFAULT_CAMERA_URL)
    p.add_argument("--no-camera", action="store_true", help="IMU only (required when vision_worker owns /stream)")
    p.add_argument("--sim-imu", action="store_true", help="simulated IMU (no hardware)")
    p.add_argument("--show", action="store_true", help="preview window; keys 0-3 vibrate, q quits")
    p.add_argument("--imu-timeout-ms", type=int, default=500, help="initial value, measure experimentally")
    p.add_argument("--camera-timeout-ms", type=int, default=2000, help="initial value, measure experimentally")
    p.add_argument("--duration", type=float, default=0, help="auto-stop after N seconds (0 = run until q)")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "output"))
    p.add_argument("--session-id", default=None)
    p.add_argument("--imu-latest", default=None, help="atomic JSON snapshot of the last IMU message")
    p.add_argument("--verbose", action="store_true", help="print '#' comment lines from the Arduino")
    args = p.parse_args()
    profile = get_serial_profile(args.profile)
    if args.baud is None:
        args.baud = profile["baud"]
    print(f"IMU serial profile={profile['id']} baud={args.baud} ~{profile['output_hz']} Hz JSON "
          f"(sketch {profile['sketch']})")

    if not args.sim_imu:
        try:
            import serial  # noqa: F401
        except ImportError:
            sys.exit("pyserial is missing: pip install pyserial")
    if not args.no_camera:
        try:
            import cv2  # noqa: F401
        except ImportError:
            sys.exit("opencv is missing: pip install opencv-python")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    shared = Shared()
    imu_log = CsvLog(out / f"gateway_imu_{stamp}.csv")
    cam_log = CsvLog(out / f"gateway_camera_meta_{stamp}.csv")
    cmd_log = CsvLog(out / f"gateway_commands_{stamp}.csv")

    imu_worker = ImuWorker(shared, args, imu_log)
    imu_worker.cmd_log = cmd_log
    imu_worker.start()
    if not args.no_camera:
        CameraWorker(shared, args, cam_log).start()
    threading.Thread(target=command_console, args=(imu_worker, shared), daemon=True).start()

    print(f"Logging to {out}")
    if args.session_id:
        print(f"session_id={args.session_id}")
    t0 = time.monotonic()
    next_status = t0 + 1.0
    try:
        while not shared.quit.is_set():
            if args.duration and time.monotonic() - t0 >= args.duration:
                break
            if args.show and not args.no_camera:
                import cv2
                with shared.lock:
                    frame = shared.cam["frame"]
                if frame is not None:
                    cv2.imshow("SmartCap gateway (0-3 vibrate, q quit)", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("0"), ord("1"), ord("2"), ord("3")):
                    imu_worker.send_vibration(int(chr(key)))
            else:
                time.sleep(0.05)
            if time.monotonic() >= next_status:
                print_status(shared, args, t0)
                next_status += 1.0
    except KeyboardInterrupt:
        pass
    finally:
        shared.quit.set()
        imu_worker.send_vibration(0, 0, "MANUAL_TEST") if (imu_worker.conn and not args.sim_imu) else None
        time.sleep(0.2)
        for log in (imu_log, cam_log, cmd_log):
            log.close()
        print("Stopped. Logs saved in", out)
        sys.stdout.flush()
        os._exit(0)   # the console thread may be blocked in input(); exit without waiting for it


if __name__ == "__main__":
    main()