#!/usr/bin/env python3
"""Supervisor for simultaneous IMU + vision logging.

Only vision_worker consumes the camera MJPEG stream. The IMU process is started
with --no-camera so the ESP32 one-client limit is respected.

This process reports sensing/component health. It does not emit ATTENTIVE.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from imu_protocol import DEFAULT_PROFILE, get_serial_profile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _python() -> str:
    return sys.executable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SmartCap session launcher (IMU serial + vision, one camera client)",
    )
    p.add_argument("--session-id", default=None, help="default: timestamp UTC")
    p.add_argument("--outdir", default=None, help="default: gateway/receiver/output/<session-id>")
    p.add_argument("--duration", type=float, default=0, help="seconds; 0 = until Ctrl+C")
    p.add_argument("--port", default=None)
    p.add_argument("--profile", default=DEFAULT_PROFILE, choices=("haptic", "led"))
    p.add_argument("--baud", type=int, default=None, help="override profile baud")
    p.add_argument("--sim-imu", action="store_true")
    p.add_argument("--camera-url", default="http://192.168.4.1/stream")
    p.add_argument("--video-path", default=None, help="optional file for vision DEMO_MODE")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--acquisition-only", action="store_true",
                   help="vision logs frames/metadata without MediaPipe (stream tests)")
    p.add_argument("--show", action="store_true")
    p.add_argument("--imu-timeout-ms", type=int, default=500)
    p.add_argument("--camera-timeout-s", type=float, default=2.0)
    p.add_argument("--fusion", action="store_true",
                   help="after stop, run one-shot rule fusion on latest JSON files")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    profile = get_serial_profile(args.profile)
    baud = args.baud if args.baud is not None else profile["baud"]
    session_id = args.session_id or datetime.now(timezone.utc).strftime("sess_%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) if args.outdir else HERE / "output" / session_id
    outdir.mkdir(parents=True, exist_ok=True)

    config = {
        "session_id": session_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "baud": baud,
        "port": args.port,
        "sim_imu": args.sim_imu,
        "camera_url": args.camera_url,
        "video_path": args.video_path,
        "vision_enabled": not args.no_vision,
        "camera_client": "vision_worker.py" if not args.no_vision else "none",
        "gateway_consumes_mjpeg": False,
        "esp32_single_client": True,
        "acquisition_only": args.acquisition_only,
        "duration_s": args.duration,
        "imu_timeout_ms": args.imu_timeout_ms,
        "camera_timeout_s": args.camera_timeout_s,
        "outdir": str(outdir),
        "clock_note": (
            "received_timestamp_ms is laptop Unix epoch ms. "
            "device_timestamp_ms IMU is Arduino millis(). "
            "device_timestamp_ms camera is ESP32 millis() when MJPEG headers are present. "
            "Clocks are not synchronized; do not treat differences as one-way latency."
        ),
        "sensing_vs_driver_state": (
            "Child processes report source health (SENSING_OK / UNKNOWN). "
            "That is not ATTENTIVE."
        ),
        "firmware_demo_mode": (
            "Arduino demoMode may vibrate from IMU-only heuristics until a "
            "vibration_command is received. That is not fused AI."
        ),
    }
    config_path = outdir / "session.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Session {session_id}")
    print(f"Config: {config_path}")
    print("Camera MJPEG client: vision_worker only" if not args.no_vision else "Vision disabled")

    imu_latest = outdir / "imu_latest.json"
    vision_latest = outdir / "vision_latest.json"

    imu_cmd = [
        _python(), "-u", str(HERE / "gateway_receiver.py"),
        "--no-camera",
        "--session-id", session_id,
        "--outdir", str(outdir),
        "--imu-timeout-ms", str(args.imu_timeout_ms),
        "--imu-latest", str(imu_latest),
        "--baud", str(baud),
        "--profile", args.profile,
    ]
    if args.sim_imu:
        imu_cmd.append("--sim-imu")
    elif args.port:
        imu_cmd.extend(["--port", args.port])
    if args.duration:
        imu_cmd.extend(["--duration", str(args.duration)])
    if args.verbose:
        imu_cmd.append("--verbose")

    vision_cmd = None
    if not args.no_vision:
        vision_cmd = [
            _python(), "-u", str(HERE / "vision_worker.py"),
            "--session-id", session_id,
            "--outdir", str(outdir),
            "--latest-json", str(vision_latest),
            "--camera-timeout-s", str(args.camera_timeout_s),
        ]
        if args.video_path:
            vision_cmd.extend(["--video-path", args.video_path])
        else:
            vision_cmd.extend(["--camera-url", args.camera_url])
        if args.acquisition_only:
            vision_cmd.append("--acquisition-only")
        if args.show:
            vision_cmd.append("--show")
        if args.duration:
            vision_cmd.extend(["--duration", str(args.duration)])

    procs = []
    logs = []
    try:
        imu_log = (outdir / "imu_process.log").open("w", encoding="utf-8")
        logs.append(imu_log)
        procs.append(subprocess.Popen(imu_cmd, cwd=str(HERE), stdout=imu_log, stderr=subprocess.STDOUT))
        if vision_cmd:
            vis_log = (outdir / "vision_process.log").open("w", encoding="utf-8")
            logs.append(vis_log)
            procs.append(subprocess.Popen(vision_cmd, cwd=str(HERE), stdout=vis_log, stderr=subprocess.STDOUT))

        deadline = time.monotonic() + args.duration if args.duration else None
        while True:
            if any(p.poll() is not None for p in procs):
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Interrupted — stopping children")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        t_end = time.monotonic() + 5
        for p in procs:
            remain = max(0.05, t_end - time.monotonic())
            try:
                p.wait(timeout=remain)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=3)
        for fh in logs:
            fh.close()

    # Do not treat a child crash as a driver state.
    results = []
    names = ["imu"] + (["vision"] if vision_cmd else [])
    for name, p in zip(names, procs):
        results.append({"component": name, "returncode": p.returncode})
        if p.returncode not in (0, None):
            print(f"COMPONENT_FAIL {name} returncode={p.returncode} (not a driver_state)")

    stop = {
        "session_id": session_id,
        "stopped_utc": datetime.now(timezone.utc).isoformat(),
        "components": results,
        "sensing_note": "component return codes are process health, not ATTENTIVE/DROWSY",
    }
    (outdir / "session_stop.json").write_text(json.dumps(stop, indent=2), encoding="utf-8")
    print(f"Stopped. Logs in {outdir}")

    if args.fusion:
        fusion_py = REPO / "edge_ai" / "fusion" / "src" / "rule_fusion.py"
        if imu_latest.exists() or vision_latest.exists():
            out_state = outdir / "driver_state_latest.json"
            cmd = [_python(), str(fusion_py), "--out", str(out_state)]
            if imu_latest.exists():
                cmd.extend(["--imu-json", str(imu_latest)])
            if vision_latest.exists():
                cmd.extend(["--vision-json", str(vision_latest)])
            subprocess.run(cmd, check=False)
            print(f"Fusion snapshot: {out_state}")
        else:
            print("Fusion skipped: no latest JSON files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
