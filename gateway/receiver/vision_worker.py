#!/usr/bin/env python3
"""
SmartCap vision worker - reads the ESP32 camera MJPEG stream, laptop webcam, or recorded video files,
and extracts visual driver-state FEATURES with the MediaPipe Face Landmarker (Tasks API, VIDEO mode).

It does NOT decide that the driver is drowsy. It only produces behavioural cues:
  EAR / MAR, eye-closure duration, time-weighted PERCLOS (30 s), blink rate,
  long closures, yawn duration + yawn count, head-turn ratio, and a vision_quality score.

Important: NO_FACE is "unknown", never "eyes closed" and never "attentive".

Run:
  python vision_worker.py --camera-url http://192.168.4.1/stream --show   # ESP32 (port 80!)
  python vision_worker.py --camera-url 0 --show                           # laptop webcam test
  python vision_worker.py --video-path demo_drowsy.mp4 --show             # DEMO_MODE video test

  Canonical MJPEG client for ESP32 HTTP streams: mjpeg_client.py (keeps X-Frame-Id /
X-Timestamp-Ms). OpenCV VideoCapture is used only for webcam indices and video files.
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from mjpeg_client import iter_mjpeg_url

MODEL_NAME = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe face-mesh indices: (corner, corner, top1, bottom1, top2, bottom2)
EYE_A = (33, 133, 159, 145, 160, 144)
EYE_B = (362, 263, 386, 374, 385, 380)
MOUTH_UP, MOUTH_LOW, MOUTH_L, MOUTH_R = 13, 14, 61, 291
NOSE_TIP = 1
OUTER_EYE_A, OUTER_EYE_B = 33, 263
OVERLAY_POINTS = sorted(set(EYE_A + EYE_B + (MOUTH_UP, MOUTH_LOW, MOUTH_L, MOUTH_R)))

# Same quality thresholds as the IMU firmware
DEGRADED_BELOW = 0.70
INVALID_BELOW = 0.30


def now_ms():
    return int(time.time() * 1000)


# ======================= landmark maths =======================

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pt(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h)


def eye_ear(lm, idx, w, h):
    p = [_pt(lm, i, w, h) for i in idx]
    horizontal = _dist(p[0], p[1])
    if horizontal <= 0:
        return 0.0
    return (_dist(p[2], p[3]) + _dist(p[4], p[5])) / (2.0 * horizontal)


def compute_ear(lm, w, h):
    """Returns (mean EAR, EAR eye A, EAR eye B)."""
    a, b = eye_ear(lm, EYE_A, w, h), eye_ear(lm, EYE_B, w, h)
    return (a + b) / 2.0, a, b


def compute_mar(lm, w, h):
    horizontal = _dist(_pt(lm, MOUTH_L, w, h), _pt(lm, MOUTH_R, w, h))
    if horizontal <= 0:
        return 0.0
    return _dist(_pt(lm, MOUTH_UP, w, h), _pt(lm, MOUTH_LOW, w, h)) / horizontal


def face_geometry(lm, w, h):
    """(face_width_ratio, head_yaw_ratio). Clipped between -1.0 and +1.0."""
    xa, xb = lm[OUTER_EYE_A].x, lm[OUTER_EYE_B].x
    span = abs(xb - xa)
    if span <= 1e-6:
        return 0.0, 0.0
    mid = (xa + xb) / 2.0
    raw_yaw = (lm[NOSE_TIP].x - mid) / (xb - xa)
    # Fix yaw spikes by clipping values to [-1.0, 1.0]
    yaw = max(-1.0, min(1.0, raw_yaw))
    return span, yaw


# ======================= temporal trackers =======================

# في EyeTracker داخل vision_worker.py:
# قم بتعديل reopen_factor من 1.10 إلى 1.02 لتسريع الخروج من حالة إغلاق العين
class EyeTracker:
    def __init__(self, threshold=0.18, window_s=30.0, ema_alpha=0.6, prolonged_s=1.5,
                 blink_min_s=0.05, blink_max_s=0.5, reopen_factor=1.02):
        self.threshold = threshold
        self.window_s = window_s
        self.alpha = ema_alpha
        self.prolonged_s = prolonged_s
        self.blink_min_s, self.blink_max_s = blink_min_s, blink_max_s
        self.reopen_factor = reopen_factor
        self.reset_all()

    def reset_all(self):
        self.ear_s = None
        self.closed = False
        self.closed_since = None
        self.closure_s = 0.0
        self.samples = deque()          # (t, dt, closed)
        self.last_t = None
        self.first_t = None
        self.blink_times = deque()
        self.blink_count = 0
        self.long_closure_count = 0
        self.last_blink_s = 0.0

    def set_threshold(self, thr):
        self.threshold = float(thr)

    def mark_no_face(self, no_face_duration=0.0):
        """Face lost: clear current running closure."""
        self.closed = False
        self.closed_since = None
        self.closure_s = 0.0
        self.last_t = None
        self.ear_s = None
        # If face lost for long duration (>2s), flush stale samples
        if no_face_duration > 2.0:
            self.samples.clear()

    def update(self, ear, t):
        self.ear_s = ear if self.ear_s is None else self.ear_s + self.alpha * (ear - self.ear_s)
        if self.first_t is None:
            self.first_t = t
        limit = self.threshold * (self.reopen_factor if self.closed else 1.0)
        now_closed = self.ear_s < limit

        if now_closed and not self.closed:
            self.closed, self.closed_since = True, t
        elif not now_closed and self.closed:
            dur = t - self.closed_since
            if self.blink_min_s <= dur <= self.blink_max_s:
                self.blink_count += 1
                self.blink_times.append(t)
                self.last_blink_s = dur
            elif dur > self.blink_max_s:
                self.long_closure_count += 1
            self.closed, self.closed_since = False, None
        self.closure_s = (t - self.closed_since) if self.closed else 0.0

        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        if 0.0 < dt <= 1.0:                          # ignore gaps (lost frames)
            self.samples.append((t, dt, self.closed))
        while self.samples and self.samples[0][0] < t - self.window_s:
            self.samples.popleft()
        while self.blink_times and self.blink_times[0] < t - 60.0:
            self.blink_times.popleft()
        return self.snapshot(t)

    def snapshot(self, t):
        total = sum(s[1] for s in self.samples)
        closed = sum(s[1] for s in self.samples if s[2])
        span = 0.0 if self.first_t is None else min(60.0, t - self.first_t)
        rate = len(self.blink_times) * 60.0 / span if span >= 10.0 else 0.0
        return {
            "ear_smooth": self.ear_s or 0.0,
            "eye_state": "CLOSED" if self.closed else "OPEN",
            "closure_duration_s": self.closure_s,
            "perclos": (closed / total) if total > 0 else 0.0,
            "blink_rate_per_min": rate,
            "blink_count": self.blink_count,
            "long_closure_count": self.long_closure_count,
            "prolonged_eye_closure": self.closure_s >= self.prolonged_s,
        }


class YawnTracker:
    """A yawn = mouth wide open for at least min_s. Counted when the mouth closes again."""

    def __init__(self, threshold=0.50, min_s=1.2, ema_alpha=0.5):
        self.threshold, self.min_s, self.alpha = threshold, min_s, ema_alpha
        self.count = 0
        self.reset()

    def reset(self):
        self.mar_s = None
        self.active = False
        self.start = None
        self.duration_s = 0.0

    def update(self, mar, t):
        self.mar_s = mar if self.mar_s is None else self.mar_s + self.alpha * (mar - self.mar_s)
        if self.mar_s > self.threshold:
            if not self.active:
                self.active, self.start = True, t
            self.duration_s = t - self.start
        else:
            if self.active and self.duration_s >= self.min_s:
                self.count += 1
            self.active, self.start, self.duration_s = False, None, 0.0
        return {
            "mar_smooth": self.mar_s,
            "yawn_active": self.active,
            "yawn_duration_s": self.duration_s,
            "yawn_count": self.count,
            "prolonged_yawn": self.active and self.duration_s >= self.min_s,
        }


class EarCalibrator:
    """Personal open-eye EAR from the first N face frames (eyes assumed open)."""

    def __init__(self, frames=60, ratio=0.72, lo=0.12, hi=0.21):
        self.frames, self.ratio, self.lo, self.hi = frames, ratio, lo, hi
        self.restart()

    def restart(self):
        self.values = []
        self.done = False
        self.baseline = None
        self.threshold = None

    def add(self, ear):
        if self.done:
            return True
        if ear > 0:
            self.values.append(ear)
        if len(self.values) >= self.frames:
            vals = sorted(self.values)
            top = vals[int(0.3 * len(vals)):]          # drop the lowest 30 % (blinks)
            self.baseline = float(np.median(top))
            self.threshold = min(self.hi, max(self.lo, self.baseline * self.ratio))
            self.done = True
        return self.done

    @property
    def progress(self):
        return min(1.0, len(self.values) / float(self.frames))


def quality_score(face, face_w, yaw, brightness, fps, ear):
    if not face:
        return 0.0
    q = 1.0
    if face_w < 0.07:
        q -= 0.6
    elif face_w < 0.10:
        q -= 0.4
    if brightness < 40 or brightness > 220:
        q -= 0.3
    if abs(yaw) > 0.35:
        q -= 0.3
    if 0 < fps < 5:
        q -= 0.2
    if ear <= 0:
        q -= 0.3
    return max(0.0, min(1.0, q))


def status_from_quality(q):
    if q < INVALID_BELOW:
        return "INVALID"
    if q < DEGRADED_BELOW:
        return "DEGRADED"
    return "OK"


# ======================= I/O helpers =======================

class CsvLog:
    def __init__(self, path, fieldnames):
        self.path, self.fieldnames = path, fieldnames
        self.file = self.writer = None
        self.n = 0

    def write(self, row):
        if self.file is None:
            self.file = open(self.path, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="ignore")
            self.writer.writeheader()
        self.writer.writerow(row)
        self.n += 1
        if self.n % 15 == 0:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def write_json_atomic(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)

def _cv2_clock_meta():
    return {
        "received_timestamp_ms": now_ms(),
        "device_timestamp_ms": None,
        "device_frame_id": None,
        "metadata_status": "MISSING",
        "camera_clock_domain": "unknown",
        "received_clock_domain": "unix_epoch_ms",
    }

class FrameGrabber(threading.Thread):
    """Reads stream/webcam/video in its own thread and keeps ONLY the newest frame."""

    def __init__(self, source, is_video_file=False):
        super().__init__(daemon=True)
        self.source = source
        self.is_video_file = is_video_file
        self.use_mjpeg_headers = (
            not is_video_file
            and isinstance(source, str)
            and source.lower().startswith(("http://", "https://"))
        )
        self.lock = threading.Lock()
        self.frame, self.frame_id, self.frame_t = None, 0, 0.0
        self.meta = _cv2_clock_meta()
        self.status = "CONNECTING"
        self.reconnects = 0
        self.stop_flag = threading.Event()

    def _open(self):
        if isinstance(self.source, int):
            api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            return cv2.VideoCapture(self.source, api)
        return cv2.VideoCapture(self.source)

    def _store(self, frame, meta):
        with self.lock:
            self.frame = frame
            self.frame_t = time.time()
            self.frame_id += 1
            self.meta = meta
            self.status = "OK"


    def run(self):
        if self.use_mjpeg_headers:
            self._run_mjpeg()
        else:
            self._run_cv2()
    def _run_mjpeg(self):
        backoff = 1.0
        while not self.stop_flag.is_set():
            try:
                for part in iter_mjpeg_url(self.source, stop_flag=self.stop_flag):
                    arr = np.frombuffer(part.jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    self._store(frame, {
                        "received_timestamp_ms": part.received_timestamp_ms,
                        "device_timestamp_ms": part.device_timestamp_ms,
                        "device_frame_id": part.device_frame_id,
                        "metadata_status": part.metadata_status,
                        "camera_clock_domain": part.camera_clock_domain,
                        "received_clock_domain": part.gateway_clock_domain,
                    })
                    backoff = 1.0
            except Exception:
                pass
            self.status = "DISCONNECTED"
            self.reconnects += 1
            self.stop_flag.wait(backoff)
            backoff = min(5.0, backoff * 1.5)
    def _run_cv2(self):
        backoff = 1.0
        while not self.stop_flag.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.status = "DISCONNECTED"
                self.reconnects += 1
                self.stop_flag.wait(backoff)
                backoff = min(5.0, backoff * 1.5)
                continue
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            backoff = 1.0
            self.status = "OK"
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                self._store(frame, _cv2_clock_meta())
                if self.is_video_file:
                    time.sleep(0.033)
            cap.release()
            self.status = "DISCONNECTED"
            self.reconnects += 1
            self.stop_flag.wait(1.0)

    def get(self, last_id):
        with self.lock:
            if self.frame is None or self.frame_id == last_id:
               return None, last_id, self.frame_t, None
            return self.frame, self.frame_id, self.frame_t, dict(self.meta)

    def stop(self):
        self.stop_flag.set()


def resolve_model(arg_model, allow_download):
    here = Path(__file__).resolve().parent
    candidates = [Path(arg_model)] if arg_model else [
        here / "models" / MODEL_NAME, here.parent / "models" / MODEL_NAME,
        here / MODEL_NAME, Path.cwd() / "models" / MODEL_NAME, Path.cwd() / MODEL_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    target = candidates[0] if arg_model else here / "models" / MODEL_NAME
    if allow_download:
        print(f"Model not found. Downloading {MODEL_NAME} to {target} ...")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(target)
            return target
        except Exception as error:
            print(f"Download failed: {error}")
    sys.exit(f"Face Landmarker model not found.\nDownload it from:\n  {MODEL_URL}\n"
             f"and put it at:\n  {target}\n(or pass --model PATH)")


CSV_FIELDS = [
    "timestamp_ms", "received_timestamp_ms", "device_timestamp_ms", "device_frame_id",
    "metadata_status", "camera_clock_domain", "received_clock_domain", "session_id",
    "acquisition_only", "frame_id", "face_detected", "ear", "ear_smooth", "mar", "mar_smooth",
    "eye_state", "closure_duration_s", "perclos_30s", "blink_rate_per_min", "blink_count",
    "long_closure_count", "prolonged_eye_closure", "yawn_active", "yawn_duration_s",
    "yawn_count", "prolonged_yawn", "head_yaw_ratio", "face_width_ratio", "brightness",
    "no_face_duration_s", "ear_threshold", "processing_fps", "vision_status", "vision_quality",
]


# ======================= worker =======================

class VisionWorker:
    def __init__(self, args):
        self.args = args
        self.acquisition_only = bool(getattr(args, "acquisition_only", False))
        self.session_id = getattr(args, "session_id", None)
        self.model_path = None if self.acquisition_only else resolve_model(
            args.model, not args.no_download
        )
        self.landmarker = None
        self.last_ts_ms = -1
        self.t0 = time.perf_counter()

        self.eyes = EyeTracker(args.ear_closed_threshold, args.perclos_window_s,
                               prolonged_s=args.prolonged_closure_s)
        self.yawn = YawnTracker(args.mar_yawn_threshold, args.yawn_min_s)
        self.calib = None if args.no_calibration else EarCalibrator(args.calib_frames, args.ear_ratio)
        self.fps = 0.0
        self.last_proc_t = None
        self.no_face_since = None
        self.last_frame_wall = 0.0
        self.last_json_t = 0.0
        self.state = {}

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = outdir / f"vision_features_{stamp}.csv"
        self.log = CsvLog(self.log_path, CSV_FIELDS)
        self.latest_path = (outdir / "vision_latest.json") if args.latest_json == "auto" \
            else (Path(args.latest_json) if args.latest_json else None)

    # ---- MediaPipe ----
    def _create_landmarker(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            sys.exit("mediapipe (>=0.10) is missing: pip install mediapipe")
        self.mp = mp
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def _detect(self, frame):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.perf_counter() - self.t0) * 1000)
        self.last_ts_ms = max(ts, self.last_ts_ms + 1)        # must strictly increase
        return self.landmarker.detect_for_video(image, self.last_ts_ms)

    # ---- per frame ----
    def process(self, frame, frame_id):
        t = time.perf_counter() - self.t0
        if self.last_proc_t is not None and t > self.last_proc_t:
            inst = 1.0 / (t - self.last_proc_t)
            self.fps = inst if self.fps == 0 else self.fps + 0.1 * (inst - self.fps)
        self.last_proc_t = t

        h, w = frame.shape[:2]
        brightness = float(cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0])
        result = self._detect(frame)

        s = {"timestamp_ms": now_ms(), "frame_id": frame_id, "brightness": brightness,
             "processing_fps": self.fps, "pts": [], "acquisition_only": False}

        if not result.face_landmarks:
            if self.no_face_since is None:
                self.no_face_since = t
            no_face_duration = t - self.no_face_since
            self.eyes.mark_no_face(no_face_duration)
            self.yawn.reset()

            s.update(face_detected=False, ear=0.0, mar=0.0, head_yaw_ratio=0.0, face_width_ratio=0.0,
                     no_face_duration_s=no_face_duration, vision_quality=0.0,
                     vision_status="NO_FACE", ear_threshold=self.eyes.threshold,
                     **self.eyes.snapshot(t), **self._yawn_snapshot())
            s["eye_state"] = "UNKNOWN"
            return s

        # Auto-recalibrate if face returns after prolonged absence (>5s)
        if self.no_face_since is not None and (t - self.no_face_since) > 5.0 and self.calib is not None:
            self.calib.restart()
            self.eyes.reset_all()

        self.no_face_since = None
        lm = result.face_landmarks[0]
        ear, _, _ = compute_ear(lm, w, h)
        mar = compute_mar(lm, w, h)
        face_w, yaw = face_geometry(lm, w, h)
        s["pts"] = [(int(lm[i].x * w), int(lm[i].y * h)) for i in OVERLAY_POINTS]

        calibrating = self.calib is not None and not self.calib.done
        if calibrating:
            if self.calib.add(ear):
                self.eyes.set_threshold(self.calib.threshold)
                self.eyes.reset_all()
                print(f"EAR calibration done: baseline={self.calib.baseline:.3f} "
                      f"threshold={self.calib.threshold:.3f}")
            eye = self.eyes.snapshot(t)
            eye["eye_state"] = "CALIBRATING"
            eye["prolonged_eye_closure"] = False
        else:
            eye = self.eyes.update(ear, t)
        yw = self.yawn.update(mar, t)

        q = quality_score(True, face_w, yaw, brightness, self.fps, ear)
        status = "CALIBRATING" if calibrating else status_from_quality(q)
        s.update(face_detected=True, ear=ear, mar=mar, head_yaw_ratio=yaw, face_width_ratio=face_w,
                 no_face_duration_s=0.0, vision_quality=q, vision_status=status,
                 ear_threshold=self.eyes.threshold, **eye, **yw)
        return s

    def process_acquisition(self, frame, frame_id):
        """Log camera timing/health without MediaPipe. Not a driver_state."""
        t = time.perf_counter() - self.t0
        if self.last_proc_t is not None and t > self.last_proc_t:
            inst = 1.0 / (t - self.last_proc_t)
            self.fps = inst if self.fps == 0 else self.fps + 0.1 * (inst - self.fps)
        self.last_proc_t = t
        h, w = frame.shape[:2]
        brightness = float(cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0])
        return {
            "timestamp_ms": now_ms(),
            "frame_id": frame_id,
            "brightness": brightness,
            "processing_fps": self.fps,
            "pts": [],
            "face_detected": False,
            "ear": 0.0,
            "mar": 0.0,
            "head_yaw_ratio": 0.0,
            "face_width_ratio": 0.0,
            "no_face_duration_s": 0.0,
            "vision_quality": 0.0,
            "vision_status": "OK",
            "ear_threshold": self.eyes.threshold,
            "acquisition_only": True,
            **self.eyes.snapshot(t),
            **self._yawn_snapshot(),
            "eye_state": "UNKNOWN",
        }
    def attach_clock(self, s, meta):
        meta = meta or {}
        s["received_timestamp_ms"] = meta.get("received_timestamp_ms") or s.get("timestamp_ms")
        s["device_timestamp_ms"] = meta.get("device_timestamp_ms")
        s["device_frame_id"] = meta.get("device_frame_id")
        s["metadata_status"] = meta.get("metadata_status") or "MISSING"
        s["camera_clock_domain"] = meta.get("camera_clock_domain") or "unknown"
        s["received_clock_domain"] = meta.get("received_clock_domain") or "unix_epoch_ms"
        s["session_id"] = self.session_id
        s["acquisition_only"] = bool(s.get("acquisition_only"))
        return s

    def _yawn_snapshot(self):
        return {"mar_smooth": 0.0, "yawn_active": False, "yawn_duration_s": 0.0,
                "yawn_count": self.yawn.count, "prolonged_yawn": False}

    # ---- output ----
    def publish(self, s):
        self.state = s
        row = dict(s)
        row["perclos_30s"] = s.get("perclos", 0.0)
        for k, v in list(row.items()):
            if isinstance(v, float):
                row[k] = round(v, 4)
        self.log.write(row)
        
        # Immediate export upon prolonged closure/yawn trigger
        urgent_event = s.get("prolonged_eye_closure") or s.get("prolonged_yawn")
        if self.latest_path and (urgent_event or (time.time() - self.last_json_t >= 0.1)):
            self.last_json_t = time.time()
            out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "pts"}
            out["type"] = "vision_features"
            try:
                write_json_atomic(self.latest_path, out)
            except OSError:
                pass

    def camera_down_state(self, status):
        t = time.perf_counter() - self.t0
        self.eyes.mark_no_face()
        self.yawn.reset()
        return {"timestamp_ms": now_ms(), "frame_id": 0, "face_detected": False, "ear": 0.0,
                "mar": 0.0, "head_yaw_ratio": 0.0, "face_width_ratio": 0.0, "brightness": 0.0,
                "no_face_duration_s": 0.0, "vision_quality": 0.0, "vision_status": status,
                "ear_threshold": self.eyes.threshold, "processing_fps": 0.0, "pts": [],
                "acquisition_only": self.acquisition_only,
                **self.eyes.snapshot(t), **self._yawn_snapshot(), "eye_state": "UNKNOWN"}

    def draw_overlay(self, frame, s):
        status = s.get("vision_status", "")
        color = {"OK": (0, 255, 0), "DEGRADED": (0, 200, 255), "CALIBRATING": (255, 200, 0)}.get(
            status, (0, 0, 255))
        for p in s.get("pts", []):
            cv2.circle(frame, p, 2, (0, 255, 255), -1)
        lines = [
            f"Status: {status}  Q={s.get('vision_quality', 0):.2f}",
            f"EAR: {s.get('ear', 0):.3f}  thr={s.get('ear_threshold', 0):.3f}",
            f"MAR: {s.get('mar', 0):.3f}",
            f"Eye: {s.get('eye_state', '-')}  Closure: {s.get('closure_duration_s', 0):.2f}s",
            f"PERCLOS-30s: {s.get('perclos', 0):.2f}",
            f"Blinks/min: {s.get('blink_rate_per_min', 0):.0f}  Long closures: {s.get('long_closure_count', 0)}",
            f"Yawn: {s.get('yawn_duration_s', 0):.1f}s  Yawns: {s.get('yawn_count', 0)}",
            f"Yaw: {s.get('head_yaw_ratio', 0):+.2f}  Face: {s.get('face_width_ratio', 0):.2f}",
            f"FPS: {s.get('processing_fps', 0):.1f}",
        ]
        y = 25
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y += 22
        hh, ww = frame.shape[:2]
        if self.calib is not None and not self.calib.done and s.get("face_detected"):
            cv2.putText(frame, f"Calibrating: look at the road, eyes open ({self.calib.progress * 100:.0f}%)",
                        (10, hh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)
        if s.get("prolonged_eye_closure"):
            cv2.rectangle(frame, (0, 0), (ww - 1, hh - 1), (0, 0, 255), 8)
            cv2.putText(frame, "PROLONGED EYE CLOSURE", (10, hh - 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 3, cv2.LINE_AA)
        elif s.get("prolonged_yawn"):
            cv2.putText(frame, "PROLONGED YAWN", (10, hh - 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 3, cv2.LINE_AA)
        return frame

    # ---- main loop ----
    def run(self):
        if not self.acquisition_only:
            self._create_landmarker()

        is_video_file = False
        if self.args.video_path:
            src = self.args.video_path
            is_video_file = True
        else:
            src = self.args.camera_url
            src = int(src) if str(src).isdigit() else src

        grabber = FrameGrabber(src, is_video_file=is_video_file)
        grabber.start()
        print(f"Vision worker started: {src}")
        if self.model_path:
            print(f"Model: {self.model_path}")
        if self.acquisition_only:
            print("acquisition-only: MediaPipe skipped (camera metadata / stream test)")
        print(f"Logging to: {self.log_path}")
        if self.calib is not None and not self.acquisition_only:
            print("EAR calibration: look at the road with eyes open until it finishes (key 'c' = redo).")

        last_id = 0
        last_down_log = 0.0
        deadline = None
        if getattr(self.args, "duration", 0):
            deadline = time.monotonic() + float(self.args.duration)
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                frame, last_id_new, ft, meta = grabber.get(last_id)
                if frame is None:
                    if ft > 0:
                        self.last_frame_wall = ft
                    age = time.time() - (self.last_frame_wall or 0)
                    waiting = self.last_frame_wall == 0.0
                    if (not waiting and age > self.args.camera_timeout_s) or grabber.status == "DISCONNECTED":
                        if time.time() - last_down_log >= 1.0:
                            last_down_log = time.time()
                            down = self.camera_down_state(
                                "WAITING_CAMERA" if waiting else "CAMERA_DISCONNECTED")
                            self.publish(self.attach_clock(down, None))
                    if self.args.show:
                        key = cv2.waitKey(5) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    else:
                        time.sleep(0.005)
                    continue
                last_id = last_id_new
                self.last_frame_wall = ft
                if self.args.flip:
                    frame = cv2.flip(frame, 1)
                if self.acquisition_only:
                    s = self.process_acquisition(frame, last_id)
                else:
                    s = self.process(frame, last_id)
                self.publish(self.attach_clock(s, meta))
                if self.args.show:
                    cv2.imshow("SmartCap Vision Worker", self.draw_overlay(frame, s))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("c") and self.calib is not None:
                        self.calib.restart()
                        self.eyes.reset_all()
                        print("EAR calibration restarted")
        except KeyboardInterrupt:
            pass
        finally:
            grabber.stop()
            self.log.close()
            try:
                if self.landmarker is not None:
                    self.landmarker.close()
            except Exception:
                pass
            cv2.destroyAllWindows()
            print("Vision worker stopped.")


def build_parser():
    p = argparse.ArgumentParser(description="SmartCap vision worker")
    p.add_argument("--camera-url", default="http://192.168.4.1/stream",
                   help="ESP32 MJPEG stream URL (port 80), or a webcam index like 0")
    p.add_argument("--video-path", default=None, help="path to a recorded video file for DEMO_MODE testing")
    p.add_argument("--model", default=None, help="path to face_landmarker.task (auto-searched/downloaded)")
    p.add_argument("--no-download", action="store_true", help="do not auto-download the model")
    p.add_argument("--show", action="store_true", help="live preview window (q/ESC quit, c recalibrate)")
    p.add_argument("--flip", action="store_true", help="mirror the image horizontally")
    p.add_argument("--ear-closed-threshold", type=float, default=0.20,
                   help="fixed EAR threshold (used until/unless calibration replaces it)")
    p.add_argument("--no-calibration", action="store_true", help="skip the personal EAR calibration")
    p.add_argument("--calib-frames", type=int, default=60, help="face frames used for EAR calibration")
    p.add_argument("--ear-ratio", type=float, default=0.72, help="threshold = open-eye EAR x this ratio")
    p.add_argument("--mar-yawn-threshold", type=float, default=0.50)
    p.add_argument("--yawn-min-s", type=float, default=1.2, help="mouth open this long = yawn")
    p.add_argument("--prolonged-closure-s", type=float, default=1.5)
    p.add_argument("--perclos-window-s", type=float, default=30.0)
    p.add_argument("--camera-timeout-s", type=float, default=2.0)
    p.add_argument("--latest-json", default="auto",
                   help="file with the latest features for other processes ('' to disable)")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "output"))
    p.add_argument("--session-id", default=None)
    p.add_argument("--duration", type=float, default=0, help="auto-stop after N seconds (0 = until q)")
    p.add_argument("--acquisition-only", action="store_true",
                   help="log frames/timestamps without MediaPipe (stream / session tests)")
    return p


def main():
    args = build_parser().parse_args()
    VisionWorker(args).run()


if __name__ == "__main__":
    main()