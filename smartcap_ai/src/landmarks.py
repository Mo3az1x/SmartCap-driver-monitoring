"""
Face landmarks -> per-frame numbers (EAR, MAR, head pose ratios).

One consistent definition used by data collection, training AND live inference.
Only numbers are produced - no image is ever stored (SmartCap spec section 14).
"""
import math
import os
import time
from pathlib import Path

import numpy as np

# MediaPipe face-mesh indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]     # corner, top1, top2, corner, bottom2, bottom1
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_CORNERS = (61, 291)
MOUTH_PAIRS = ((13, 14), (82, 87), (312, 317))   # inner-lip top/bottom pairs (vertical opening)
NOSE, FOREHEAD, CHIN = 1, 10, 152
OUTER_L, OUTER_R = 33, 263

ROW_FIELDS = ["t_s", "face", "ear", "ear_l", "ear_r", "mar", "yaw", "pitch", "roll",
              "face_w", "brightness"]

ESP32_DEFAULT_URL = "http://192.168.4.1/stream"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def _xy(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(lm, idx, w, h):
    p = [_xy(lm, i, w, h) for i in idx]
    horizontal = _d(p[0], p[3])
    if horizontal <= 1e-6:
        return 0.0
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * horizontal)


def mouth_aspect_ratio(lm, w, h):
    width = _d(_xy(lm, MOUTH_CORNERS[0], w, h), _xy(lm, MOUTH_CORNERS[1], w, h))
    if width <= 1e-6:
        return 0.0
    opens = [_d(_xy(lm, a, w, h), _xy(lm, b, w, h)) for a, b in MOUTH_PAIRS]
    return sum(opens) / (len(opens) * width)


def head_pose_ratios(lm, w, h):
    """Cheap head pose from 2D landmarks (no camera calibration needed).

    yaw   : nose offset from the eye midpoint / eye distance, clipped to [-1, 1] (0 = facing camera)
    pitch : nose position between forehead (0) and chin (1); changes when the head tilts up/down
    roll  : tilt of the line between the outer eye corners, degrees
    face_w: eye-corner distance as a fraction of the frame width (small = far away)
    """
    xa, ya = _xy(lm, OUTER_L, w, h)
    xb, yb = _xy(lm, OUTER_R, w, h)
    span = xb - xa
    if abs(span) <= 1e-6:
        return 0.0, 0.5, 0.0, 0.0
    mid_x = (xa + xb) / 2.0
    yaw = max(-1.0, min(1.0, (lm[NOSE].x * w - mid_x) / span))
    top, bottom = lm[FOREHEAD].y * h, lm[CHIN].y * h
    pitch = (lm[NOSE].y * h - top) / (bottom - top) if abs(bottom - top) > 1e-6 else 0.5
    roll = math.degrees(math.atan2(yb - ya, xb - xa))
    return yaw, pitch, roll, abs(span) / w


def features_from_landmarks(lm, w, h):
    ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
    ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
    yaw, pitch, roll, face_w = head_pose_ratios(lm, w, h)
    return {"face": 1, "ear": (ear_l + ear_r) / 2.0, "ear_l": ear_l, "ear_r": ear_r,
            "mar": mouth_aspect_ratio(lm, w, h), "yaw": yaw, "pitch": pitch, "roll": roll,
            "face_w": face_w}


def no_face_row():
    nan = float("nan")
    return {"face": 0, "ear": nan, "ear_l": nan, "ear_r": nan, "mar": nan, "yaw": nan,
            "pitch": nan, "roll": nan, "face_w": nan}


def parse_source(value):
    """'1' -> 1 (USB webcam index); 'esp32' -> default cap URL; 'http://...' -> that URL."""
    v = str(value).strip()
    if v.isdigit():
        return int(v)
    if v.lower() in ("esp", "esp32", "cap"):
        return ESP32_DEFAULT_URL
    if "://" not in v:                      # bare IP such as 192.168.4.1
        v = f"http://{v}/stream"
    return v


class UrlCamera:
    """MJPEG stream (ESP32 cap camera). A thread always keeps only the NEWEST frame, so the
    Wi-Fi stream never builds a queue (= no growing lag). Reconnects if the stream drops."""

    def __init__(self, url, timeout_s=10.0):
        import threading
        import cv2
        self._cv2, self.url = cv2, url
        self._lock = threading.Lock()
        self._frame, self._seq, self._read_seq = None, 0, 0
        self._stop = False
        self._cap = self._open()
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open {url}\n"
                "Check: laptop is connected to the Wi-Fi 'SmartCap-CAM', no other browser tab/app is "
                "showing the stream (the ESP32 serves only one client), and the IP is right.")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < timeout_s:
            time.sleep(0.05)
        if self._frame is None:
            self.release()
            raise RuntimeError(f"Connected to {url} but no frame arrived in {timeout_s:.0f} s.")

    def _open(self):
        cap = self._cv2.VideoCapture(self.url)
        cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        fails = 0
        while not self._stop:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                fails = 0
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
            else:
                fails += 1
                if fails >= 5:                      # stream dropped -> reconnect
                    self._cap.release()
                    time.sleep(1.0)
                    self._cap = self._open()
                    fails = 0
                else:
                    time.sleep(0.02)

    def read(self):
        # wait for a frame we have not returned yet (max 2 s)
        t0 = time.time()
        while time.time() - t0 < 2.0:
            with self._lock:
                if self._frame is not None and self._seq != self._read_seq:
                    self._read_seq = self._seq
                    return True, self._frame.copy()
            time.sleep(0.005)
        return False, None

    def release(self):
        self._stop = True
        try:
            self._cap.release()
        except Exception:
            pass


def open_camera(source=0, width=640, height=480):
    """source: int (USB / laptop webcam), URL string, 'esp32', or a bare IP."""
    import cv2
    source = parse_source(source) if not isinstance(source, int) else source
    if isinstance(source, str):
        return UrlCamera(source)          # frame size is set on the ESP32 itself (see README)
    index = source
    # DirectShow is Windows-only; on Linux / Raspberry Pi use the default backend (V4L2).
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Close other apps using it or try --camera 1.")
    if os.name == "nt":
        # MJPG lets most Windows webcams run at 30 fps instead of 5-15 fps (must be set before the size)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # no queue of old frames -> less lag
    return cap


class FaceFeatureExtractor:
    """MediaPipe Face Landmarker (pretrained) wrapped to return one feature row per frame."""

    def __init__(self, model_path=DEFAULT_MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")
        self._mp = mp
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last_ms = -1

    def process(self, frame_bgr):
        import cv2
        t_s = time.perf_counter() - self._t0
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int(t_s * 1000), self._last_ms + 1)      # must strictly increase
        self._last_ms = ts_ms
        result = self._landmarker.detect_for_video(image, ts_ms)
        row = {"t_s": t_s, "brightness": float(frame_bgr.mean())}
        if result.face_landmarks:
            row.update(features_from_landmarks(result.face_landmarks[0], w, h))
        else:
            row.update(no_face_row())
        return row

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
