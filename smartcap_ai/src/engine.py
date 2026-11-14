"""
DriverStateEngine: per-frame rows -> `driver_state` message (SmartCap spec section 10).

Two interchangeable model back-ends (chosen by the file you pass to --model):
  *.joblib  RandomForest on 21 window features          (src/train.py)
  *.onnx    1D-CNN + GRU on the raw 10 s sequence        (src/train_dl.py, needs onnxruntime only)

Rules that are NOT left to the model (spec: missing data must never become ATTENTIVE):
  * no face for more than 1.5 s        -> UNKNOWN
  * not enough face frames in window   -> UNKNOWN
  * first seconds (window not full)    -> UNKNOWN (warming up)
  * model not confident enough         -> UNKNOWN

Vision-only for now: imu_quality is null until the IMU is fused (changing the spec needs team agreement).
"""
import json
import time
from pathlib import Path

import numpy as np

from windowing import FEATURE_NAMES, WindowFeaturizer

STATE_FOR_CLASS = {"attentive": "ATTENTIVE", "distracted": "DISTRACTED", "drowsy": "DROWSINESS_SUSPECTED"}


def risk_level_for(score):
    if score < 0.25:
        return "LOW"
    if score < 0.5:
        return "MEDIUM"
    if score < 0.8:
        return "HIGH"
    return "CRITICAL"


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class DriverStateEngine:
    def __init__(self, model_path, ema=0.4, min_conf=0.5, compute_every_s=0.5, no_face_limit_s=1.5):
        path = Path(model_path)
        self.session = self.model = None
        if path.suffix == ".onnx":
            import onnxruntime as ort
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.classes, self.steps = list(meta["classes"]), int(meta["steps"])
            params = meta["params"]
            self.backend = "deep_learning_onnx"
        else:
            import joblib
            bundle = joblib.load(path)
            self.model = bundle["model"]
            self.feature_names = bundle["feature_names"]
            self.classes = list(bundle["classes"])
            params = bundle["params"]
            self.backend = "random_forest"
        self.featurizer = WindowFeaturizer(**params)
        self.ema, self.min_conf = ema, min_conf
        self.compute_every_s, self.no_face_limit_s = compute_every_s, no_face_limit_s
        self.probs = None
        self.features = None
        self.fps = 0.0
        self._last_t = None
        self._last_compute_t = -1e9
        self._status = "WARMUP"
        self.message = self._unknown(["warming_up"])

    # ------------------------------------------------------------------
    def _predict(self, feats):
        if self.session is not None:
            seq = self.featurizer.sequence(self.steps)
            if seq is None:
                return None
            logits = self.session.run(None, {"seq": seq[None].astype(np.float32)})[0][0]
            return _softmax(logits.astype(np.float64))
        x = np.array([[feats[n] for n in self.feature_names]])
        return self.model.predict_proba(x)[0]

    def update(self, row):
        t = float(row["t_s"])
        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t)
            self.fps = inst if self.fps == 0 else self.fps + 0.1 * (inst - self.fps)
        self._last_t = t
        self.featurizer.push(row)

        if self.featurizer.no_face_duration() > self.no_face_limit_s:
            self.probs = self.features = None
            self._status = "NO_FACE"
            self.message = self._unknown(["face_not_visible"])
            return self.message

        if t - self._last_compute_t >= self.compute_every_s:
            self._last_compute_t = t
            self._status = self.featurizer.status()
            if self._status != "OK":
                self.probs = self.features = None
                reason = "warming_up" if self._status == "WARMUP" else "face_not_visible"
                self.message = self._unknown([reason])
            else:
                feats = self.featurizer.compute()
                p = self._predict(feats) if feats is not None else None
                if p is None:
                    self.probs = self.features = None
                    self.message = self._unknown(["features_unavailable"])
                else:
                    self.features = feats
                    self.probs = p if self.probs is None else self.probs + self.ema * (p - self.probs)
                    self.message = self._build()
        return self.message

    # ------------------------------------------------------------------
    def camera_quality(self):
        return round(float(self.featurizer.face_ratio() * min(1.0, self.fps / 10.0)), 2)

    def _unknown(self, evidence):
        return {"type": "driver_state", "timestamp_ms": int(time.time() * 1000), "state": "UNKNOWN",
                "risk_level": "UNKNOWN", "risk_score": None, "confidence": 0.0,
                "camera_quality": self.camera_quality() if self._last_t is not None else 0.0,
                "imu_quality": None, "evidence": evidence, "system_status": "DEGRADED",
                "probs": None, "source": "vision_only"}

    def _evidence(self):
        f, ev = self.features, []
        if f["longest_closure_s"] >= 1.5:
            ev.append("long_eye_closure")
        if f["perclos"] >= 0.30:
            ev.append("high_perclos")
        if f["longest_mouth_open_s"] >= 1.2:
            ev.append("yawning")
        if f["turned_frac"] >= 0.40:
            ev.append("head_turned")
        return ev

    def _build(self):
        probs = dict(zip(self.classes, (float(v) for v in self.probs)))
        best = max(probs, key=probs.get)
        conf = probs[best]
        risk = min(1.0, probs.get("drowsy", 0.0) + 0.4 * probs.get("distracted", 0.0))
        if conf < self.min_conf:
            msg = self._unknown(["low_confidence"] + self._evidence())
            msg["probs"] = probs
            return msg
        state = STATE_FOR_CLASS.get(best, "UNKNOWN")
        if state == "DROWSINESS_SUSPECTED" and risk >= 0.85:
            state = "HIGH_RISK"
        return {"type": "driver_state", "timestamp_ms": int(time.time() * 1000), "state": state,
                "risk_level": risk_level_for(risk), "risk_score": round(risk, 3),
                "confidence": round(conf, 3), "camera_quality": self.camera_quality(),
                "imu_quality": None, "evidence": self._evidence(), "system_status": "OK",
                "probs": {k: round(v, 3) for k, v in probs.items()}, "source": "vision_only"}
