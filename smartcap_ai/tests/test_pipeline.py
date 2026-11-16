"""
Run:  python tests/test_pipeline.py
Checks (no camera needed): landmark maths, training on FAKE data, engine behaviour incl. UNKNOWN rules.
"""
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine import DriverStateEngine  # noqa: E402
from landmarks import (LEFT_EYE, MOUTH_CORNERS, MOUTH_PAIRS, eye_aspect_ratio,  # noqa: E402
                       features_from_landmarks, mouth_aspect_ratio)


def fake_landmarks():
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    return lm


def test_ear():
    lm = fake_landmarks()
    # 100x100 frame; eye 20 px wide, each vertical gap 6 px -> EAR = (6+6)/(2*20) = 0.30
    lm[33].x, lm[33].y = 0.30, 0.50
    lm[133].x, lm[133].y = 0.50, 0.50
    lm[160].x, lm[160].y = 0.36, 0.47
    lm[144].x, lm[144].y = 0.36, 0.53
    lm[158].x, lm[158].y = 0.44, 0.47
    lm[153].x, lm[153].y = 0.44, 0.53
    ear = eye_aspect_ratio(lm, LEFT_EYE, 100, 100)
    assert abs(ear - 0.30) < 1e-6, ear
    print("ok  EAR maths:", round(ear, 3))


def test_mar():
    lm = fake_landmarks()
    lm[MOUTH_CORNERS[0]].x, lm[MOUTH_CORNERS[0]].y = 0.40, 0.70
    lm[MOUTH_CORNERS[1]].x, lm[MOUTH_CORNERS[1]].y = 0.60, 0.70   # width 20 px
    for k, (a, b) in enumerate(MOUTH_PAIRS):
        lm[a].x = lm[b].x = 0.45 + 0.05 * k
        lm[a].y, lm[b].y = 0.67, 0.73                              # opening 6 px each
    mar = mouth_aspect_ratio(lm, 100, 100)
    assert abs(mar - 0.30) < 1e-6, mar
    f = features_from_landmarks(lm, 100, 100)
    assert f["face"] == 1 and math.isfinite(f["yaw"])
    print("ok  MAR maths:", round(mar, 3))


def test_train_and_engine():
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        model = Path(tmp) / "model.joblib"
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_data.py"), "--out", str(raw),
                        "--participants", "4"], check=True, capture_output=True)
        res = subprocess.run([sys.executable, str(ROOT / "src" / "train.py"), "--data", str(raw),
                              "--out", str(model)], check=True, capture_output=True, text=True)
        print(res.stdout.strip().splitlines()[0])
        assert model.exists()

        df = pd.read_csv(raw / "P01_synthetic.csv")
        counts = {}
        for label in ("attentive", "distracted", "drowsy"):
            eng = DriverStateEngine(model)
            states = []
            for rec in df[df.label == label].to_dict("records"):
                states.append(eng.update(rec)["state"])
            tail = states[len(states) // 3:]
            counts[label] = pd.Series(tail).value_counts().to_dict()
        print("states per class (fake data):", counts)
        assert max(counts["attentive"], key=counts["attentive"].get) == "ATTENTIVE"
        assert max(counts["drowsy"], key=counts["drowsy"].get) in ("DROWSINESS_SUSPECTED", "HIGH_RISK")
        print("ok  engine predicts the right state on fake data")

        # rule: face lost -> UNKNOWN (never ATTENTIVE)
        eng = DriverStateEngine(model)
        rows = df[df.label == "attentive"].head(300).to_dict("records")
        for rec in rows:
            eng.update(rec)
        assert eng.message["state"] == "ATTENTIVE", eng.message["state"]
        t = rows[-1]["t_s"]
        for i in range(1, 60):                                    # 4 s without a face
            nf = {"t_s": t + i / 15, "face": 0, "ear": float("nan"), "ear_l": float("nan"),
                  "ear_r": float("nan"), "mar": float("nan"), "yaw": float("nan"),
                  "pitch": float("nan"), "roll": float("nan"), "face_w": float("nan")}
            msg = eng.update(nf)
        assert msg["state"] == "UNKNOWN" and "face_not_visible" in msg["evidence"], msg
        print("ok  face lost -> UNKNOWN (not ATTENTIVE)")

        # rule: warm-up is UNKNOWN
        eng = DriverStateEngine(model)
        for rec in rows[:30]:
            msg = eng.update(rec)
        assert msg["state"] == "UNKNOWN" and "warming_up" in msg["evidence"], msg
        print("ok  warm-up -> UNKNOWN")


def test_deep_learning():
    try:
        import onnxruntime  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("skip  deep-learning test (pip install -r requirements-dl.txt to enable it)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        raw, model = Path(tmp) / "raw", Path(tmp) / "dl" / "driver_state_dl.onnx"
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_data.py"), "--out", str(raw),
                        "--participants", "3"], check=True, capture_output=True)
        subprocess.run([sys.executable, str(ROOT / "src" / "train_dl.py"), "--data", str(raw),
                        "--out", str(model), "--epochs", "15"], check=True, capture_output=True)
        assert model.exists() and model.with_suffix(".json").exists()
        assert not Path(str(model) + ".data").exists(), "weights must be inside the single .onnx file"
        df = pd.read_csv(raw / "P01_synthetic.csv")
        for label, ok_states in (("attentive", ("ATTENTIVE",)), ("distracted", ("DISTRACTED",)),
                                 ("drowsy", ("DROWSINESS_SUSPECTED", "HIGH_RISK"))):
            eng = DriverStateEngine(model)
            assert eng.backend == "deep_learning_onnx"
            states = [eng.update(rec)["state"] for rec in df[df.label == label].to_dict("records")]
            tail = pd.Series(states[len(states) // 3:]).value_counts()
            assert tail.idxmax() in ok_states, (label, tail.to_dict())
        print("ok  deep-learning ONNX engine predicts the right state on fake data")
        rows = df[df.label == "attentive"].head(300).to_dict("records")
        eng = DriverStateEngine(model)
        for rec in rows:
            eng.update(rec)
        t = rows[-1]["t_s"]
        for i in range(1, 60):
            msg = eng.update({"t_s": t + i / 15, "face": 0, "ear": float("nan"), "ear_l": float("nan"),
                              "ear_r": float("nan"), "mar": float("nan"), "yaw": float("nan"),
                              "pitch": float("nan"), "roll": float("nan"), "face_w": float("nan")})
        assert msg["state"] == "UNKNOWN" and "face_not_visible" in msg["evidence"]
        print("ok  deep-learning engine: face lost -> UNKNOWN")

        # large model + 2-seed ensemble (the Colab configuration, just with very few epochs)
        big = Path(tmp) / "big" / "driver_state_dl_large.onnx"
        subprocess.run([sys.executable, str(ROOT / "src" / "train_dl.py"), "--data", str(raw), "--arch", "large",
                        "--epochs", "8", "--seeds", "2", "--out", str(big)], check=True, capture_output=True)
        assert big.exists() and not Path(str(big) + ".data").exists()
        eng = DriverStateEngine(big)
        assert eng.backend == "deep_learning_onnx"
        states = [eng.update(r)["state"] for r in df[df.label == "drowsy"].to_dict("records")]
        assert pd.Series(states[len(states) // 3:]).value_counts().idxmax() in ("DROWSINESS_SUSPECTED", "HIGH_RISK")
        print("ok  large model + ensemble trains, exports to ONE onnx file and runs in the engine")


if __name__ == "__main__":
    test_ear()
    test_mar()
    test_train_and_engine()
    test_deep_learning()
    print("ALL TESTS PASSED")
