"""
Live driver state from the IMAGE model (CNN): camera -> face crop -> CNN -> smoothing -> SmartCap driver_state.

    python src/live_img.py --model models/img_cnn.pt --camera 1          # laptop front camera
    python src/live_img.py --model models/img_cnn.pt --camera esp32      # cap camera
    python src/live_img.py --model models/img_cnn.onnx --camera 1        # ONNX (what the Raspberry Pi would run)
    python src/live_img.py --model models/img_cnn.pt --images my_images/alert --no-show   # test on a folder, no camera

Rules (starting values, tune with the team):
  no face for > 1.5 s            -> UNKNOWN (never ATTENTIVE when the camera sees nothing)
  smoothed confidence < 0.55     -> UNKNOWN
  alert                          -> ATTENTIVE
  yawning                        -> DROWSINESS_SUSPECTED
  microsleep (eyes closed)       -> DROWSINESS_SUSPECTED, and HIGH_RISK after it lasts >= 2 s
Press q / ESC to quit. Research prototype, not a medical or certified safety system.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgmodels as im  # noqa: E402
from engine import risk_level_for  # noqa: E402
from landmarks import DEFAULT_MODEL, open_camera  # noqa: E402

COLOR = {"ATTENTIVE": (0, 200, 0), "DROWSINESS_SUSPECTED": (0, 100, 255), "HIGH_RISK": (0, 0, 255),
         "UNKNOWN": (180, 180, 180)}
PRE = 128      # same two-step resize the test images went through (saved 128x128 crops -> model size)


class Predictor:
    """Loads img_*.pt (torch) or img_*.onnx (+ .json next to it) and returns class probabilities."""

    def __init__(self, path):
        path = Path(path)
        if path.suffix == ".onnx":
            import onnxruntime as ort
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.torch_model = None
            self.classes, self.img, self.gray = meta["classes"], meta["img"], meta["gray"]
        else:
            import torch
            self.device = im.get_device()
            self.torch_model, ck = im.load_checkpoint(path, self.device)
            self.classes, self.img, self.gray = ck["classes"], ck["img"], ck["gray"]

    def __call__(self, crop_bgr):
        import cv2
        a = cv2.resize(crop_bgr, (PRE, PRE), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (self.img, self.img), interpolation=cv2.INTER_AREA)
        if self.gray:
            x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)[None]
        else:
            x = np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1))
        x = (x[None].astype(np.float32)) / 255.0
        if self.torch_model is None:
            z = self.sess.run(None, {"image": x})[0][0]
        else:
            import torch
            with torch.no_grad():
                z = self.torch_model(torch.from_numpy(x).to(self.device)).cpu().numpy()[0]
        e = np.exp(z - z.max())
        return e / e.sum()


class StateMachine:
    def __init__(self, classes, ema=0.5, min_conf=0.55, microsleep_s=2.0, no_face_s=1.5):
        self.classes, self.ema_a, self.min_conf = classes, ema, min_conf
        self.micro_s, self.no_face_s = microsleep_s, no_face_s
        low = [c.lower() for c in classes]
        self.i_alert = next((i for i, c in enumerate(low) if "alert" in c), 0)
        self.i_micro = next((i for i, c in enumerate(low) if "micro" in c or "sleep" in c), None)
        self.i_yawn = next((i for i, c in enumerate(low) if "yawn" in c), None)
        self.p = None
        self.last_face = None
        self.micro_since = None

    def _msg(self, state, risk, conf, evidence, probs):
        return {"state": state, "risk_level": risk_level_for(risk) if risk is not None else "UNKNOWN",
                "risk_score": None if risk is None else round(float(risk), 3), "confidence": round(float(conf), 3),
                "imu_quality": None, "evidence": evidence,
                "system_status": "OK" if state != "UNKNOWN" else "DEGRADED",
                "probs": probs, "source": "vision_only_cnn"}

    def update(self, probs, now):
        """probs = class probabilities for this frame, or None when no face was found."""
        if probs is None:
            if self.last_face is None or now - self.last_face > self.no_face_s:
                self.p, self.micro_since = None, None
                return self._msg("UNKNOWN", None, 0.0, ["no_face"], None)
            if self.p is None:
                return self._msg("UNKNOWN", None, 0.0, ["no_face"], None)
        else:
            self.last_face = now
            self.p = probs if self.p is None else self.ema_a * probs + (1 - self.ema_a) * self.p
        p = self.p
        pd = {c: round(float(v), 3) for c, v in zip(self.classes, p)}
        k = int(p.argmax())
        conf = float(p[k])
        p_micro = float(p[self.i_micro]) if self.i_micro is not None else 0.0
        p_yawn = float(p[self.i_yawn]) if self.i_yawn is not None else 0.0
        risk = min(1.0, p_micro + 0.5 * p_yawn)
        if p_micro > 0.5:
            self.micro_since = self.micro_since or now
        else:
            self.micro_since = None
        if conf < self.min_conf:
            return self._msg("UNKNOWN", None, conf, ["low_confidence"], pd)
        if k == self.i_micro:
            dur = now - (self.micro_since or now)
            ev = [f"eyes_closed_{dur:.1f}s"]
            return self._msg("HIGH_RISK" if dur >= self.micro_s else "DROWSINESS_SUSPECTED", risk, conf, ev, pd)
        if k == self.i_yawn:
            return self._msg("DROWSINESS_SUSPECTED", risk, conf, ["yawning"], pd)
        return self._msg("ATTENTIVE", risk, conf, [], pd)


def draw(frame, msg, box, fps):
    import cv2
    h, w = frame.shape[:2]
    color = COLOR.get(msg["state"], (255, 255, 255))
    if box is not None:
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
    if msg["state"] in ("HIGH_RISK", "DROWSINESS_SUSPECTED"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    cv2.putText(frame, f"State: {msg['state']}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 3)
    risk = "-" if msg["risk_score"] is None else f"{msg['risk_score']:.2f}"
    cv2.putText(frame, f"risk {msg['risk_level']} ({risk})  conf {msg['confidence']:.2f}  fps {fps:.0f}", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if msg["evidence"]:
        cv2.putText(frame, "evidence: " + ", ".join(msg["evidence"]), (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
    y = 112
    for name, v in (msg["probs"] or {}).items():
        cv2.rectangle(frame, (12, y - 12), (12 + int(160 * v), y + 2), color, -1)
        cv2.putText(frame, f"{name} {v:.2f}", (180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22
    cv2.putText(frame, "q / ESC = quit", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    p = argparse.ArgumentParser(description="SmartCap live driver state (image CNN)")
    p.add_argument("--model", required=True, help="models/img_cnn.pt or .onnx")
    p.add_argument("--camera", default="0", help="0/1 webcam | esp32 | URL | IP")
    p.add_argument("--images", default=None, help="folder of face-crop images instead of a camera (for testing)")
    p.add_argument("--landmarker", default=str(DEFAULT_MODEL))
    p.add_argument("--margin", type=float, default=1.0, help="face crop size relative to the face box")
    p.add_argument("--ema", type=float, default=0.5, help="smoothing 0-1 (higher = faster, flickers more)")
    p.add_argument("--min-conf", type=float, default=0.55)
    p.add_argument("--microsleep-s", type=float, default=2.0, help="eyes closed this long -> HIGH_RISK")
    p.add_argument("--log", default=None, help="append driver_state JSON lines (no images)")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    pred = Predictor(args.model)
    sm = StateMachine(pred.classes, args.ema, args.min_conf, args.microsleep_s)
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    last_print = [0.0]

    def handle(msg, t):
        if t - last_print[0] >= 1.0:
            last_print[0] = t
            print(f"{msg['state']:<22} risk={msg['risk_level']:<8} conf={msg['confidence']:.2f} {msg['evidence']}")
            if log:
                log.write(json.dumps(msg) + "\n")
                log.flush()

    if args.images:                                   # replay a folder (already face crops), no detector
        import cv2
        files = sorted(f for f in Path(args.images).rglob("*") if f.suffix.lower() in im.EXT)
        if not files:
            sys.exit(f"No images in {args.images}")
        t = 0.0
        for f in files:
            frame = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            t += 0.33
            msg = sm.update(pred(frame), t)
            last_print[0] = -1e9
            handle(msg, t)
        return

    import cv2
    from capture_images import make_detector
    detect = make_detector(args.landmarker, margin=args.margin)
    cap = open_camera(args.camera)
    print("Live (CNN) started. Press q to quit.")
    t_prev, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame.")
                break
            box = detect(frame)
            probs = None
            if box is not None:
                x, y, w, h = box
                probs = pred(frame[y:y + h, x:x + w])
            now = time.time()
            msg = sm.update(probs, now)
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            handle(msg, now)
            if not args.no_show:
                draw(frame, msg, box, fps)
                cv2.imshow("SmartCap - driver state (image CNN)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log.close()


if __name__ == "__main__":
    main()
