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
ACCURACY OF THE LIVE DEMO (you tell it the true state while you act it out):
    python src/live_img.py --model models/img_cnn.pt --camera 1 --eval
    keys: 1 = alert, 2 = microsleep (eyes closed), 3 = yawning, 0 = pause (not counted), q = finish + report
    -> reports/live_eval/live_eval.md (+ csv, confusion png): per-frame accuracy, state accuracy, UNKNOWN rate, latency
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


class LiveEval:
    """Counts how often the live output matches the state YOU said you were acting out."""

    def __init__(self, classes, settle_s=1.0):
        self.classes, self.settle_s = classes, settle_s
        n = len(classes)
        self.cm = np.zeros((n, n), dtype=int)            # rows: true, columns: raw CNN prediction (per frame)
        self.state_ok = np.zeros(n, dtype=int)
        self.state_n = np.zeros(n, dtype=int)
        self.unknown = np.zeros(n, dtype=int)
        self.no_face = 0
        self.ms = []
        self.rows = []
        self.label, self.label_t = None, 0.0
        low = [c.lower() for c in classes]
        self.expected = [{"ATTENTIVE"} if "alert" in c else {"DROWSINESS_SUSPECTED", "HIGH_RISK"} for c in low]

    def set_label(self, idx, now):
        if idx != self.label:
            self.label, self.label_t = idx, now

    def add(self, now, probs, msg, ms):
        if self.label is None or now - self.label_t < self.settle_s:      # paused / just changed: not counted
            return
        if probs is None:
            self.no_face += 1
            return
        k = self.label
        pred = int(probs.argmax())
        self.cm[k, pred] += 1
        self.state_n[k] += 1
        self.state_ok[k] += msg["state"] in self.expected[k]
        self.unknown[k] += msg["state"] == "UNKNOWN"
        self.ms.append(ms)
        self.rows.append((round(now, 2), self.classes[k], self.classes[pred], msg["state"], msg["confidence"]))

    def report(self, out_dir):
        import csv
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = int(self.cm.sum())
        if n == 0:
            print("Eval: no labelled frames were counted (press 1/2/3 and keep your face in view).")
            return
        frame_acc = np.trace(self.cm) / n
        state_acc = self.state_ok.sum() / max(self.state_n.sum(), 1)
        lines = ["# Live evaluation (CNN)", "",
                 f"- Labelled frames counted: {n} (frames in the first {self.settle_s:.1f} s after each key press, "
                 f"paused frames and no-face frames [{self.no_face}] are not counted)",
                 f"- **Per-frame accuracy** (raw CNN, no smoothing): **{frame_acc*100:.1f}%**",
                 f"- **State accuracy** (after smoothing + rules; UNKNOWN counts as wrong): **{state_acc*100:.1f}%**",
                 f"- UNKNOWN rate: {self.unknown.sum() / max(self.state_n.sum(), 1) * 100:.1f}%",
                 f"- Time per frame (face crop -> CNN -> state, this machine): mean {np.mean(self.ms):.1f} ms, "
                 f"p95 {np.percentile(self.ms, 95):.1f} ms", "",
                 "| true state | frames | per-frame acc | state acc | UNKNOWN |", "|---|---|---|---|---|"]
        for i, c in enumerate(self.classes):
            t = int(self.cm[i].sum())
            if t:
                lines.append(f"| {c} | {t} | {self.cm[i, i] / t * 100:.1f}% | "
                             f"{self.state_ok[i] / max(self.state_n[i], 1) * 100:.1f}% | "
                             f"{self.unknown[i] / max(self.state_n[i], 1) * 100:.1f}% |")
        lines += ["", "Confusion matrix (rows: what you acted, columns: what the CNN said, per frame):", "",
                  "| true \\ predicted | " + " | ".join(self.classes) + " |", "|---" * (len(self.classes) + 1) + "|"]
        for i, c in enumerate(self.classes):
            lines.append(f"| {c} | " + " | ".join(str(v) for v in self.cm[i]) + " |")
        lines += ["", "Honest notes: you are the only subject, the labels are what you *meant* to act (a half-closed eye "
                      "while labelled 'alert' counts as an error), and neighbouring frames are almost identical, so treat "
                      "this as a demo check, not a final benchmark."]
        (out / "live_eval.md").write_text("\n".join(lines), encoding="utf-8")
        with open(out / "live_eval_frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "true", "raw_pred", "state", "confidence"])
            w.writerows(self.rows)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, a = plt.subplots(figsize=(1.2 * len(self.classes) + 3, 1.2 * len(self.classes) + 2.5))
            a.imshow(self.cm, cmap="Blues")
            a.set(xticks=range(len(self.classes)), yticks=range(len(self.classes)), xticklabels=self.classes,
                  yticklabels=self.classes, xlabel="CNN said", ylabel="you acted",
                  title=f"Live per-frame accuracy {frame_acc*100:.1f}%")
            for i in range(len(self.classes)):
                for j in range(len(self.classes)):
                    a.text(j, i, int(self.cm[i, j]), ha="center", va="center",
                           color="white" if self.cm[i, j] > self.cm.max() / 2 else "black")
            fig.tight_layout()
            fig.savefig(out / "live_eval_confusion.png", dpi=130)
            plt.close(fig)
        except Exception:
            pass
        print("\n".join(lines))
        print(f"\nSaved in {out}")


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
    p.add_argument("--eval", action="store_true", help="live accuracy: press 1/2/3 to tell it the true state")
    p.add_argument("--settle", type=float, default=1.0,
                   help="--eval: ignore this many seconds after each key press (you are still changing pose)")
    p.add_argument("--eval-out", default=str(Path(__file__).resolve().parents[1] / "reports" / "live_eval"))
    args = p.parse_args()

    pred = Predictor(args.model)
    sm = StateMachine(pred.classes, args.ema, args.min_conf, args.microsleep_s)
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    last_print = [0.0]
    ev = LiveEval(pred.classes, args.settle) if args.eval else None

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
            t0 = time.perf_counter()
            probs = pred(frame)
            msg = sm.update(probs, t)
            ms = (time.perf_counter() - t0) * 1000
            if ev is not None and f.parent.name in pred.classes:        # folder name = true label
                ev.set_label(pred.classes.index(f.parent.name), t)
                ev.add(t, probs, msg, ms)
            last_print[0] = -1e9
            handle(msg, t)
        if ev is not None:
            ev.report(args.eval_out)
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
            t0 = time.perf_counter()
            box = detect(frame)
            probs = None
            if box is not None:
                x, y, w, h = box
                probs = pred(frame[y:y + h, x:x + w])
            now = time.time()
            msg = sm.update(probs, now)
            ms = (time.perf_counter() - t0) * 1000
            if ev is not None:
                ev.add(now, probs, msg, ms)
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            handle(msg, now)
            if not args.no_show:
                draw(frame, msg, box, fps)
                if ev is not None:
                    names = " ".join(f"{i + 1}={c}" for i, c in enumerate(pred.classes))
                    cur = "PAUSED" if ev.label is None else pred.classes[ev.label]
                    cv2.putText(frame, f"EVAL acting: {cur}   counted: {int(ev.cm.sum())}", (12, frame.shape[0] - 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(frame, f"{names}  0=pause  q=report", (12, frame.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("SmartCap - driver state (image CNN)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if ev is not None:
                    if key == ord("0"):
                        ev.set_label(None, now)
                    elif ord("1") <= key < ord("1") + len(pred.classes):
                        ev.set_label(key - ord("1"), now)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log.close()
        if ev is not None:
            ev.report(args.eval_out)


if __name__ == "__main__":
    main()
