"""
Live driver-state demo on the laptop webcam + laptop screen.

    python src/live.py
    python src/live.py --log reports/live_log.jsonl      # also write driver_state JSON (no video)
    python src/live.py --replay data/raw/P01_xxx.csv --no-show   # test without a camera

Press q or ESC to quit. This is a research prototype, not a medical or certified safety system.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import DriverStateEngine  # noqa: E402
from landmarks import DEFAULT_MODEL, FaceFeatureExtractor, open_camera  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATE_COLOR = {"ATTENTIVE": (0, 200, 0), "DISTRACTED": (0, 200, 255),
               "DROWSINESS_SUSPECTED": (0, 100, 255), "HIGH_RISK": (0, 0, 255),
               "UNKNOWN": (180, 180, 180)}


def draw(frame, msg, engine):
    import cv2
    h, w = frame.shape[:2]
    state = msg["state"]
    color = STATE_COLOR.get(state, (255, 255, 255))
    if state in ("HIGH_RISK", "DROWSINESS_SUSPECTED"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    cv2.putText(frame, f"State: {state}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 3)
    risk = "-" if msg["risk_score"] is None else f"{msg['risk_score']:.2f}"
    cv2.putText(frame, f"risk {msg['risk_level']} ({risk})   conf {msg['confidence']:.2f}   "
                       f"cam-q {msg['camera_quality']:.2f}   fps {engine.fps:.0f}",
                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if msg["evidence"]:
        cv2.putText(frame, "evidence: " + ", ".join(msg["evidence"]), (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if msg["probs"]:
        y = 112
        for name, p in msg["probs"].items():
            cv2.rectangle(frame, (12, y - 12), (12 + int(180 * p), y + 2), STATE_COLOR.get(
                {"attentive": "ATTENTIVE", "distracted": "DISTRACTED", "drowsy": "DROWSINESS_SUSPECTED"}[name]), -1)
            cv2.putText(frame, f"{name} {p:.2f}", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22
    cv2.putText(frame, "q / ESC = quit", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    p = argparse.ArgumentParser(description="SmartCap live driver-state demo")
    p.add_argument("--model", default=str(ROOT / "models" / "driver_state_model.joblib"))
    p.add_argument("--landmarker", default=str(DEFAULT_MODEL))
    p.add_argument("--camera", default="0",
                   help="0/1 = USB or laptop webcam | esp32 = cap camera (http://192.168.4.1/stream) | any URL or IP")
    p.add_argument("--log", default=None, help="append driver_state JSON lines to this file")
    p.add_argument("--replay", default=None, help="replay a recorded CSV instead of using the camera")
    p.add_argument("--no-show", action="store_true", help="no window (console output only)")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640, help="camera width (try 480 or 320 if it is slow)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--ema", type=float, default=0.4,
                   help="smoothing speed 0-1: higher = reacts faster but flickers more (try 0.7)")
    p.add_argument("--every", type=float, default=0.5,
                   help="seconds between model evaluations (0.25 = more responsive)")
    args = p.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}\nTrain first: python src/train.py")
    engine = DriverStateEngine(args.model, min_conf=args.min_conf, ema=args.ema, compute_every_s=args.every)
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    last_print = 0.0

    def handle(msg):
        nonlocal last_print
        now = time.time()
        if now - last_print >= 1.0:
            last_print = now
            print(f"{msg['state']:<22} risk={msg['risk_level']:<8} conf={msg['confidence']:.2f} "
                  f"evidence={msg['evidence']}")
            if log:
                log.write(json.dumps(msg) + "\n")
                log.flush()

    if args.replay:
        for rec in pd.read_csv(args.replay).to_dict("records"):
            handle(engine.update(rec))
        return

    import cv2
    extractor = FaceFeatureExtractor(args.landmarker)
    cap = open_camera(args.camera, args.width, args.height)
    print("Live demo started. The first ~10 s are warm-up (state UNKNOWN). Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the camera.")
                break
            msg = engine.update(extractor.process(frame))
            handle(msg)
            if not args.no_show:
                draw(frame, msg, engine)
                cv2.imshow("SmartCap - driver state (vision only)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
        if log:
            log.close()


if __name__ == "__main__":
    main()
