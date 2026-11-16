"""
Record labelled training data from the laptop webcam.

ONLY NUMBERS are saved (EAR, MAR, head pose...). No video, no images (SmartCap spec section 14).

    python src/collect.py --participant P01

Keys:  1 = attentive   2 = distracted   3 = drowsy (SIMULATED)   0 = pause   q / ESC = quit
Rules: written consent from every participant, simulate drowsiness only (never skip sleep),
       lab / parked vehicle only, anonymous codes (P01, P02 ...).
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import DEFAULT_MODEL, ROW_FIELDS, FaceFeatureExtractor, open_camera  # noqa: E402

KEYS = {ord("0"): "none", ord("1"): "attentive", ord("2"): "distracted", ord("3"): "drowsy"}
HELP = {
    "none": "PAUSED - get ready, then press 1 / 2 / 3",
    "attentive": "ATTENTIVE: look at the road, blink normally",
    "distracted": "DISTRACTED: look left / right / down (away from the road) every few seconds",
    "drowsy": "DROWSY (simulated): slow blinks, close eyes 1-3 s, yawn, let your head nod",
}
COLORS = {"none": (200, 200, 200), "attentive": (0, 200, 0), "distracted": (0, 200, 255),
          "drowsy": (0, 0, 255)}


def main():
    import cv2
    p = argparse.ArgumentParser(description="SmartCap data collection (numbers only, no video)")
    p.add_argument("--participant", required=True, help="anonymous code, e.g. P01")
    p.add_argument("--camera", default="0",
                   help="0/1 = USB or laptop webcam | esp32 = cap camera (http://192.168.4.1/stream) | any URL or IP")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.participant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print("Reminder: you need written consent from the participant. Only numbers are saved.")
    print(f"Saving to: {path}")
    print("Keys: 1 attentive | 2 distracted | 3 drowsy (simulated) | 0 pause | q quit")

    extractor = FaceFeatureExtractor(args.model)
    cap = open_camera(args.camera)
    label = "none"
    seconds = {"attentive": 0.0, "distracted": 0.0, "drowsy": 0.0}
    last_t = None

    with open(path, "w", newline="", encoding="utf-8", buffering=1) as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS + ["label", "participant"])
        writer.writeheader()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read a frame from the camera.")
                    break
                row = extractor.process(frame)
                dt = 0.0 if last_t is None else row["t_s"] - last_t
                last_t = row["t_s"]
                if label in seconds and row["face"]:
                    seconds[label] += min(dt, 0.5)
                row["label"], row["participant"] = label, args.participant
                writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})

                color = COLORS[label]
                cv2.putText(frame, f"Label: {label.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, HELP[label][:70], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                face_txt = "face OK" if row["face"] else "NO FACE"
                cv2.putText(frame, face_txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if row["face"] else (0, 0, 255), 2)
                y = 125
                for name in seconds:
                    cv2.putText(frame, f"{name}: {seconds[name]:5.0f} s (aim: 180+)", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[name], 1)
                    y += 24
                cv2.putText(frame, "1 attentive  2 distracted  3 drowsy  0 pause  q quit",
                            (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("SmartCap - collect data", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in KEYS:
                    label = KEYS[key]
                elif key in (ord("q"), 27):
                    break
        finally:
            cap.release()
            extractor.close()
            cv2.destroyAllWindows()
    print(f"Done. Recorded seconds per label: { {k: round(v) for k, v in seconds.items()} }")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
