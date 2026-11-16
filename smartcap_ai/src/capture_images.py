"""
Capture YOUR OWN labelled face images (for testing the image models) from the laptop webcam or the ESP32.

    python src/capture_images.py --participant P01 --classes alert drowsy
    python src/capture_images.py --participant P01 --classes alert drowsy --camera esp32

Keys: 1..9 = choose class (in the order of --classes) and START saving ~3 images/second | 0 = pause | q / ESC = quit
Images are face crops (MediaPipe face landmarker, same model file as the rest of the project) saved to my_images/<class>/ . The folder names MUST match the class
names of the dataset you trained on. These are photos of people: keep them on your machine, get consent, never commit
them to git (SmartCap spec section 14 - the product itself never stores video/images).
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import DEFAULT_MODEL, open_camera  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def make_detector(model_path=DEFAULT_MODEL, margin=1.25):
    """frame_bgr -> (x, y, w, h) square face box, or None. (OpenCV 5 no longer ships the Haar xml files.)"""
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    lm = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1))

    def detect(frame):
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        xs = [p.x * w for p in res.face_landmarks[0]]
        ys = [p.y * h for p in res.face_landmarks[0]]
        side = int(max(max(xs) - min(xs), max(ys) - min(ys)) * margin)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        x0, y0 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
        x1, y1 = min(w, x0 + side), min(h, y0 + side)
        return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 > 40 and y1 - y0 > 40 else None
    return detect


def main():
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument("--participant", required=True)
    p.add_argument("--classes", nargs="+", required=True, help="e.g. alert drowsy (same names as dataset folders)")
    p.add_argument("--camera", default="0")
    p.add_argument("--out", default=str(ROOT / "my_images"))
    p.add_argument("--rate", type=float, default=3.0, help="images per second while saving")
    p.add_argument("--size", type=int, default=128, help="saved crop size (pixels)")
    p.add_argument("--margin", type=float, default=1.0,
                   help="crop size relative to the face box (1.0 = tight, like the FL3D images; 1.25 = more background)")
    args = p.parse_args()

    detect = make_detector(margin=args.margin)
    cap = open_camera(args.camera)
    out = Path(args.out)
    for c in args.classes:
        (out / c).mkdir(parents=True, exist_ok=True)
    cur, last_save, saved = None, 0.0, {c: 0 for c in args.classes}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Keys:", ", ".join(f"{i+1}={c}" for i, c in enumerate(args.classes)), "| 0 pause | q quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame.")
                break
            box = detect(frame)
            if box is not None:
                x, y, w, h = box
                if cur and time.time() - last_save >= 1.0 / args.rate:
                    # crop from the CLEAN frame (the green box is drawn afterwards, only for display)
                    crop = cv2.resize(frame[y:y + h, x:x + w], (args.size, args.size))
                    n = saved[cur]
                    cv2.imencode(".jpg", crop)[1].tofile(str(out / cur / f"{args.participant}_{stamp}_{n:04d}.jpg"))
                    saved[cur] += 1
                    last_save = time.time()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"saving: {cur or 'PAUSED'}  face: {'OK' if box is not None else 'NO'}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "  ".join(f"{c}:{n}" for c, n in saved.items()), (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("SmartCap - capture images", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("0"):
                cur = None
            elif ord("1") <= k <= ord("9") and k - ord("1") < len(args.classes):
                cur = args.classes[k - ord("1")]
            elif k in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Saved:", saved, "->", out)


if __name__ == "__main__":
    main()
