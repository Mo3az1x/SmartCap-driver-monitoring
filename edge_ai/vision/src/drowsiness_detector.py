import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ear_mar import calculate_eye_metrics, calculate_mar

print("[1] Script started...")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"

# --- العتبات المسخرجة من التحليل ---
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.50

DROWSY_FRAMES_THRESHOLD = 30  # ~1.5 ثانية
YAWN_FRAMES_THRESHOLD = 25    # ~1.2 ثانية


def main():
    print(f"[2] Checking model path: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    print("[3] Opening webcam...")
    # استخدام 0 مباشر بدون CAP_DSHOW لتجنب تعليق بعض الكاميرات في ويندوز
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[4] Initializing MediaPipe FaceLandmarker...")
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ear_counter = 0
    yawn_counter = 0
    yawn_total_count = 0

    start_time = time.perf_counter()
    previous_time = time.time()

    print("[5] Starting loop... Press Q on camera window to exit.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            if not success:
                print("[WARNING] Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_time = time.time()
            fps = 1 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            drowsy_alert = False
            yawn_alert = False

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                height, width = frame.shape[:2]

                eye_metrics = calculate_eye_metrics(landmarks, width, height)
                mar = calculate_mar(landmarks, width, height)
                ear = eye_metrics["ear"]

                # 1. كشف النعاس
                if ear < EAR_THRESHOLD:
                    ear_counter += 1
                    if ear_counter >= DROWSY_FRAMES_THRESHOLD:
                        drowsy_alert = True
                else:
                    ear_counter = 0

                # 2. كشف التثاؤب
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= YAWN_FRAMES_THRESHOLD:
                        yawn_alert = True
                else:
                    if yawn_counter >= YAWN_FRAMES_THRESHOLD:
                        yawn_total_count += 1
                    yawn_counter = 0

                # عرض المؤشرات
                cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Yawns: {yawn_total_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if drowsy_alert:
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 8)
                    cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (100, 240), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 3)

                elif yawn_alert:
                    cv2.putText(frame, "WARNING: YAWNING DETECTED", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

            else:
                cv2.putText(frame, "FACE NOT DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                ear_counter = 0
                yawn_counter = 0

            cv2.imshow("SmartCap - Live Drowsiness Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Exiting...")
                break

    camera.release()
    cv2.destroyAllWindows()
    print("[6] Done.")


if __name__ == "__main__":
    main()