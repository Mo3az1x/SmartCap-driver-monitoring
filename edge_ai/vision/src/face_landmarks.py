import csv
from datetime import datetime
from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ear_mar import calculate_eye_metrics, calculate_mar


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
CSV_PATH = Path(__file__).resolve().parent / "face_metrics.csv"

# خريطة المفاتيح لتغيير الـ Label أثناء التشغيل
LABEL_MAP = {
    ord("1"): "OPEN_EYES",
    ord("2"): "NORMAL_BLINK",
    ord("3"): "EYES_CLOSED",
    ord("4"): "TALKING",
    ord("5"): "YAWN",
    ord("6"): "OTHER",
}


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def init_csv_file(file_path):
    file_exists = file_path.exists()
    # استخدام buffering=1 لكتابة السطور فوراً إلى القرص
    file = open(file_path, mode="a", newline="", encoding="utf-8", buffering=1)
    fieldnames = [
        "timestamp",
        "label",
        "face_detected",
        "left_ear",
        "right_ear",
        "ear",
        "mar",
        "fps",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    return file, writer


def main():
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nFace Landmarker model was not found.\n"
            f"Expected path: {MODEL_PATH}\n"
            "Download face_landmarker.task and place it in edge_ai/vision/models/"
        )

    print("Opening webcam...")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Close other apps using the camera "
            "or change camera index from 0 to 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    csv_file, csv_writer = init_csv_file(CSV_PATH)
    current_label = "OPEN_EYES"

    previous_time = time.time()
    start_time = time.perf_counter()

    print("Starting Face Landmarker & CSV Logging...")
    print("Press 1: OPEN_EYES | 2: NORMAL_BLINK | 3: EYES_CLOSED | 4: TALKING | 5: YAWN | 6: OTHER")
    print("Press Q or ESC to exit.")

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Failed to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                current_time = time.time()
                fps = 1 / max(current_time - previous_time, 1e-6)
                previous_time = current_time

                iso_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    draw_landmarks(frame, landmarks)

                    height, width = frame.shape[:2]
                    eye_metrics = calculate_eye_metrics(landmarks, width, height)
                    mar = calculate_mar(landmarks, width, height)
                    ear = eye_metrics["ear"]

                    status = "Face: DETECTED"
                    status_color = (0, 255, 0)

                    csv_writer.writerow({
                        "timestamp": iso_timestamp,
                        "label": current_label,
                        "face_detected": True,
                        "left_ear": round(eye_metrics["left_ear"], 4),
                        "right_ear": round(eye_metrics["right_ear"], 4),
                        "ear": round(ear, 4),
                        "mar": round(mar, 4),
                        "fps": round(fps, 1),
                    })

                    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.3f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                else:
                    status = "Face: NOT DETECTED"
                    status_color = (0, 0, 255)

                    csv_writer.writerow({
                        "timestamp": iso_timestamp,
                        "label": current_label,
                        "face_detected": False,
                        "left_ear": None,
                        "right_ear": None,
                        "ear": None,
                        "mar": None,
                        "fps": round(fps, 1),
                    })

                cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Label: {current_label} (Keys 1-6)", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("SmartCap - Face Landmarks & Logger", frame)

                key = cv2.waitKey(1) & 0xFF

                if key in LABEL_MAP:
                    current_label = LABEL_MAP[key]
                    print(f"Switched Label to: {current_label}")

                if key in (ord("q"), 27):
                    break

    finally:
        csv_file.close()
        camera.release()
        cv2.destroyAllWindows()
        print(f"\nSaved session metrics to: {CSV_PATH}")


if __name__ == "__main__":
    main()