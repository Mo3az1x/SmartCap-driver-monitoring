<<<<<<< Updated upstream
=======
import csv
import time
from datetime import datetime
>>>>>>> Stashed changes
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ear_mar import calculate_eye_metrics, calculate_mar


# The model is expected in edge_ai/vision/models/
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
<<<<<<< Updated upstream
=======
CSV_PATH = Path(__file__).resolve().parent / "face_metrics_dashboard.csv"

LABELS = {
    ord("1"): "OPEN_EYES",
    ord("2"): "NORMAL_BLINK",
    ord("3"): "EYES_CLOSED",
    ord("4"): "TALKING",
    ord("5"): "YAWN",
    ord("6"): "OTHER",
}
>>>>>>> Stashed changes


def open_webcam():
    """Try camera indices 0, 1, and 2 using different Windows backends."""
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Automatic", cv2.CAP_ANY),
    ]

    for index in (0, 1, 2):
        for backend_name, backend in backends:
            camera = cv2.VideoCapture(index, backend)

            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                success, _ = camera.read()
                if success:
                    print(f"Webcam opened: index {index}, backend {backend_name}")
                    return camera

            camera.release()

    raise RuntimeError(
        "Could not open the webcam. Close the Camera app or Zoom, "
        "and check Windows camera permissions."
    )


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

<<<<<<< Updated upstream
    for landmark in landmarks:
=======
    # Draw every third landmark to keep the image clear.
    for index, landmark in enumerate(landmarks):
        if index % 3 != 0:
            continue

>>>>>>> Stashed changes
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, (80, 230, 170), -1)


<<<<<<< Updated upstream
=======
def draw_dashboard(frame, face_found, ear, mar, fps, label):
    height, width = frame.shape[:2]

    # Add semi-transparent header and footer bars.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 70), (15, 26, 42), -1)
    cv2.rectangle(overlay, (0, height - 68), (width, height), (15, 26, 42), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Header
    cv2.putText(
        frame,
        "SMARTCAP  /  VISION LAB",
        (20, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 245, 250),
        2,
    )
    cv2.putText(
        frame,
        "FEATURE DEMO  -  NO CLASSIFIER",
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (160, 180, 195),
        1,
    )

    # Face detection status and frame rate
    if face_found:
        face_text = "FACE DETECTED"
        face_color = (75, 220, 160)
    else:
        face_text = "NO FACE - CHECK POSITION"
        face_color = (80, 130, 255)

    cv2.putText(
        frame,
        face_text,
        (width - 300, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        face_color,
        2,
    )
    cv2.putText(
        frame,
        f"FPS  {fps:.1f}",
        (width - 300, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 230, 240),
        1,
    )

    # Feature values in the footer
    ear_text = "--" if ear is None else f"{ear:.3f}"
    mar_text = "--" if mar is None else f"{mar:.3f}"

    cv2.putText(
        frame,
        f"EAR  {ear_text}",
        (20, height - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 230, 240),
        2,
    )
    cv2.putText(
        frame,
        f"MAR  {mar_text}",
        (190, height - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 230, 240),
        2,
    )
    cv2.putText(
        frame,
        f"MANUAL LABEL  {label}",
        (width - 270, height - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (190, 220, 205),
        1,
    )
    cv2.putText(
        frame,
        "Keys 1-6: label   |   Q / ESC: quit",
        (20, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (155, 175, 190),
        1,
    )


>>>>>>> Stashed changes
def main():
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
<<<<<<< Updated upstream
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

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )
=======
            "The face landmarker model file was not found.\n"
            "Place face_landmarker.task here:\n"
            f"{MODEL_PATH}"
        )

    camera = open_webcam()
    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8", buffering=1)

    columns = [
        "timestamp",
        "label",
        "face_detected",
        "left_ear",
        "right_ear",
        "ear",
        "mar",
        "fps",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=columns)

    if CSV_PATH.stat().st_size == 0:
        writer.writeheader()
>>>>>>> Stashed changes

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

<<<<<<< Updated upstream
    previous_time = time.time()
=======
    current_label = "OPEN_EYES"
    previous_time = time.perf_counter()
>>>>>>> Stashed changes
    start_time = time.perf_counter()
    last_timestamp_ms = -1

<<<<<<< Updated upstream
    print("Starting Face Landmarker...")
    print("Press Q or ESC to exit.")
=======
    print("Starting face landmark test. Press Q or ESC to exit.")
    print("Keys 1-6 change the manual label saved in the CSV file.")
>>>>>>> Stashed changes

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

<<<<<<< Updated upstream
            if not success:
                print("Failed to read webcam frame.")
                break
=======
                if not success:
                    print("Could not read a frame from the webcam.")
                    break
>>>>>>> Stashed changes

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

<<<<<<< Updated upstream
            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )
=======
                # MediaPipe VIDEO mode requires increasing timestamps.
                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
>>>>>>> Stashed changes

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            current_time = time.time()
            fps = 1 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                draw_landmarks(frame, landmarks)

                height, width = frame.shape[:2]

                eye_metrics = calculate_eye_metrics(
                    landmarks,
                    width,
                    height,
                )
<<<<<<< Updated upstream

                mar = calculate_mar(
                    landmarks,
                    width,
                    height,
                )

                ear = eye_metrics["ear"]

                status = "Face: DETECTED"
                status_color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            else:
                status = "Face: NOT DETECTED"
                status_color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Press Q or ESC to exit",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("SmartCap - Face Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    camera.release()
    cv2.destroyAllWindows()
=======
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                now = time.perf_counter()
                fps = 1 / max(now - previous_time, 1e-6)
                previous_time = now

                ear = None
                mar = None
                face_found = bool(result.face_landmarks)

                if face_found:
                    landmarks = result.face_landmarks[0]
                    draw_landmarks(frame, landmarks)

                    frame_height, frame_width = frame.shape[:2]
                    eye_metrics = calculate_eye_metrics(
                        landmarks, frame_width, frame_height
                    )
                    ear = eye_metrics["ear"]
                    mar = calculate_mar(
                        landmarks, frame_width, frame_height
                    )

                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "label": current_label,
                    "face_detected": face_found,
                    "left_ear": (
                        round(eye_metrics["left_ear"], 4)
                        if face_found else None
                    ),
                    "right_ear": (
                        round(eye_metrics["right_ear"], 4)
                        if face_found else None
                    ),
                    "ear": round(ear, 4) if ear is not None else None,
                    "mar": round(mar, 4) if mar is not None else None,
                    "fps": round(fps, 1),
                })

                draw_dashboard(frame, face_found, ear, mar, fps, current_label)
                cv2.imshow("SmartCap - Vision Dashboard", frame)

                key = cv2.waitKey(1) & 0xFF

                if key in LABELS:
                    current_label = LABELS[key]
                    print(f"Manual label changed to: {current_label}")

                if key in (ord("q"), 27):
                    break

    finally:
        csv_file.close()
        camera.release()
        cv2.destroyAllWindows()
        print(f"Feature measurements saved to: {CSV_PATH}")
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()