from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ear_mar import calculate_eye_metrics, calculate_mar


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


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

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

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

    previous_time = time.time()
    start_time = time.perf_counter()

    print("Starting Face Landmarker...")
    print("Press Q or ESC to exit.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

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


if __name__ == "__main__":
    main()