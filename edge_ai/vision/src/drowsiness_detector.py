import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ear_mar import calculate_eye_metrics, calculate_mar
from eye_closure_timer import EyeClosureTimer


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.50

PROLONGED_CLOSURE_SECONDS = 1.5
YAWN_DURATION_SECONDS = 1.2


def main():
    print("[1] SmartCap temporal vision baseline started.")
    print(f"[2] Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nFace Landmarker model was not found.\n"
            f"Expected path: {MODEL_PATH}\n"
            "Place face_landmarker.task in edge_ai/vision/models/"
        )

    print("[3] Opening webcam...")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Close apps using the camera "
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

    eye_timer = EyeClosureTimer(
        ear_threshold=EAR_THRESHOLD,
        perclos_window_seconds=30.0,
    )

    yawn_is_active = False
    yawn_start_time_s = None
    yawn_total_count = 0
    current_yawn_duration_s = 0.0

    previous_time = time.perf_counter()
    start_time = time.perf_counter()

    print("[4] Starting live detection. Press Q or ESC to exit.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("[WARNING] Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            current_timestamp_s = time.perf_counter() - start_time
            timestamp_ms = int(current_timestamp_s * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            now = time.perf_counter()
            fps = 1 / max(now - previous_time, 1e-6)
            previous_time = now

            warning_text = None
            warning_color = (0, 0, 255)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                height, width = frame.shape[:2]

                eye_metrics = calculate_eye_metrics(
                    landmarks,
                    width,
                    height,
                )

                ear = eye_metrics["ear"]

                mar = calculate_mar(
                    landmarks,
                    width,
                    height,
                )

                eye_state = eye_timer.update(
                    ear=ear,
                    timestamp_s=current_timestamp_s,
                )

                closure_duration_s = eye_state["closure_duration_s"]
                perclos = eye_state["perclos"]

                if mar > MAR_THRESHOLD:
                    if not yawn_is_active:
                        yawn_is_active = True
                        yawn_start_time_s = current_timestamp_s

                    current_yawn_duration_s = (
                        current_timestamp_s - yawn_start_time_s
                    )

                else:
                    if (
                        yawn_is_active
                        and current_yawn_duration_s >= YAWN_DURATION_SECONDS
                    ):
                        yawn_total_count += 1

                    yawn_is_active = False
                    yawn_start_time_s = None
                    current_yawn_duration_s = 0.0

                prolonged_eye_closure = (
                    closure_duration_s >= PROLONGED_CLOSURE_SECONDS
                )

                prolonged_yawn = (
                    yawn_is_active
                    and current_yawn_duration_s >= YAWN_DURATION_SECONDS
                )

                if prolonged_eye_closure:
                    warning_text = "WARNING: PROLONGED EYE CLOSURE"
                    warning_color = (0, 0, 255)

                    cv2.rectangle(
                        frame,
                        (0, 0),
                        (width - 1, height - 1),
                        warning_color,
                        8,
                    )

                elif prolonged_yawn:
                    warning_text = "WARNING: PROLONGED YAWN"
                    warning_color = (0, 255, 255)

                cv2.putText(
                    frame,
                    "Face: DETECTED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Eye: {eye_state['eye_state']}",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Closure: {closure_duration_s:.2f}s",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"PERCLOS-30s: {perclos:.2f}",
                    (20, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Current Yawn: {current_yawn_duration_s:.2f}s",
                    (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Yawns: {yawn_total_count}",
                    (20, 315),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                if warning_text:
                    cv2.putText(
                        frame,
                        warning_text,
                        (40, 380),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        warning_color,
                        3,
                    )

            else:
                cv2.putText(
                    frame,
                    "Face: NOT DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "State: UNKNOWN",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Press Q or ESC to exit",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow(
                "SmartCap - Temporal Vision Baseline",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("[INFO] Exiting...")
                break

    camera.release()
    cv2.destroyAllWindows()
    print("[5] SmartCap temporal vision baseline stopped.")


if __name__ == "__main__":
    main()