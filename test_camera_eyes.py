
import cv2
import time
import numpy as np
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. Project paths and camera configuration
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "edge_ai"
    / "vision"
    / "models"
    / "face_landmarker.task"
)

CAMERA_URL = "http://192.168.4.1/stream"

# Camera rotation:
# 180 = upside down camera
# 90 clockwise = ROTATE_90_CLOCKWISE
# 90 counterclockwise = ROTATE_90_COUNTERCLOCKWISE

CAMERA_ROTATION = cv2.ROTATE_180


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face Landmarker model not found: {MODEL_PATH}"
    )


# ==========================================
# 2. Eye landmark indexes
# ==========================================

LEFT_EYE = [
    33, 133, 160, 159, 158, 157,
    173, 144, 145, 153, 154, 155
]

RIGHT_EYE = [
    362, 263, 387, 386, 385, 384,
    398, 373, 374, 380, 381, 382
]


# ==========================================
# 3. Create MediaPipe Face Landmarker
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# ==========================================
# 4. Draw eye bounding box
# ==========================================

def draw_eye_box(frame, landmarks, indexes, label):

    height, width = frame.shape[:2]

    points = []

    for index in indexes:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)

        points.append((x, y))

    points = np.array(points)

    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    padding_x = max(5, int((x_max - x_min) * 0.35))
    padding_y = max(5, int((y_max - y_min) * 0.70))

    x1 = max(0, x_min - padding_x)
    y1 = max(0, y_min - padding_y)
    x2 = min(width, x_max + padding_x)
    y2 = min(height, y_max + padding_y)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# ==========================================
# 5. Connect to ESP32-S3 CAM
# ==========================================

print("[INFO] Connecting to ESP32-S3 CAM...")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError(
        "Cannot connect to ESP32 camera. Check Wi-Fi and camera URL."
    )

print("[SUCCESS] Camera connected!")
print("[INFO] Waiting for face detection...")
print("[INFO] Press Q to exit.")


# ==========================================
# 6. Process live video
# ==========================================

start_time = time.monotonic()
frame_count = 0

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Frame read failed.")
            time.sleep(0.1)
            continue

        # Rotate frame BEFORE MediaPipe processing
        frame = cv2.rotate(frame, CAMERA_ROTATION)

        frame_count += 1

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            draw_eye_box(
                frame,
                landmarks,
                LEFT_EYE,
                "Eye 1"
            )

            draw_eye_box(
                frame,
                landmarks,
                RIGHT_EYE,
                "Eye 2"
            )

            status = "FACE DETECTED"
            color = (0, 255, 0)

        else:

            status = "NO FACE DETECTED"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.imshow(
            "SmartCap - Eye Detection Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("[INFO] Camera test stopped.")
    print("[INFO] Total processed frames:", frame_count)
