
import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

CAMERA_URL = "http://192.168.4.1/stream"

FACE_MODEL_PATH = (
    PROJECT_DIR / "edge_ai" / "vision" / "models"
    / "face_landmarker.task"
)

EYE_MODEL_PATH = (
    PROJECT_DIR / "edge_ai" / "vision" / "models"
    / "best_eye_model_v4_smartcap.keras"
)

THRESHOLD = 0.40

CAMERA_ROTATION = cv2.ROTATE_180

IMAGE_SIZE = 64


# ==========================================
# 2. Check model files
# ==========================================

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face model not found: {FACE_MODEL_PATH}"
    )

if not EYE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Eye AI model not found: {EYE_MODEL_PATH}"
    )


# ==========================================
# 3. Load V4 AI model
# ==========================================

print("[INFO] Loading SmartCap V4 AI model...")

eye_model = tf.keras.models.load_model(
    str(EYE_MODEL_PATH),
    compile=False
)

print("[SUCCESS] V4 model loaded!")


# ==========================================
# 4. MediaPipe eye landmarks
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
# 5. Create Face Landmarker
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=str(FACE_MODEL_PATH)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ==========================================
# 6. Extract eye image
# ==========================================

def extract_eye(frame, landmarks, indexes):

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

    if x2 <= x1 or y2 <= y1:
        return None, None

    eye_crop = frame[y1:y2, x1:x2]

    if eye_crop.size == 0:
        return None, None

    return eye_crop, (x1, y1, x2, y2)


# ==========================================
# 7. Predict eye state using V4
# ==========================================

def predict_eye(eye_crop):

    # Convert BGR to grayscale
    gray = cv2.cvtColor(
        eye_crop,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to model input size
    resized = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Normalize pixels to [0, 1]
    normalized = resized.astype("float32") / 255.0

    # Shape: (1, 64, 64, 1)
    model_input = normalized[np.newaxis, :, :, np.newaxis]

    # Model output = probability of CLOSED
    probability_closed = float(
        eye_model.predict(model_input, verbose=0)[0][0]
    )

    if probability_closed >= THRESHOLD:
        state = "CLOSED"
    else:
        state = "OPEN"

    return state, probability_closed


# ==========================================
# 8. Draw prediction on frame
# ==========================================

def draw_prediction(frame, box, state, probability, label):

    x1, y1, x2, y2 = box

    if state == "CLOSED":
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    text = (
        f"{label}: {state} "
        f"({probability * 100:.1f}%)"
    )

    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


# ==========================================
# 9. Connect to ESP32-S3 CAM
# ==========================================

print("[INFO] Connecting to ESP32-S3 CAM...")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError(
        "Cannot connect to ESP32-S3 CAM. Check Wi-Fi and URL."
    )

print("[SUCCESS] Camera connected!")
print("[INFO] V4 eye detection started.")
print("[INFO] Press Q to exit.")


# ==========================================
# 10. Real-time processing
# ==========================================

start_time = time.monotonic()
frame_count = 0

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Failed to read camera frame.")
            time.sleep(0.1)
            continue

        # Correct camera orientation
        frame = cv2.rotate(
            frame,
            CAMERA_ROTATION
        )

        frame_count += 1

        # Convert frame to RGB for MediaPipe
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

            eye_configs = [
                ("LEFT EYE", LEFT_EYE),
                ("RIGHT EYE", RIGHT_EYE)
            ]

            for label, indexes in eye_configs:

                eye_crop, box = extract_eye(
                    frame,
                    landmarks,
                    indexes
                )

                if eye_crop is None:
                    continue

                state, probability = predict_eye(
                    eye_crop
                )

                draw_prediction(
                    frame,
                    box,
                    state,
                    probability,
                    label
                )

            status = "FACE DETECTED"
            status_color = (0, 255, 0)

        else:

            status = "NO FACE DETECTED"
            status_color = (0, 0, 255)

        # Display system status
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"V4 Threshold: {THRESHOLD}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "SmartCap - V4 AI Eye Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("[INFO] SmartCap V4 test stopped.")
    print("[INFO] Total frames:", frame_count)
