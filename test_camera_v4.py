
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
IMAGE_SIZE = 64

# Camera is physically upside down
CAMERA_ROTATION = cv2.ROTATE_180

LEFT_EYE = [
    33, 133, 160, 159, 158, 157,
    173, 144, 145, 153, 154, 155
]

RIGHT_EYE = [
    362, 263, 387, 386, 385, 384,
    398, 373, 374, 380, 381, 382
]


# ==========================================
# 2. Check model files
# ==========================================

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face model not found: {FACE_MODEL_PATH}"
    )

if not EYE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Eye model not found: {EYE_MODEL_PATH}"
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
print("[INFO] Input shape:", eye_model.input_shape)
print("[INFO] Output shape:", eye_model.output_shape)


# ==========================================
# 4. Initialize MediaPipe
# ==========================================

print("[INFO] Loading MediaPipe FaceLandmarker...")

base_options = python.BaseOptions(
    model_asset_path=str(FACE_MODEL_PATH)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)

print("[SUCCESS] MediaPipe loaded!")


# ==========================================
# 5. Extract eye crop
# ==========================================

def extract_eye(frame, landmarks, indices):

    height, width = frame.shape[:2]

    x_values = [
        int(landmarks[i].x * width)
        for i in indices
    ]

    y_values = [
        int(landmarks[i].y * height)
        for i in indices
    ]

    x_min = min(x_values)
    x_max = max(x_values)

    y_min = min(y_values)
    y_max = max(y_values)

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    if eye_width < 2 or eye_height < 2:
        return None, None

    pad_x = int(eye_width * 0.35)
    pad_y = int(eye_height * 0.60)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)

    x2 = min(width, x_max + pad_x)
    y2 = min(height, y_max + pad_y)

    eye_crop = frame[y1:y2, x1:x2]

    if eye_crop.size == 0:
        return None, None

    return eye_crop, (x1, y1, x2, y2)


# ==========================================
# 6. V4 AI prediction
# ==========================================

def predict_eye(eye_crop):

    gray = cv2.cvtColor(
        eye_crop,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    normalized = (
        resized.astype("float32") / 255.0
    )

    input_data = normalized.reshape(
        1, IMAGE_SIZE, IMAGE_SIZE, 1
    )

    prediction = eye_model.predict(
        input_data,
        verbose=0
    )

    probability_closed = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    state = (
        "CLOSED"
        if probability_closed >= THRESHOLD
        else "OPEN"
    )

    return state, probability_closed


# ==========================================
# 7. Draw AI prediction
# ==========================================

def draw_prediction(
    frame,
    box,
    label,
    state,
    probability
):

    if box is None:
        return

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

    text_y = max(25, y1 - 10)

    cv2.putText(
        frame,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


# ==========================================
# 8. Connect to ESP32-S3 CAM
# ==========================================

def connect_camera():

    print("[INFO] Connecting to ESP32-S3 CAM...")

    cap = cv2.VideoCapture(
        CAMERA_URL,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        print("[WARNING] FFmpeg connection failed.")
        print("[INFO] Trying default backend...")

        cap.release()

        cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():

        cap.release()

        raise RuntimeError(
            "Cannot connect to ESP32-S3 CAM."
        )

    print("[SUCCESS] Camera connection opened!")

    for attempt in range(20):

        ret, frame = cap.read()

        if ret and frame is not None:

            print(
                "[SUCCESS] Camera frame received:",
                frame.shape
            )

            return cap

        print(
            f"[INFO] Waiting for frame "
            f"{attempt + 1}/20"
        )

        time.sleep(0.2)

    cap.release()

    raise RuntimeError(
        "Camera connected but no frame received."
    )


# ==========================================
# 9. Main AI processing
# ==========================================

cap = None

try:

    cap = connect_camera()

    print("[INFO] SmartCap V4 AI started!")
    print("[INFO] Press Q to exit.")

    start_time = time.monotonic()
    last_timestamp_ms = -1
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:

            print("[WARNING] Failed to read frame.")
            time.sleep(0.1)
            continue

        # Rotate camera
        frame = cv2.rotate(
            frame,
            CAMERA_ROTATION
        )

        frame_count += 1

        # MediaPipe timestamp
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        timestamp_ms = max(
            timestamp_ms,
            last_timestamp_ms + 1
        )

        last_timestamp_ms = timestamp_ms

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Correct MediaPipe Image constructor
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Face detection
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Extract eyes
            left_crop, left_box = extract_eye(
                frame,
                landmarks,
                LEFT_EYE
            )

            right_crop, right_box = extract_eye(
                frame,
                landmarks,
                RIGHT_EYE
            )

            # LEFT EYE AI
            if left_crop is not None:

                left_state, left_probability = predict_eye(
                    left_crop
                )

                draw_prediction(
                    frame,
                    left_box,
                    "LEFT EYE",
                    left_state,
                    left_probability
                )

            # RIGHT EYE AI
            if right_crop is not None:

                right_state, right_probability = predict_eye(
                    right_crop
                )

                draw_prediction(
                    frame,
                    right_box,
                    "RIGHT EYE",
                    right_state,
                    right_probability
                )

            cv2.putText(
                frame,
                "FACE DETECTED | V4 AI ACTIVE",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # FPS
        elapsed = time.monotonic() - start_time

        fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # Show live AI output
        cv2.imshow(
            "SmartCap V4 - Live Eye AI",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


except KeyboardInterrupt:

    print("\n[INFO] Stopped by user.")

finally:

    if cap is not None:
        cap.release()

    landmarker.close()

    cv2.destroyAllWindows()

    print("[INFO] SmartCap V4 test finished.")
