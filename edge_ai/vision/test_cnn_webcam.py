
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. Paths and CNN Model
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_eye_model.keras"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face Landmarker model not found: {FACE_MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")


# ==========================================
# 2. MediaPipe Face Landmarker
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

face_landmarker = vision.FaceLandmarker.create_from_options(options)

print("MediaPipe Face Landmarker loaded!")


# ==========================================
# 3. Eye Landmarks
# ==========================================

LEFT_EYE = [33, 133, 159, 145, 160, 144]
RIGHT_EYE = [362, 263, 386, 374, 385, 380]


# ==========================================
# 4. Extract Eye
# ==========================================

def extract_eye(frame, landmarks, indices):

    h, w = frame.shape[:2]

    points = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)]
        for i in indices
    ])

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    x_padding = int(eye_width * 0.30)
    y_padding = int(eye_height * 0.80)

    x1 = max(0, x_min - x_padding)
    x2 = min(w, x_max + x_padding)

    y1 = max(0, y_min - y_padding)
    y2 = min(h, y_max + y_padding)

    eye = frame[y1:y2, x1:x2]

    if eye.size == 0:
        return None

    # Convert to grayscale
    eye_gray = cv2.cvtColor(
        eye,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to training image size
    eye_resized = cv2.resize(
        eye_gray,
        (64, 64)
    )

    # Normalize exactly like training
    eye_normalized = (
        eye_resized.astype("float32") / 255.0
    )

    eye_input = np.expand_dims(
        np.expand_dims(eye_normalized, axis=0),
        axis=-1
    )

    return eye_input, eye_resized


# ==========================================
# 5. CNN Prediction
# ==========================================

def predict_eye(eye_input):

    prediction = model.predict(
        eye_input,
        verbose=0
    )

    raw_output = float(prediction[0][0])

    # Training labels:
    # 0 = OPEN
    # 1 = CLOSED
    #
    # Therefore sigmoid output = P(CLOSED)

    p_closed = raw_output
    p_open = 1.0 - raw_output

    state = (
        "CLOSED"
        if p_closed >= 0.5
        else "OPEN"
    )

    return state, p_open, p_closed, raw_output


# ==========================================
# 6. Start Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    face_landmarker.close()

    raise RuntimeError(
        "Cannot open webcam. Check camera index."
    )

print("\nWebcam started!")
print("Press Q to quit.")
print("Press S to save eye crops.\n")


# ==========================================
# 7. Main Loop
# ==========================================

frame_timestamp_ms = 0

while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp_ms += 33

    results = face_landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    eye_crops = {}

    if results.face_landmarks:

        landmarks = results.face_landmarks[0]

        eye_data = [
            ("LEFT", LEFT_EYE),
            ("RIGHT", RIGHT_EYE)
        ]

        for eye_name, indices in eye_data:

            result = extract_eye(
                frame,
                landmarks,
                indices
            )

            if result is None:
                continue

            eye_input, eye_crop = result

            eye_crops[eye_name] = eye_crop

            state, p_open, p_closed, raw_output = (
                predict_eye(eye_input)
            )

            color = (
                (0, 255, 0)
                if state == "OPEN"
                else (0, 0, 255)
            )

            y_position = (
                40 if eye_name == "LEFT" else 100
            )

            # Display eye state
            cv2.putText(
                frame,
                f"{eye_name}: {state}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Display probabilities
            cv2.putText(
                frame,
                f"Open: {p_open:.3f} Closed: {p_closed:.3f}",
                (20, y_position + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            # Print probabilities in terminal
            print(
                f"{eye_name} | "
                f"Raw Output: {raw_output:.4f} | "
                f"P(Open): {p_open:.4f} | "
                f"P(Closed): {p_closed:.4f} | "
                f"State: {state}"
            )

            # Display extracted eye
            cv2.imshow(
                f"{eye_name} Eye Crop",
                eye_crop
            )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "CNN Eye Detection",
        frame
    )

    # ======================================
    # Keyboard Controls
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        debug_dir = BASE_DIR / "debug_crops"

        debug_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for eye_name, eye_crop in eye_crops.items():

            save_path = (
                debug_dir / f"{eye_name.lower()}.jpg"
            )

            cv2.imwrite(
                str(save_path),
                eye_crop
            )

            print(f"Saved: {save_path}")

    if key == ord("q"):
        break


# ==========================================
# 8. Cleanup
# ==========================================

cap.release()
face_landmarker.close()

cv2.destroyAllWindows()

print("\nTest finished!")
