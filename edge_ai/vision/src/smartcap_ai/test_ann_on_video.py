
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp

from pathlib import Path
from collections import Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. PATHS AND SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR / "input_videos" /
    "SmartCap_20261112_163011" / "smartcap_test.mp4"
)

MODEL_PATH = BASE_DIR / "models" / "fl3d_ann_best.pth"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

OUTPUT_DIR = BASE_DIR / "output" / "ann_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "ann_original_vs_roi.mp4"
OUTPUT_CSV = OUTPUT_DIR / "ann_original_vs_roi.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "ann_original_vs_roi_summary.txt"

CLASS_NAMES = {
    0: "Alert",
    1: "Yawning",
    2: "Microsleep"
}

LANDMARK_INDICES = [468, 473, 1, 61, 291]

# Padding around the MediaPipe face bounding box.
# This is an experimental ROI, not a confirmed copy
# of the exact training-image crop.
ROI_PADDING = 0.15

DEVICE = torch.device("cpu")


# =========================================================
# 2. ANN ARCHITECTURE
# =========================================================

class LandmarkANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ANN checkpoint not found: {MODEL_PATH}"
    )

model = LandmarkANN().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

state_dict = {
    key.replace("module.", "", 1)
    if key.startswith("module.") else key: value
    for key, value in state_dict.items()
}

model.load_state_dict(state_dict, strict=True)
model.eval()

print("LandmarkANN loaded successfully.")


# =========================================================
# 4. MEDIAPIPE FACE LANDMARKER
# =========================================================

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {FACE_MODEL_PATH}"
    )

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(FACE_MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.35,
    min_face_presence_confidence=0.35,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

detector = vision.FaceLandmarker.create_from_options(options)


# =========================================================
# 5. FEATURE EXTRACTION
# =========================================================

def extract_original_features(face_landmarks, width, height):

    features = []

    for index in LANDMARK_INDICES:

        point = face_landmarks[index]

        # Original method:
        # Coordinates normalized relative to full frame.

        features.extend([
            point.x,
            point.y
        ])

    return torch.tensor(
        [features],
        dtype=torch.float32,
        device=DEVICE
    )


def get_face_roi(face_landmarks, width, height):

    # Use all detected face landmarks to estimate face bounds.

    xs = [point.x * width for point in face_landmarks]
    ys = [point.y * height for point in face_landmarks]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    face_width = x_max - x_min
    face_height = y_max - y_min

    if face_width <= 0 or face_height <= 0:
        return None

    # Add padding around the detected face.

    pad_x = face_width * ROI_PADDING
    pad_y = face_height * ROI_PADDING

    roi_x1 = max(0, x_min - pad_x)
    roi_y1 = max(0, y_min - pad_y)

    roi_x2 = min(width, x_max + pad_x)
    roi_y2 = min(height, y_max + pad_y)

    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1

    if roi_width <= 0 or roi_height <= 0:
        return None

    return roi_x1, roi_y1, roi_x2, roi_y2


def extract_roi_features(
    face_landmarks,
    roi,
    width,
    height
):

    if roi is None:
        return None

    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1

    features = []

    for index in LANDMARK_INDICES:

        point = face_landmarks[index]

        x_pixel = point.x * width
        y_pixel = point.y * height

        # Normalize coordinates relative to face ROI.

        x_normalized = (x_pixel - roi_x1) / roi_width
        y_normalized = (y_pixel - roi_y1) / roi_height

        features.extend([
            x_normalized,
            y_normalized
        ])

    return torch.tensor(
        [features],
        dtype=torch.float32,
        device=DEVICE
    )


# =========================================================
# 6. ANN PREDICTION
# =========================================================

def predict_ann(input_tensor):

    with torch.inference_mode():

        logits = model(input_tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    return (
        predicted_class,
        probabilities.cpu().tolist(),
        logits[0].cpu().tolist()
    )


# =========================================================
# 7. OPEN VIDEO
# =========================================================

if not VIDEO_PATH.exists():
    detector.close()
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    detector.close()
    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("\n========== VIDEO INFORMATION ==========")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", total_frames)


# =========================================================
# 8. VIDEO WRITER
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():

    cap.release()
    detector.close()

    raise RuntimeError(
        "Could not create output video."
    )


# =========================================================
# 9. STATISTICS
# =========================================================

original_counts = Counter()
roi_counts = Counter()

detected_frames = 0
missed_frames = 0
frame_number = 0

original_probabilities = []
roi_probabilities = []

agreement_count = 0
roi_invalid_count = 0


# =========================================================
# 10. RUN BOTH ANN INPUT METHODS
# =========================================================

try:

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([

            "frame",
            "timestamp_ms",

            "original_prediction",
            "original_class",

            "original_alert_prob",
            "original_yawning_prob",
            "original_microsleep_prob",

            "roi_prediction",
            "roi_class",

            "roi_alert_prob",
            "roi_yawning_prob",
            "roi_microsleep_prob",

            "predictions_agree",

            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2"
        ])

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            timestamp_ms = (
                (frame_number - 1) * 1000.0 / fps
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            if not result.face_landmarks:

                missed_frames += 1

                cv2.putText(
                    frame,
                    "Face not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                csv_writer.writerow([
                    frame_number,
                    round(timestamp_ms, 2),
                    "", "No face",
                    "", "", "",
                    "", "No face",
                    "", "", "",
                    "",
                    "", "", "", ""
                ])

                writer.write(frame)
                continue

            detected_frames += 1

            face_landmarks = result.face_landmarks[0]

            # ---------------------------------------------
            # A. ORIGINAL FULL-FRAME FEATURES
            # ---------------------------------------------

            original_input = extract_original_features(
                face_landmarks,
                width,
                height
            )

            original_class, original_probs, original_logits = (
                predict_ann(original_input)
            )

            original_name = CLASS_NAMES[original_class]

            original_counts[original_name] += 1
            original_probabilities.append(original_probs)

            # ---------------------------------------------
            # B. FACE ROI FEATURES
            # ---------------------------------------------

            roi = get_face_roi(
                face_landmarks,
                width,
                height
            )

            roi_input = extract_roi_features(
                face_landmarks,
                roi,
                width,
                height
            )

            if roi_input is not None:

                roi_class, roi_probs, roi_logits = predict_ann(
                    roi_input
                )

                roi_name = CLASS_NAMES[roi_class]

                roi_counts[roi_name] += 1
                roi_probabilities.append(roi_probs)

                predictions_agree = (
                    original_class == roi_class
                )

                if predictions_agree:
                    agreement_count += 1

            else:

                roi_invalid_count += 1

                roi_class = None
                roi_name = "Invalid ROI"
                roi_probs = [None, None, None]
                roi_logits = []

                predictions_agree = False

            # ---------------------------------------------
            # C. DIAGNOSTIC: FIRST 10 DETECTED FRAMES
            # ---------------------------------------------

            if detected_frames <= 10:

                print(
                    f"\n========== DETECTED FRAME "
                    f"{detected_frames} | VIDEO FRAME "
                    f"{frame_number} =========="
                )

                print("\nOriginal features:")
                print(original_input[0].tolist())

                print("Original logits:")
                print(original_logits)

                print("Original probabilities:")
                print(original_probs)

                if roi_input is not None:

                    print("\nROI features:")
                    print(roi_input[0].tolist())

                    print("ROI logits:")
                    print(roi_logits)

                    print("ROI probabilities:")
                    print(roi_probs)

                print(
                    "\nPredictions:",
                    original_name,
                    "|",
                    roi_name
                )

            # ---------------------------------------------
            # D. DRAW LANDMARKS AND ROI
            # ---------------------------------------------

            for index in LANDMARK_INDICES:

                point = face_landmarks[index]

                x = int(point.x * width)
                y = int(point.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            if roi is not None:

                x1, y1, x2, y2 = [
                    int(value) for value in roi
                ]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            # ---------------------------------------------
            # E. DRAW BOTH PREDICTIONS
            # ---------------------------------------------

            original_text = (
                f"Original: {original_name} "
                f"({original_probs[original_class] * 100:.1f}%)"
            )

            cv2.putText(
                frame,
                original_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            if roi_class is not None:

                roi_text = (
                    f"Face ROI: {roi_name} "
                    f"({roi_probs[roi_class] * 100:.1f}%)"
                )

                roi_color = (
                    (0, 255, 0)
                    if predictions_agree
                    else (0, 165, 255)
                )

            else:

                roi_text = "Face ROI: Invalid"

                roi_color = (0, 0, 255)

            cv2.putText(
                frame,
                roi_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                roi_color,
                2
            )

            writer.write(frame)

            # ---------------------------------------------
            # F. SAVE CSV
            # ---------------------------------------------

            if roi is not None:

                roi_values = [
                    round(value, 2)
                    for value in roi
                ]

            else:

                roi_values = ["", "", "", ""]

            csv_writer.writerow([

                frame_number,
                round(timestamp_ms, 2),

                original_class,
                original_name,

                round(original_probs[0], 6),
                round(original_probs[1], 6),
                round(original_probs[2], 6),

                "" if roi_class is None else roi_class,
                roi_name,

                *[
                    "" if value is None
                    else round(value, 6)
                    for value in roi_probs
                ],

                predictions_agree,

                *roi_values
            ])

            if frame_number % 100 == 0:

                print(
                    f"Processed {frame_number}/"
                    f"{total_frames} frames"
                )


finally:

    cap.release()
    writer.release()
    detector.close()


# =========================================================
# 11. SAVE SUMMARY
# =========================================================

def get_percentage(count, total):

    if total == 0:
        return 0.0

    return count / total * 100


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as file:

    file.write("========== ANN ORIGINAL VS ROI ==========\n\n")

    file.write(f"Total frames: {frame_number}\n")
    file.write(f"Detected frames: {detected_frames}\n")
    file.write(f"Missed frames: {missed_frames}\n")

    file.write(
        f"Invalid ROI frames: {roi_invalid_count}\n"
    )

    detection_rate = get_percentage(
        detected_frames,
        frame_number
    )

    file.write(
        f"Face detection rate: {detection_rate:.2f}%\n\n"
    )

    file.write("========== ORIGINAL PREDICTIONS ==========\n")

    for class_name in CLASS_NAMES.values():

        count = original_counts[class_name]

        file.write(
            f"{class_name}: {count} "
            f"({get_percentage(count, detected_frames):.2f}%)\n"
        )

    file.write("\n========== ROI PREDICTIONS ==========\n")

    valid_roi_frames = sum(roi_counts.values())

    for class_name in CLASS_NAMES.values():

        count = roi_counts[class_name]

        file.write(
            f"{class_name}: {count} "
            f"({get_percentage(count, valid_roi_frames):.2f}%)\n"
        )

    file.write("\n========== COMPARISON ==========\n")

    agreement_rate = get_percentage(
        agreement_count,
        valid_roi_frames
    )

    file.write(
        f"Original vs ROI agreement: "
        f"{agreement_count}/{valid_roi_frames} "
        f"({agreement_rate:.2f}%)\n"
    )

    file.write(
        "\nROI padding: "
        f"{ROI_PADDING * 100:.0f}%\n"
    )

    file.write(
        "\nNo temporal smoothing or thresholds applied.\n"
    )


# =========================================================
# 12. FINAL OUTPUT
# =========================================================

print("\n========== ANN COMPARISON COMPLETED ==========")

print("Total frames:", frame_number)
print("Detected frames:", detected_frames)
print("Missed frames:", missed_frames)
print("Invalid ROI frames:", roi_invalid_count)

print("\n========== ORIGINAL PREDICTIONS ==========")

for class_name in CLASS_NAMES.values():

    count = original_counts[class_name]

    percentage = get_percentage(
        count,
        detected_frames
    )

    print(
        f"{class_name}: {count} ({percentage:.2f}%)"
    )

print("\n========== ROI PREDICTIONS ==========")

valid_roi_frames = sum(roi_counts.values())

for class_name in CLASS_NAMES.values():

    count = roi_counts[class_name]

    percentage = get_percentage(
        count,
        valid_roi_frames
    )

    print(
        f"{class_name}: {count} ({percentage:.2f}%)"
    )

print("\n========== AGREEMENT ==========")

print(
    f"{agreement_count}/{valid_roi_frames} "
    f"({get_percentage(agreement_count, valid_roi_frames):.2f}%)"
)

print("\nSaved files:")
print(OUTPUT_VIDEO)
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)
