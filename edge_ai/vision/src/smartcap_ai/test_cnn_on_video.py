
from pathlib import Path
import csv
import cv2
import torch
import numpy as np

from torchvision import models
from torchvision.transforms import Normalize


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fl3d_cnn_mobilenetv2_best.pth"

VIDEO_PATH = (
    BASE_DIR
    / "input_videos"
    / "SmartCap_20261112_163011"
    / "smartcap_test.mp4"
)

OUTPUT_DIR = BASE_DIR / "output" / "cnn_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "mobilenetv2_predictions.mp4"
OUTPUT_CSV = OUTPUT_DIR / "mobilenetv2_predictions.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "mobilenetv2_summary.txt"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

CLASS_NAMES = {
    0: "Alert",
    1: "Yawning",
    2: "Microsleep"
}

NUM_CLASSES = 3

IMAGE_SIZE = (224, 224)

DEVICE = torch.device("cpu")

# Initial experimental threshold for Yawning
YAWNING_THRESHOLD = 0.80

normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ============================================================
# 3. CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Input video not found:\n{VIDEO_PATH}"
    )


# ============================================================
# 4. LOAD MOBILENETV2
# ============================================================

print("Loading MobileNetV2...")

model = models.mobilenet_v2(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.last_channel,
    NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint, strict=True)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)
print("Yawning threshold:", YAWNING_THRESHOLD)


# ============================================================
# 5. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or fps <= 0:
    fps = 25.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVideo information:")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", total_frames)


# ============================================================
# 6. OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")


# ============================================================
# 7. STATISTICS
# ============================================================

class_counts = {
    name: 0 for name in CLASS_NAMES.values()
}

raw_class_counts = {
    name: 0 for name in CLASS_NAMES.values()
}

confidence_values = {
    name: [] for name in CLASS_NAMES.values()
}

processed_frames = 0
rejected_yawning_frames = 0


# ============================================================
# 8. PROCESS VIDEO
# ============================================================

print("\nStarting CNN inference...\n")

try:

    with open(
        OUTPUT_CSV,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "frame_number",
            "timestamp_seconds",
            "raw_prediction",
            "final_prediction",
            "confidence",
            "alert_probability",
            "yawning_probability",
            "microsleep_probability",
            "yawning_rejected"
        ])

        with torch.inference_mode():

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                processed_frames += 1

                # --------------------------------------------
                # Prepare image
                # --------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                resized = cv2.resize(
                    rgb_frame,
                    IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )

                image_array = (
                    resized.astype(np.float32) / 255.0
                )

                image_tensor = torch.from_numpy(
                    image_array
                )

                image_tensor = image_tensor.permute(2, 0, 1)

                image_tensor = normalize(image_tensor)

                image_tensor = image_tensor.unsqueeze(0)

                image_tensor = image_tensor.to(DEVICE)

                # --------------------------------------------
                # CNN prediction
                # --------------------------------------------

                outputs = model(image_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )[0]

                probs = probabilities.cpu().numpy()

                raw_index = int(np.argmax(probs))

                raw_prediction = CLASS_NAMES[raw_index]

                alert_prob = float(probs[0])
                yawning_prob = float(probs[1])
                microsleep_prob = float(probs[2])

                raw_class_counts[raw_prediction] += 1

                # --------------------------------------------
                # Yawning threshold
                # --------------------------------------------

                yawning_rejected = False

                if (
                    raw_prediction == "Yawning"
                    and yawning_prob < YAWNING_THRESHOLD
                ):

                    yawning_rejected = True
                    rejected_yawning_frames += 1

                    # Choose between Alert and Microsleep
                    if alert_prob >= microsleep_prob:
                        predicted_class = "Alert"
                    else:
                        predicted_class = "Microsleep"

                    confidence = max(
                        alert_prob,
                        microsleep_prob
                    )

                else:

                    predicted_class = raw_prediction
                    confidence = float(probs[raw_index])

                # --------------------------------------------
                # Statistics
                # --------------------------------------------

                class_counts[predicted_class] += 1

                confidence_values[predicted_class].append(
                    confidence
                )

                timestamp = (processed_frames - 1) / fps

                csv_writer.writerow([
                    processed_frames,
                    round(timestamp, 3),
                    raw_prediction,
                    predicted_class,
                    round(confidence, 5),
                    round(alert_prob, 5),
                    round(yawning_prob, 5),
                    round(microsleep_prob, 5),
                    yawning_rejected
                ])

                # --------------------------------------------
                # Draw prediction without background
                # --------------------------------------------

                class_colors = {
                    "Alert": (0, 255, 0),
                    "Yawning": (0, 165, 255),
                    "Microsleep": (0, 0, 255)
                }

                text_color = class_colors[predicted_class]

                main_text = (
                    f"{predicted_class}: "
                    f"{confidence * 100:.1f}%"
                )

                # Main text outline
                cv2.putText(
                    frame,
                    main_text,
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    main_text,
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    text_color,
                    2,
                    cv2.LINE_AA
                )

                probability_text = (
                    f"Alert: {alert_prob:.2f} | "
                    f"Yawning: {yawning_prob:.2f} | "
                    f"Microsleep: {microsleep_prob:.2f}"
                )

                # Probability text outline
                cv2.putText(
                    frame,
                    probability_text,
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    probability_text,
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                writer.write(frame)

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                if processed_frames % 100 == 0:

                    print(
                        f"Processed "
                        f"{processed_frames}/{total_frames} frames"
                    )

finally:

    cap.release()
    writer.release()


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n========== CNN TEST SUMMARY ==========")

summary_lines = [
    "MobileNetV2 SmartCap Video Test",
    "=" * 40,
    f"Total video frames: {total_frames}",
    f"Processed frames: {processed_frames}",
    f"FPS: {fps:.2f}",
    f"Yawning threshold: {YAWNING_THRESHOLD}",
    "",
    "Raw predictions before threshold:",
]

for name, count in raw_class_counts.items():

    percentage = (
        count / processed_frames * 100
        if processed_frames else 0
    )

    line = f"{name}: {count} ({percentage:.2f}%)"

    print(line)
    summary_lines.append(line)

summary_lines.append("")
summary_lines.append("Final predictions after threshold:")

for name, count in class_counts.items():

    percentage = (
        count / processed_frames * 100
        if processed_frames else 0
    )

    values = confidence_values[name]

    average_confidence = (
        sum(values) / len(values)
        if values else 0
    )

    line = (
        f"{name}: {count} ({percentage:.2f}%), "
        f"average confidence: "
        f"{average_confidence * 100:.2f}%"
    )

    print(line)
    summary_lines.append(line)

summary_lines.append("")
summary_lines.append(
    f"Rejected Yawning frames: {rejected_yawning_frames}"
)

summary_lines.append(f"Output video: {OUTPUT_VIDEO}")
summary_lines.append(f"Output CSV: {OUTPUT_CSV}")

print(
    "\nRejected Yawning frames:",
    rejected_yawning_frames
)

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as summary_file:

    summary_file.write("\n".join(summary_lines))

print("\n======================================")
print("Processing completed successfully.")

print("\nGenerated files:")
print(OUTPUT_VIDEO)
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)
