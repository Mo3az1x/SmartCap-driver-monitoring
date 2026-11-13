import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LANDMARKS_CSV = os.path.join(
    BASE_DIR,
    "output",
    "face_landmarks.csv"
)

CNN_CSV = os.path.join(
    BASE_DIR,
    "output",
    "cnn_test",
    "mobilenetv2_predictions.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "output",
    "drowsiness_features.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(LANDMARKS_CSV):
    raise FileNotFoundError(
        f"Landmarks CSV not found:\n{LANDMARKS_CSV}"
    )

if not os.path.exists(CNN_CSV):
    raise FileNotFoundError(
        f"CNN CSV not found:\n{CNN_CSV}"
    )


print("=" * 70)
print("SmartCap Drowsiness Feature Analysis")
print("=" * 70)

print("\nLoading MediaPipe landmarks...")
landmarks_df = pd.read_csv(LANDMARKS_CSV)

print(f"Landmarks rows: {len(landmarks_df)}")

print("\nLoading CNN predictions...")
cnn_df = pd.read_csv(CNN_CSV)

print(f"CNN rows: {len(cnn_df)}")


# ============================================================
# SHOW COLUMNS
# ============================================================

print("\nLandmarks CSV columns:")
for col in landmarks_df.columns[:20]:
    print(" ", col)

if len(landmarks_df.columns) > 20:
    print(f"  ... and {len(landmarks_df.columns) - 20} more columns")


# ============================================================
# DETECT LANDMARK COLUMN NAMES
# ============================================================

def find_landmark_column(x, coordinate):
    """
    Find a column corresponding to landmark x/y.

    Expected possible formats:
        landmark_0_x
        landmark_0_y
        x_0
        y_0
        lm_0_x
        lm_0_y
    """

    possible_names = [
        f"landmark_{x}_{coordinate}",
        f"landmark{x}_{coordinate}",
        f"lm_{x}_{coordinate}",
        f"lm{x}_{coordinate}",
        f"{coordinate}_{x}",
        f"{coordinate}{x}",
    ]

    for name in possible_names:
        if name in landmarks_df.columns:
            return name

    return None


# ============================================================
# FIND FRAME / TIMESTAMP COLUMNS
# ============================================================

frame_candidates = [
    "frame",
    "frame_number",
    "frame_id"
]

timestamp_candidates = [
    "timestamp_ms",
    "timestamp",
    "timestamp_seconds",
    "time"
]

frame_col = None
timestamp_col = None

for col in frame_candidates:
    if col in landmarks_df.columns:
        frame_col = col
        break

for col in timestamp_candidates:
    if col in landmarks_df.columns:
        timestamp_col = col
        break


print("\nDetected columns:")

print("Frame column:", frame_col)
print("Timestamp column:", timestamp_col)


# ============================================================
# IMPORTANT:
# MediaPipe Face Landmarker uses 478 landmarks.
#
# We need actual eye/mouth landmark indices.
#
# Standard MediaPipe Face Mesh indices:
#
# LEFT EYE:
# 33, 160, 158, 133, 153, 144
#
# RIGHT EYE:
# 362, 385, 387, 263, 373, 380
#
# MOUTH:
# 61, 291, 13, 14
#
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]

RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_VERTICAL = [13, 14]
MOUTH_HORIZONTAL = [61, 291]


# ============================================================
# GET LANDMARK COORDINATES
# ============================================================

def get_xy(df, landmark_index):
    x_col = find_landmark_column(landmark_index, "x")
    y_col = find_landmark_column(landmark_index, "y")

    if x_col is None or y_col is None:
        raise KeyError(
            f"Could not find x/y columns for landmark {landmark_index}."
        )

    return (
        pd.to_numeric(df[x_col], errors="coerce").to_numpy(),
        pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    )


# ============================================================
# DISTANCE
# ============================================================

def distance(x1, y1, x2, y2):
    return np.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


# ============================================================
# EAR
# ============================================================

def calculate_ear(df, eye_indices):
    """
    EAR =
    (||p2-p6|| + ||p3-p5||)
    /
    (2 * ||p1-p4||)

    Lower EAR generally indicates a more closed eye.
    """

    points = []

    for idx in eye_indices:
        x, y = get_xy(df, idx)
        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = distance(
        p2[0], p2[1],
        p6[0], p6[1]
    )

    vertical_2 = distance(
        p3[0], p3[1],
        p5[0], p5[1]
    )

    horizontal = distance(
        p1[0], p1[1],
        p4[0], p4[1]
    )

    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal + 1e-8
    )

    return ear


# ============================================================
# MAR
# ============================================================

def calculate_mar(df):
    """
    Simple Mouth Aspect Ratio:

    MAR = mouth vertical opening /
          mouth horizontal width
    """

    x13, y13 = get_xy(df, 13)
    x14, y14 = get_xy(df, 14)

    x61, y61 = get_xy(df, 61)
    x291, y291 = get_xy(df, 291)

    vertical = distance(
        x13, y13,
        x14, y14
    )

    horizontal = distance(
        x61, y61,
        x291, y291
    )

    mar = vertical / (
        horizontal + 1e-8
    )

    return mar


# ============================================================
# CALCULATE FEATURES
# ============================================================

print("\nCalculating EAR...")

left_ear = calculate_ear(
    landmarks_df,
    LEFT_EYE
)

right_ear = calculate_ear(
    landmarks_df,
    RIGHT_EYE
)

average_ear = (
    left_ear + right_ear
) / 2.0


print("Calculating MAR...")

mar = calculate_mar(
    landmarks_df
)


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

features_df = pd.DataFrame()

if frame_col is not None:
    features_df["frame_number"] = landmarks_df[frame_col]
else:
    features_df["frame_number"] = np.arange(
        len(landmarks_df)
    )


if timestamp_col is not None:

    timestamp_values = pd.to_numeric(
        landmarks_df[timestamp_col],
        errors="coerce"
    )

    # MediaPipe video timestamps are usually milliseconds.
    if timestamp_col == "timestamp_ms":
        timestamp_values = timestamp_values / 1000.0

    features_df["timestamp_seconds"] = timestamp_values

else:

    # Approximate only if timestamp doesn't exist.
    # 15 FPS is used as a fallback.
    features_df["timestamp_seconds"] = (
        features_df["frame_number"] / 15.0
    )


features_df["left_EAR"] = left_ear
features_df["right_EAR"] = right_ear
features_df["average_EAR"] = average_ear
features_df["MAR"] = mar


# ============================================================
# MERGE WITH CNN
# ============================================================

print("\nMerging CNN predictions...")

cnn_columns = [
    "frame_number",
    "timestamp_seconds",
    "raw_prediction",
    "final_prediction",
    "confidence",
    "alert_probability",
    "yawning_probability",
    "microsleep_probability",
    "yawning_rejected"
]

available_cnn_columns = [
    col for col in cnn_columns
    if col in cnn_df.columns
]

cnn_merge = cnn_df[
    available_cnn_columns
].copy()


# Avoid duplicate timestamp column during merge
if "timestamp_seconds" in cnn_merge.columns:
    cnn_merge = cnn_merge.drop(
        columns=["timestamp_seconds"]
    )


features_df = features_df.merge(
    cnn_merge,
    on="frame_number",
    how="left"
)


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

print("\nEAR:")
print(
    features_df[
        [
            "left_EAR",
            "right_EAR",
            "average_EAR"
        ]
    ].describe()
)

print("\nMAR:")
print(
    features_df["MAR"].describe()
)


# ============================================================
# FIND LOW EAR FRAMES
# ============================================================

print("\n" + "=" * 70)
print("LOW EAR ANALYSIS")
print("=" * 70)

ear_threshold = 0.20

features_df["eyes_closed_candidate"] = (
    features_df["average_EAR"] < ear_threshold
)

closed_count = features_df[
    "eyes_closed_candidate"
].sum()

print(
    f"\nFrames with EAR < {ear_threshold}: "
    f"{closed_count}"
)

print(
    f"Percentage: "
    f"{closed_count / len(features_df) * 100:.2f}%"
)


# ============================================================
# COMPARE MICROSLEEP WITH EAR
# ============================================================

print("\n" + "=" * 70)
print("MICROSLEEP vs EAR")
print("=" * 70)

microsleep_mask = (
    features_df["final_prediction"]
    == "Microsleep"
)

microsleep_rows = features_df[
    microsleep_mask
]

if len(microsleep_rows) > 0:

    print(
        f"\nMicrosleep frames: "
        f"{len(microsleep_rows)}"
    )

    print(
        "\nAverage EAR during Microsleep prediction:"
    )

    print(
        microsleep_rows[
            "average_EAR"
        ].mean()
    )

    print(
        "\nPercentage of Microsleep frames "
        "with EAR below threshold:"
    )

    microsleep_closed = (
        microsleep_rows[
            "eyes_closed_candidate"
        ].mean() * 100
    )

    print(
        f"{microsleep_closed:.2f}%"
    )


# ============================================================
# COMPARE YAWNING WITH MAR
# ============================================================

print("\n" + "=" * 70)
print("YAWNING vs MAR")
print("=" * 70)

yawning_mask = (
    features_df["final_prediction"]
    == "Yawning"
)

yawning_rows = features_df[
    yawning_mask
]

if len(yawning_rows) > 0:

    print(
        f"\nYawning frames: "
        f"{len(yawning_rows)}"
    )

    print(
        "\nAverage MAR during Yawning prediction:"
    )

    print(
        yawning_rows[
            "MAR"
        ].mean()
    )


# ============================================================
# SAVE
# ============================================================

features_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nSaved analysis file:\n"
    f"{OUTPUT_CSV}"
)

print("\nColumns in output:")

for col in features_df.columns:
    print(" ", col)

# ============================================================
# DETAILED DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("FRAME ALIGNMENT CHECK")
print("=" * 70)

print("\nLandmarks frame range:")
print(landmarks_df["frame"].min(), "to", landmarks_df["frame"].max())

print("\nCNN frame range:")
print(cnn_df["frame_number"].min(), "to", cnn_df["frame_number"].max())

print("\nMatched rows:", features_df["final_prediction"].notna().sum())

print("\n" + "=" * 70)
print("MICROSLEEP FRAMES WITH EAR")
print("=" * 70)

cols = [
    "frame_number",
    "timestamp_seconds",
    "average_EAR",
    "MAR",
    "final_prediction",
    "microsleep_probability"
]

print(
    features_df.loc[
        features_df["final_prediction"] == "Microsleep",
        cols
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("LOWEST EAR FRAMES")
print("=" * 70)

print(
    features_df.nsmallest(15, "average_EAR")[
        cols
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("HIGHEST MAR FRAMES")
print("=" * 70)

print(
    features_df.nlargest(15, "MAR")[
        cols
    ].to_string(index=False)
)