
from pathlib import Path
import cv2
import csv
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR
    / "input_videos"
    / "SmartCap_20261112_163011"
    / "smartcap_test.mp4"
)

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

OUTPUT_DIR = BASE_DIR / "output"
MISSED_DIR = OUTPUT_DIR / "missed_frames"

OUTPUT_VIDEO = OUTPUT_DIR / "face_landmarks_output.mp4"
LANDMARKS_CSV = OUTPUT_DIR / "face_landmarks.csv"
BLENDSHAPES_CSV = OUTPUT_DIR / "face_blendshapes.csv"
TRANSFORM_CSV = OUTPUT_DIR / "face_transformation_matrix.csv"
MISSED_CSV = OUTPUT_DIR / "missed_frames.csv"


# ============================================================
# SETTINGS
# ============================================================

MIN_FACE_DETECTION_CONFIDENCE = 0.35
MIN_FACE_PRESENCE_CONFIDENCE = 0.35
MIN_TRACKING_CONFIDENCE = 0.35

MAX_MISSED_IMAGES = 30


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MISSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print(f"Video found: {VIDEO_PATH}")
print(f"Model found: {MODEL_PATH}")


# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

mp_vision = mp.tasks.vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=mp_vision.RunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
    min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,

    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)


# ============================================================
# IMAGE QUALITY METRICS
# ============================================================

def calculate_brightness(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return float(gray.mean())


def calculate_blur(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def estimate_reason(brightness, blur):

    if brightness < 50:
        return "very_dark"

    if brightness < 75:
        return "dark"

    if blur < 30:
        return "very_blurry"

    if blur < 80:
        return "blurry"

    return "unknown"


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        "Could not open input video."
    )

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS.")


print("\nVideo information:")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")

print("\nDetection settings:")
print(
    f"Face detection confidence: "
    f"{MIN_FACE_DETECTION_CONFIDENCE}"
)
print(
    f"Face presence confidence: "
    f"{MIN_FACE_PRESENCE_CONFIDENCE}"
)
print(
    f"Tracking confidence: "
    f"{MIN_TRACKING_CONFIDENCE}"
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# COUNTERS
# ============================================================

processed_frames = 0
detected_frames = 0
missed_frames = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

print("\nProcessing video...\n")

try:

    with open(
        LANDMARKS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as landmark_file, open(
        BLENDSHAPES_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as blendshape_file, open(
        TRANSFORM_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as transform_file, open(
        MISSED_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as missed_file:

        landmark_writer = csv.writer(landmark_file)
        blendshape_writer = csv.writer(blendshape_file)
        transform_writer = csv.writer(transform_file)
        missed_writer = csv.writer(missed_file)

        # ====================================================
        # CSV HEADERS
        # ====================================================

        landmark_header = [
            "frame",
            "timestamp_ms"
        ]

        for i in range(478):

            landmark_header.extend([
                f"x_{i}",
                f"y_{i}",
                f"z_{i}"
            ])

        landmark_writer.writerow(landmark_header)

        blendshape_writer.writerow([
            "frame",
            "timestamp_ms",
            "blendshape",
            "score"
        ])

        transform_header = [
            "frame",
            "timestamp_ms"
        ]

        for i in range(16):
            transform_header.append(f"m{i}")

        transform_writer.writerow(transform_header)

        missed_writer.writerow([
            "frame",
            "timestamp_ms",
            "brightness",
            "blur",
            "reason"
        ])

        # ====================================================
        # MEDIAPIPE FACE LANDMARKER
        # ====================================================

        with FaceLandmarker.create_from_options(options) as landmarker:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                processed_frames += 1

                # --------------------------------------------
                # TIMESTAMP
                # --------------------------------------------

                timestamp_ms = int(
                    (processed_frames - 1) * 1000 / fps
                )

                # --------------------------------------------
                # USE ORIGINAL FRAME
                # NO CLAHE / NO PREPROCESSING
                # --------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # --------------------------------------------
                # FACE DETECTION
                # --------------------------------------------

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # --------------------------------------------
                # FACE DETECTED
                # --------------------------------------------

                if result.face_landmarks:

                    detected_frames += 1

                    face_landmarks = result.face_landmarks[0]

                    # ========================================
                    # SAVE 478 LANDMARKS
                    # ========================================

                    row = [
                        processed_frames,
                        timestamp_ms
                    ]

                    for landmark in face_landmarks:

                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    landmark_writer.writerow(row)

                    # ========================================
                    # SAVE BLENDSHAPES
                    # ========================================

                    if result.face_blendshapes:

                        blendshapes = result.face_blendshapes[0]

                        for blendshape in blendshapes:

                            blendshape_writer.writerow([
                                processed_frames,
                                timestamp_ms,
                                blendshape.category_name,
                                blendshape.score
                            ])

                    # ========================================
                    # SAVE TRANSFORMATION MATRIX
                    # ========================================

                    if result.facial_transformation_matrixes:

                        matrix = (
                            result.facial_transformation_matrixes[0]
                        )

                        matrix_values = []

                        for r in range(4):

                            for c in range(4):

                                matrix_values.append(
                                    matrix[r][c]
                                )

                        transform_writer.writerow([
                            processed_frames,
                            timestamp_ms
                        ] + matrix_values)

                    # ========================================
                    # DRAW LANDMARKS
                    # ========================================

                    annotated_frame = frame.copy()

                    for landmark in face_landmarks:

                        x = int(
                            landmark.x * width
                        )

                        y = int(
                            landmark.y * height
                        )

                        if (
                            0 <= x < width
                            and 0 <= y < height
                        ):

                            cv2.circle(
                                annotated_frame,
                                (x, y),
                                1,
                                (0, 255, 0),
                                -1
                            )

                    cv2.putText(
                        annotated_frame,
                        "FACE DETECTED",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                # --------------------------------------------
                # FACE NOT DETECTED
                # --------------------------------------------

                else:

                    missed_frames += 1

                    brightness = calculate_brightness(frame)
                    blur = calculate_blur(frame)

                    reason = estimate_reason(
                        brightness,
                        blur
                    )

                    missed_writer.writerow([
                        processed_frames,
                        timestamp_ms,
                        round(brightness, 2),
                        round(blur, 2),
                        reason
                    ])

                    annotated_frame = frame.copy()

                    cv2.putText(
                        annotated_frame,
                        "FACE NOT DETECTED",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Brightness: {brightness:.1f}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Blur: {blur:.1f}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    # ========================================
                    # SAVE SAMPLE MISSED FRAMES
                    # ========================================

                    if missed_frames <= MAX_MISSED_IMAGES:

                        missed_image_path = (
                            MISSED_DIR
                            / f"missed_{processed_frames:04d}_{reason}.jpg"
                        )

                        cv2.imwrite(
                            str(missed_image_path),
                            frame
                        )

                # --------------------------------------------
                # WRITE OUTPUT VIDEO
                # --------------------------------------------

                writer.write(annotated_frame)

                # --------------------------------------------
                # PROGRESS
                # --------------------------------------------

                if processed_frames % 30 == 0:

                    print(
                        f"Processed "
                        f"{processed_frames}/{total_frames} frames"
                    )

finally:

    cap.release()
    writer.release()


# ============================================================
# FINAL REPORT
# ============================================================

detection_rate = (
    detected_frames / processed_frames * 100
    if processed_frames > 0
    else 0
)

miss_rate = (
    missed_frames / processed_frames * 100
    if processed_frames > 0
    else 0
)

print("\n" + "=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)

print(f"Total processed frames : {processed_frames}")
print(f"Detected frames        : {detected_frames}")
print(f"Missed frames          : {missed_frames}")

print(f"Detection rate         : {detection_rate:.2f}%")
print(f"Miss rate              : {miss_rate:.2f}%")

print("\nOutput files:")

print(f"Annotated video:\n{OUTPUT_VIDEO}")

print(f"\nLandmarks CSV:\n{LANDMARKS_CSV}")

print(f"\nBlendshapes CSV:\n{BLENDSHAPES_CSV}")

print(f"\nTransformation Matrix CSV:\n{TRANSFORM_CSV}")

print(f"\nMissed frames CSV:\n{MISSED_CSV}")

print(f"\nSample missed images:\n{MISSED_DIR}")

print("=" * 60)
