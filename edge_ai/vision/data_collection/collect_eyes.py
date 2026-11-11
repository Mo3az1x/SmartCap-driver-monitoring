
import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "edge_ai"
    / "vision"
    / "models"
    / "face_landmarker.task"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "edge_ai"
    / "vision"
    / "data_raw"
)


# =========================================================
# MEDIAPIPE EYE LANDMARKS
# =========================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =========================================================
# SETTINGS
# =========================================================

EAR_THRESHOLD = 0.20

# Camera is physically mounted upside down.
CAMERA_ROTATION = 180


# =========================================================
# CAMERA ORIENTATION
# =========================================================

def fix_camera_orientation(frame):

    if CAMERA_ROTATION == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)

    return frame


# =========================================================
# DISTANCE
# =========================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1, dtype=np.float32)
        - np.array(p2, dtype=np.float32)
    )


# =========================================================
# EAR CALCULATION
# =========================================================

def calculate_ear(landmarks, indices):

    points = [
        (landmarks[i].x, landmarks[i].y)
        for i in indices
    ]

    p1, p2, p3, p4, p5, p6 = points

    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return None

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return float(ear)


# =========================================================
# EYE CROPPING
# =========================================================

def crop_eye(frame, landmarks, indices):

    height, width = frame.shape[:2]

    points = []

    for index in indices:

        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    points = np.array(points)

    x_min = int(points[:, 0].min())
    x_max = int(points[:, 0].max())

    y_min = int(points[:, 1].min())
    y_max = int(points[:, 1].max())

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    pad_x = int(eye_width * 0.30)
    pad_y = int(max(eye_height, 5) * 0.50) + 5

    x_min = max(0, x_min - pad_x)
    x_max = min(width, x_max + pad_x)

    y_min = max(0, y_min - pad_y)
    y_max = min(height, y_max + pad_y)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = frame[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    return gray


# =========================================================
# SAFE IMAGE SAVE
# =========================================================

def safe_save_image(image, final_path):

    """
    Save image to a temporary PNG first.
    Verify that the image exists, is readable,
    and has the expected dimensions.
    Then rename it to the final filename.
    """

    final_path = Path(final_path)

    temp_path = final_path.with_name(
        final_path.stem + "_temp.png"
    )

    try:

        success = cv2.imwrite(
            str(temp_path),
            image
        )

        if not success:
            print("ERROR: cv2.imwrite failed:", temp_path)
            return False

        if not temp_path.exists():
            print("ERROR: Temporary image was not created.")
            return False

        if temp_path.stat().st_size == 0:
            print("ERROR: Temporary image is empty.")
            temp_path.unlink(missing_ok=True)
            return False

        check_image = cv2.imread(
            str(temp_path),
            cv2.IMREAD_GRAYSCALE
        )

        if check_image is None:
            print("ERROR: Saved image cannot be read.")
            temp_path.unlink(missing_ok=True)
            return False

        if check_image.shape != (64, 64):
            print("ERROR: Incorrect image dimensions:", check_image.shape)
            temp_path.unlink(missing_ok=True)
            return False

        os.replace(
            str(temp_path),
            str(final_path)
        )

        if not final_path.exists() or final_path.stat().st_size == 0:
            print("ERROR: Final image save verification failed.")
            return False

        return True

    except Exception as error:

        print("IMAGE SAVE ERROR:", error)

        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        return False


# =========================================================
# DRAW EYE POINTS
# =========================================================

def draw_eye_points(frame, landmarks, indices):

    height, width = frame.shape[:2]

    for index in indices:

        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 0),
            -1
        )


# =========================================================
# CLASSIFY EYE USING EAR
# =========================================================

def estimate_eye_state(average_ear):

    if average_ear is None:
        return "UNKNOWN"

    if average_ear >= EAR_THRESHOLD:
        return "OPEN"

    return "CLOSED"


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="SmartCap Eye Dataset Collector"
    )

    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", default="s01")

    parser.add_argument(
        "--url",
        default="http://192.168.4.1/stream"
    )

    parser.add_argument("--lighting", default="normal")
    parser.add_argument("--glasses", default="0")
    parser.add_argument("--placement", default="cap")

    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--skip", type=float, default=1.5)

    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))

    args = parser.parse_args()

    # -----------------------------------------------------
    # PATHS
    # -----------------------------------------------------

    output_root = Path(args.out)

    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    model_path = Path(args.model)

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():

        print("ERROR: Face Landmarker model not found:")
        print(model_path)
        return

    session_dir = (
        output_root
        / args.subject
        / args.session
    )

    open_dir = session_dir / "open"
    closed_dir = session_dir / "closed"

    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    label_directories = {
        "open": open_dir,
        "closed": closed_dir
    }

    csv_path = session_dir / "log.csv"
    metadata_path = session_dir / "meta.json"

    metadata = {
        "subject": args.subject,
        "session": args.session,
        "lighting": args.lighting,
        "glasses": args.glasses,
        "placement": args.placement,
        "camera_url": args.url,
        "camera_rotation": CAMERA_ROTATION,
        "image_size": "64x64",
        "label_method": "manual_keyboard_selection",
        "ear_threshold": EAR_THRESHOLD,
        "created_at": datetime.now().isoformat()
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metadata, file, indent=4)

    # -----------------------------------------------------
    # MEDIAPIPE
    # -----------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    print("\nConnecting to ESP32-S3 camera...")
    print("URL:", args.url)
    print("Camera rotation:", CAMERA_ROTATION)

    cap = cv2.VideoCapture(args.url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():

        print("ERROR: Cannot open camera stream.")

        landmarker.close()
        return

    print("Camera connected successfully.")

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    csv_file = open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(csv_file)

    if csv_path.stat().st_size == 0:

        csv_writer.writerow([
            "timestamp",
            "frame_number",
            "selected_label",
            "ear_estimated_state",
            "left_ear",
            "right_ear",
            "average_ear",
            "left_image",
            "right_image"
        ])

        csv_file.flush()

    # -----------------------------------------------------
    # COLLECTION VARIABLES
    # -----------------------------------------------------

    active_label = None

    last_label_change = 0
    last_save_time = 0

    frame_number = 0
    saved_count = 0

    print("\n================================")
    print("SMARTCAP DATASET COLLECTOR")
    print("================================")
    print("O = Select OPEN")
    print("C = Select CLOSED")
    print("SPACE = Pause")
    print("Q = Quit")
    print("================================\n")

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("WARNING: Camera frame not received.")
                time.sleep(0.1)
                continue

            # Correct upside-down camera mounting
            frame = fix_camera_orientation(frame)

            frame_number += 1

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame)
            )

            results = landmarker.detect(mp_image)

            display = frame.copy()

            left_ear = None
            right_ear = None
            average_ear = None
            landmarks = None

            # -------------------------------------------------
            # FACE DETECTION
            # -------------------------------------------------

            if results.face_landmarks:

                landmarks = results.face_landmarks[0]

                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE
                )

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE
                )

                valid_values = [
                    value
                    for value in [left_ear, right_ear]
                    if value is not None
                ]

                if valid_values:

                    average_ear = float(
                        np.mean(valid_values)
                    )

                draw_eye_points(
                    display,
                    landmarks,
                    LEFT_EYE
                )

                draw_eye_points(
                    display,
                    landmarks,
                    RIGHT_EYE
                )

            # -------------------------------------------------
            # EAR ESTIMATION
            # -------------------------------------------------

            estimated_state = estimate_eye_state(
                average_ear
            )

            if average_ear is not None:

                cv2.putText(
                    display,
                    f"EAR: {average_ear:.4f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"EAR estimate: {estimated_state}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # -------------------------------------------------
            # MANUAL LABEL STATUS
            # -------------------------------------------------

            if active_label is None:

                status = "PAUSED - Press O or C"

            else:

                status = (
                    f"SELECTED LABEL: {active_label.upper()}"
                )

            cv2.putText(
                display,
                status,
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Saved samples: {saved_count}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # -------------------------------------------------
            # LABEL VALIDATION DISPLAY ONLY
            # -------------------------------------------------

            if active_label is not None and estimated_state != "UNKNOWN":

                if active_label.upper() == estimated_state:

                    validation_text = "EAR MATCH"

                    validation_color = (0, 255, 0)

                else:

                    validation_text = "EAR MISMATCH - MANUAL LABEL USED"

                    validation_color = (0, 165, 255)

                cv2.putText(
                    display,
                    validation_text,
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    validation_color,
                    2
                )

            cv2.imshow(
                "SmartCap Dataset Collector",
                display
            )

            # -------------------------------------------------
            # SAVE IMAGES
            # -------------------------------------------------

            now = time.time()

            can_save = (
                active_label in label_directories
                and landmarks is not None
                and average_ear is not None
                and now - last_label_change >= args.skip
                and now - last_save_time >= args.interval
            )

            if can_save:

                left_crop = crop_eye(
                    frame,
                    landmarks,
                    LEFT_EYE
                )

                right_crop = crop_eye(
                    frame,
                    landmarks,
                    RIGHT_EYE
                )

                target_dir = label_directories[active_label]

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                filename_base = (
                    f"{args.subject}_{args.session}_{timestamp}"
                )

                left_filename = ""
                right_filename = ""

                left_saved = False
                right_saved = False

                if left_crop is not None:

                    left_filename = filename_base + "_L.png"

                    left_saved = safe_save_image(
                        left_crop,
                        target_dir / left_filename
                    )

                    if not left_saved:
                        left_filename = ""

                if right_crop is not None:

                    right_filename = filename_base + "_R.png"

                    right_saved = safe_save_image(
                        right_crop,
                        target_dir / right_filename
                    )

                    if not right_saved:
                        right_filename = ""

                # Count and log only if at least one image was saved
                if left_saved or right_saved:

                    csv_writer.writerow([
                        datetime.now().isoformat(),
                        frame_number,
                        active_label,
                        estimated_state,
                        left_ear,
                        right_ear,
                        average_ear,
                        left_filename,
                        right_filename
                    ])

                    csv_file.flush()

                    saved_count += 1
                    last_save_time = time.time()

                    print(
                        f"SAVED: {active_label.upper()} | "
                        f"EAR: {average_ear:.4f} | "
                        f"FOLDER: {target_dir.name} | "
                        f"LEFT: {left_saved} | "
                        f"RIGHT: {right_saved} | "
                        f"TOTAL: {saved_count}"
                    )

                else:

                    print("WARNING: No eye image was saved.")

            # -------------------------------------------------
            # KEYBOARD CONTROLS
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("o"), ord("O")):

                active_label = "open"
                last_label_change = time.time()

                print("\nSelected OPEN.")
                print("Images will be saved ONLY in:", open_dir)

            elif key in (ord("c"), ord("C")):

                active_label = "closed"
                last_label_change = time.time()

                print("\nSelected CLOSED.")
                print("Images will be saved ONLY in:", closed_dir)

            elif key == ord(" "):

                active_label = None

                print("\nCollection paused.")

            elif key in (ord("q"), ord("Q")):

                print("\nStopping collection...")
                break

    except KeyboardInterrupt:

        print("\nInterrupted by user.")

    finally:

        cap.release()
        landmarker.close()
        csv_file.close()
        cv2.destroyAllWindows()

        print("\n================================")
        print("COLLECTION FINISHED")
        print("Total saved samples:", saved_count)
        print("Dataset folder:", session_dir)
        print("================================")


if __name__ == "__main__":
    main()
