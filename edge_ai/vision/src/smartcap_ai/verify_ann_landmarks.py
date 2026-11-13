
import cv2
import mediapipe as mp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR / "input_videos" /
    "SmartCap_20261112_163011" / "smartcap_test.mp4"
)

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

OUTPUT_PATH = BASE_DIR / "output" / "ann_landmarks_check.jpg"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

POINTS = {
    "Eye center 1": 468,
    "Eye center 2": 473,
    "Nose tip": 1,
    "Mouth corner 1": 61,
    "Mouth corner 2": 291,
}

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.35,
    min_face_presence_confidence=0.35,
)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open input video.")

detector = vision.FaceLandmarker.create_from_options(options)

frame_number = 0
found = False

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        height, width = frame.shape[:2]

        for name, index in POINTS.items():
            point = landmarks[index]

            x = int(point.x * width)
            y = int(point.y * height)

            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"{index}: {name}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        cv2.imwrite(str(OUTPUT_PATH), frame)

        print("Face detected successfully!")
        print("Frame number:", frame_number)
        print("Image saved:", OUTPUT_PATH)

        found = True
        break

    frame_number += 1

cap.release()
detector.close()

if not found:
    raise RuntimeError("No face detected in the entire video.")
