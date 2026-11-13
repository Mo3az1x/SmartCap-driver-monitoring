import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# مجلد النماذج
MODEL_DIR = Path(r"C:\AI\smartcap_models")

STATE_MODEL_PATH = MODEL_DIR / "smartcap_model.keras"
STATE_CLASSES_PATH = MODEL_DIR / "class_names.json"

GLASSES_MODEL_PATH = MODEL_DIR / "glasses_model.keras"
GLASSES_CLASSES_PATH = MODEL_DIR / "glasses_classes.json"

CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_required_files():
    required_files = [
        STATE_MODEL_PATH,
        STATE_CLASSES_PATH,
        GLASSES_MODEL_PATH,
        GLASSES_CLASSES_PATH,
        CASCADE_PATH,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Missing required file(s):\n" + "\n".join(missing_files)
        )


def open_webcam():
    # جرّبي الكاميرا رقم 0 ثم رقم 1
    for camera_index in (0, 1):
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return camera

        camera.release()

    raise RuntimeError(
        "Could not open webcam. Close apps using the camera and try again."
    )


def main():
    check_required_files()

    print("Loading models...")
    state_classes = load_json(STATE_CLASSES_PATH)
    glasses_classes = load_json(GLASSES_CLASSES_PATH)

    state_model = tf.keras.models.load_model(STATE_MODEL_PATH)
    glasses_model = tf.keras.models.load_model(GLASSES_MODEL_PATH)

    # نقرأ ملف Haar cascade من المسار الذي حفظناه يدويًا
    face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if face_detector.empty():
        raise RuntimeError(
            "OpenCV could not read the face detector file:\n"
            f"{CASCADE_PATH}"
        )

    camera = open_webcam()

    state_history = deque(maxlen=5)
    glasses_history = deque(maxlen=5)

    current_state = "Waiting..."
    current_glasses = "Waiting..."
    state_confidence = 0.0
    glasses_confidence = 0.0
    last_prediction_time = 0.0

    print("Webcam started. Press Q or Esc in the camera window to close.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            if len(faces) == 0:
                current_state = "Face not detected"
                current_glasses = "Waiting..."
                state_confidence = 0.0
                glasses_confidence = 0.0
                state_history.clear()
                glasses_history.clear()

            else:
                # استخدام أكبر وجه ظاهر
                x, y, w, h = max(
                    faces,
                    key=lambda face: face[2] * face[3],
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 220, 255),
                    2,
                )

                now = time.time()

                # تحديث التوقعات مرتين تقريبًا في الثانية
                if now - last_prediction_time >= 0.5:
                    face_crop = frame[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2RGB,
                    )
                    face_image = cv2.resize(face_rgb, (224, 224))
                    face_batch = np.expand_dims(
                        face_image.astype(np.float32),
                        axis=0,
                    )

                    # توقع حالة السائق من صورة الوجه
                    state_probabilities = state_model.predict(
                        face_batch,
                        verbose=0,
                    )[0]

                    state_history.append(state_probabilities)
                    average_state = np.mean(state_history, axis=0)

                    state_index = int(np.argmax(average_state))
                    current_state = state_classes[state_index]
                    state_confidence = float(
                        average_state[state_index]
                    )

                    # موديل النظارة تدرب على إطار الفيديو الكامل
                    full_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )
                    full_image = cv2.resize(full_rgb, (224, 224))
                    full_batch = np.expand_dims(
                        full_image.astype(np.float32),
                        axis=0,
                    )

                    # خرج Sigmoid يمثل احتمال الفئة الثانية
                    no_glasses_probability = float(
                        glasses_model.predict(
                            full_batch,
                            verbose=0,
                        )[0][0]
                    )

                    glasses_probabilities = np.array([
                        1.0 - no_glasses_probability,
                        no_glasses_probability,
                    ])

                    glasses_history.append(glasses_probabilities)
                    average_glasses = np.mean(
                        glasses_history,
                        axis=0,
                    )

                    glasses_index = int(np.argmax(average_glasses))
                    current_glasses = glasses_classes[glasses_index]
                    glasses_confidence = float(
                        average_glasses[glasses_index]
                    )

                    last_prediction_time = now

            # عرض النتائج
            cv2.putText(
                frame,
                f"Driver state: {current_state} "
                f"({state_confidence * 100:.0f}%)",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Eyewear: {current_glasses} "
                f"({glasses_confidence * 100:.0f}%)",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )

            cv2.putText(
                frame,
                "Press Q or Esc to quit",
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "SmartCap - Driver State and Glasses",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Webcam closed.")


if __name__ == "__main__":
    main()