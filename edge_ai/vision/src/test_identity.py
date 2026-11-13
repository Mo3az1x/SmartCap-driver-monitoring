from pathlib import Path
import cv2
from deepface import DeepFace

MODEL_DIR = Path(r"C:\AI\smartcap_models")
KNOWN_DIR = Path(r"C:\AI\known_faces")
GALLERY_DIR = Path(r"C:\AI\known_faces_cropped")

OMNIA_DIR = KNOWN_DIR / "Omnia"
OMNIA_CROPPED_DIR = GALLERY_DIR / "Omnia"
CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"


def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    # هامش صغير حول الوجه
    pad = int(max(w, h) * 0.15)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    return image[y1:y2, x1:x2]


if not CASCADE_PATH.exists():
    raise FileNotFoundError(f"Face detector not found:\n{CASCADE_PATH}")

if not OMNIA_DIR.exists():
    raise FileNotFoundError(f"Omnia reference folder not found:\n{OMNIA_DIR}")

face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))
if face_detector.empty():
    raise RuntimeError(f"Could not load face detector:\n{CASCADE_PATH}")

OMNIA_CROPPED_DIR.mkdir(parents=True, exist_ok=True)

# تجهيز صور أُمنية المرجعية
prepared = 0

for image_path in OMNIA_DIR.iterdir():
    if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        continue

    face = crop_face(image)
    if face is None:
        print("No face found in reference:", image_path.name)
        continue

    output_path = OMNIA_CROPPED_DIR / f"{image_path.stem}.jpg"
    if cv2.imwrite(str(output_path), face):
        prepared += 1

print("Prepared Omnia reference photos:", prepared)

if prepared == 0:
    raise RuntimeError(
        "No reference faces were prepared. Check the photos in the Omnia folder."
    )

# افتحي كاميرا اللابتوب
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

print("Webcam ready. Press 1 to identify, or Q to quit.")

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read webcam frame.")
            break

        preview = frame.copy()
        cv2.putText(
            preview,
            "Press 1 to identify | Q to quit",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("SmartCap - Face Identity Test", preview)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key == ord("1"):
            face = crop_face(frame)

            if face is None:
                print("No face detected. Center your full face in the camera.")
                continue

            try:
                results = DeepFace.find(
                    img_path=face,
                    db_path=str(GALLERY_DIR),
                    model_name="Facenet512",
                    detector_backend="skip",
                    align=False,
                    silent=True,
                    refresh_database=True,
                )

                matches = results[0] if results else None

                if matches is None or matches.empty:
                    print("Result: Unknown")
                else:
                    identity_path = str(matches.iloc[0]["identity"])
                    identity_name = Path(identity_path).parent.name
                    print("Result:", identity_name)

            except Exception as error:
                print("Face recognition error:", error)

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Identity test closed.")