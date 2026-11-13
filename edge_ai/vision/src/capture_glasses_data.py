from pathlib import Path
import time
import cv2

NO_GLASSES_DIR = Path(r"C:\AI\camera_glasses_data\no_glasses")
GLASSES_DIR = Path(r"C:\AI\camera_glasses_data\glasses")

NO_GLASSES_DIR.mkdir(parents=True, exist_ok=True)
GLASSES_DIR.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("مش قادر أفتح الكاميرا. اقفلي البرامج اللي بتستخدمها وجربي تاني.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

no_glasses_count = len(list(NO_GLASSES_DIR.glob("*.jpg")))
glasses_count = len(list(GLASSES_DIR.glob("*.jpg")))

print("الكاميرا اشتغلت.")
print("في نافذة الكاميرا: 1 = من غير نظارة | 2 = بنظارة | Q = إنهاء")
print("اضغطي رقم التصنيف بعد ما تخلي وضع وجهك واضحًا ومختلفًا قليلًا عن الصورة السابقة.")

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("ما قدرتش أقرأ صورة من الكاميرا.")
            break

        preview = frame.copy()

        cv2.putText(
            preview,
            f"No glasses: {no_glasses_count}   Glasses: {glasses_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "1 = No glasses | 2 = Glasses | Q = Quit",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Collect Glasses Training Photos", preview)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("1"), ord("2")):
            timestamp = time.time_ns()

            if key == ord("1"):
                image_path = NO_GLASSES_DIR / f"no_glasses_{timestamp}.jpg"
                label = "no_glasses"
            else:
                image_path = GLASSES_DIR / f"glasses_{timestamp}.jpg"
                label = "glasses"

            if cv2.imwrite(str(image_path), frame):
                if label == "no_glasses":
                    no_glasses_count += 1
                else:
                    glasses_count += 1

                print(f"Saved {label}: {image_path.name}")

            # يمنع حفظ لقطات متعددة بالخطأ من ضغطة طويلة
            cv2.waitKey(300)

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nانتهى التصوير.")
print("من غير نظارة:", no_glasses_count)
print("بنظارة:", glasses_count)