
import cv2
import time

CAMERA_URL = "http://192.168.4.1/stream"

print("[INFO] Connecting to ESP32-S3 CAM...")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("[ERROR] Could not connect to camera.")
    print("[CHECK] Make sure your PC is connected to SmartCap-CAM.")
    raise SystemExit(1)

print("[SUCCESS] Camera connected!")
print("[INFO] Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[WARNING] Failed to read frame.")
        time.sleep(0.1)
        continue

    cv2.putText(
        frame,
        "ESP32-S3 CAM - LIVE",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("SmartCap - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Camera test stopped.")
