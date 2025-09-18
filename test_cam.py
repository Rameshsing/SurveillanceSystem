# test_cam.py
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # try CAP_MSMF on Windows
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # fallback on Windows
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # generic fallback

print("Opened:", cap.isOpened())
while True:
    ret, frame = cap.read()
    if not ret:
        print("read() failed; exiting.")
        break
    cv2.imshow("webcam 0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
