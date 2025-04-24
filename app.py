from flask import Flask, Response
import cv2
from main import process_frame
# access via http://localhost:5000/video

app = Flask(__name__)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    processed = process_frame(frame, camera_id="live_cam", user_email="alif.rahman.c@gmail.com")

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame, camera_id="live_cam", user_email="alif.rahman.c@gmail.com")
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)
