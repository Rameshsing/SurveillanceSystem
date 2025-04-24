import cv2
import eventlet
from main import process_frame
from flask import Flask, Response, render_template, request, jsonify
from flask_socketio import SocketIO
# access via http://localhost:5000/video

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app)
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

@app.route('/')
def index():
    return render_template('index.html')

def send_real_time_data():
    import time
    import random
    while True:
        data = {'value': random.randint(0, 100)}
        socketio.emit('update', data)
        time.sleep(5)

if __name__ == '__main__':
    import threading
    thread = threading.Thread(target=send_real_time_data)
    thread.daemon = True
    thread.start()
    socketio.run(app, debug=True)

@app.route('/api/alert', methods=['POST'])
def receive_alert():
    alert_data = request.json
    # Process the alert_data as needed
    print(f"Received alert: {alert_data}")
    return jsonify({'status': 'Alert received'}), 200