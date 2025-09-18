# app.py
import cv2
import eventlet
from flask import Flask, Response, render_template, request, jsonify
from flask_socketio import SocketIO
from core_processing import process_frame  # Import from core_processing instead of main
from alert_service import send_alert  # Import alert function from proper module
from loitering_detector import LoiteringDetector
from alerts import send_advanced_alert
from reporting import generate_report

app = Flask(__name__)
socketio = SocketIO(app)
eventlet.monkey_patch()

loitering_detector = LoiteringDetector()
cap = cv2.VideoCapture(0)

def gen_frames():
    """Generate video frames for web streaming"""
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Use the refactored process_frame function
        processed_frame, alerts = process_frame(frame, camera_id="live_cam", 
                                              user_email="alif.rahman.c@gmail.com")
        
        # Send alerts if any detected
        if alerts:
            send_alert(alert_type="live_detection", details=alerts, camera_id="live_cam")
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/alert', methods=['POST'])
def receive_alert():
    alert_data = request.json
    print(f"Received alert: {alert_data}")
    return jsonify({'status': 'Alert received'}), 200

@app.route('/api/alerts', methods=['POST'])
def api_send_alert():  # Renamed to avoid conflict
    alert_type = request.json.get('alert_type', 'general')
    details = request.json.get('details', '')
    camera_id = request.json.get('camera_id', 'unknown')
    
    send_alert(alert_type=alert_type, details=details, camera_id=camera_id)
    return jsonify({"status": "alert sent"})

@app.route('/api/reports', methods=['GET'])
def get_report():
    data = {"columns": ["time", "in", "out"], "rows": [["2025-04-25 12:00:00", 5, 2]]}
    report_file = generate_report(data, "traffic_report.csv")
    return jsonify({"status": "report generated", "file": report_file})

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
