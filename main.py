import os
import cv2
import csv
import numpy as np
from datetime import datetime
from db import init_db, insert_log
from app import send_alert
from dashboard import app as dashboard_app

from detector import PersonDetector, detect_faces
from tracker import CentroidTracker
from line_counter import LineCounter
from pose_utils import PoseDetector
from posture_classifier import PostureClassifier
from object_detector import ObjectDetector
from loitering_detector import LoiteringDetector
from alerts import send_email_alert, send_whatsapp_alert
from posture_classifier import DemographicsDetector
from detectors.zone_intrusion import ZoneIntrusionDetector

camera_feeds = {
    "store_front": "data/test_videos/front.mp4",
    "back_exit": "data/test_videos/back.mp4"
}

def process_camera(camera_id, path, user_email="recipient@example.com"):
    abandoned_objects = {}  # Moved to local variable inside function

    cap = cv2.VideoCapture(path)

    detector = PersonDetector()
    tracker = CentroidTracker()
    pose_detector = PoseDetector()
    object_detector = ObjectDetector()
    zone_detector = ZoneIntrusionDetector()
    posture_classifier = PostureClassifier()
    demographics_detector = DemographicsDetector()
    loitering_detector = LoiteringDetector()
    counter = LineCounter(line_position=300)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame.copy()
        alert_text = ""

        # People detection and tracking
        boxes = detector.detect(frame)
        tracked = tracker.update(boxes)
        counter.update(tracked)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for (object_id, (cx, cy)) in tracked.items():
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, str(object_id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Update loitering detection
        loitering_alerts = loitering_detector.update(tracked)
        
        if loitering_alerts:
            for alert in loitering_alerts:
                send_alert(alert_type="loitering", details=alert)

        cv2.imshow(f"Camera {camera_id}", frame)

        # Inactivity check
        inactive_objects = []
        for obj_id, points in tracker.object_history.items():
            if len(points) >= 15:
                xs, ys = zip(*points)
                if max(xs) - min(xs) < 10 and max(ys) - min(ys) < 10:
                    inactive_objects.append(obj_id)

        # Face blur
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y:y + h, x:x + w] = face_roi

        # Pose detection
        pose_result = pose_detector.detect_pose(frame)
        frame = pose_detector.draw_landmarks(frame, pose_result)
        posture = posture_classifier.classify(pose_result.pose_landmarks) if pose_result.pose_landmarks else "Unknown"

        # Posture alert
        if posture == "Lying":
            cv2.putText(frame, "ALERT: Possible Fall!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            alert_text += "Fall "

        # Crowd alert
        if len(tracked) > 10:
            cv2.putText(frame, "⚠️ Crowd Alert!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            alert_text += "Crowd "

        # Inactivity alert
        if any(obj_id in inactive_objects for obj_id in tracked.keys()):
            alert_text += "Inactivity "

        # Zone counting
        zone_grid = (4, 4)  # 4x4 grid
        zone_counts = np.zeros(zone_grid, dtype=np.int32)

        frame_height, frame_width = frame.shape[:2]
        zone_height = frame_height // zone_grid[0]
        zone_width = frame_width // zone_grid[1]

        # Track Person Position in Zones
        for (object_id, (cx, cy)) in tracked.items():
            zone_y = min(cy // zone_height, zone_grid[0] - 1)
            zone_x = min(cx // zone_width, zone_grid[1] - 1)
            zone_counts[zone_y, zone_x] += 1
            
            # Crop the face or upper body of the detected person
            face_roi = frame[cy-50:cy+50, cx-50:cx+50]

            # Get Age and Gender predictions
            age, gender = demographics_detector.detect_age_gender(face_roi)
            cv2.putText(frame, f"Age: {age}, Gender: {gender}", (cx, cy - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Line and status info
        cv2.line(frame, (0, counter.line_y), (frame.shape[1], counter.line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"IN: {counter.count_in} | OUT: {counter.count_out}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {camera_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Posture: {posture}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Object detection and abandoned object check
        object_boxes = object_detector.detect_objects(frame)
        object_tracked = tracker.update([box for (box, _, _) in object_boxes])

        for obj_id, (cx, cy) in object_tracked.items():
            if obj_id in abandoned_objects:
                abandoned_objects[obj_id]["frames"] += 1
            else:
                abandoned_objects[obj_id] = {"centroid": (cx, cy), "frames": 1}

            if abandoned_objects[obj_id]["frames"] > 150:
                alert_text += "Abandoned Object "
                cv2.putText(frame, "⚠️ ABANDONED OBJECT", (cx, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw zones and detect intrusions
        frame = zone_detector.draw_zones(frame, camera_id)
        tracked_wrapped = {oid: {"centroid": pos} for oid, pos in tracked.items()}
        intrusions = zone_detector.detect_intrusions(camera_id, tracked_wrapped)

        for intrusion in intrusions:
            object_id = intrusion["object_id"]
            zone_id = intrusion["zone_id"]
            centroid = intrusion["centroid"]

            cv2.putText(frame, f"INTRUSION: {zone_id}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"[ALERT] Object {object_id} entered restricted zone {zone_id}")
            alert_text += f"Intrusion({zone_id}) "
        
        # Loitering detection
        loitering_time = 150  # frames (~5 seconds if 30 FPS)

        for obj_id, points in tracker.object_history.items():
            if len(points) >= loitering_time:
                xs, ys = zip(*points[-loitering_time:])
                if max(xs) - min(xs) < 15 and max(ys) - min(ys) < 15:
                    alert_text += f"Loitering({obj_id}) "
                    cx, cy = points[-1]
                    cv2.putText(frame, "⚠️ Loitering Detected", (cx, cy + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Logging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        counter.history.append({
            "time": timestamp,
            "in": counter.count_in,
            "out": counter.count_out,
            "posture": posture,
            "alert": alert_text.strip(),
            "camera_id": camera_id
        })

        if alert_text.strip():
            send_email_alert("⚠️ Camera Alert", f"{alert_text.strip()} @ {timestamp}")
            send_whatsapp_alert(f"⚠️ {alert_text.strip()} @ {timestamp}")
        
        # Zone heatmap overlay
        zone_overlay = frame.copy()
        max_count = zone_counts.max() or 1

        for row in range(zone_grid[0]):
            for col in range(zone_grid[1]):
                x1, y1 = col * zone_width, row * zone_height
                x2, y2 = x1 + zone_width, y1 + zone_height
                alpha = zone_counts[row, col] / max_count
                overlay_color = (0, int(255 * (1 - alpha)), int(255 * alpha))  # blue to red
                cv2.rectangle(zone_overlay, (x1, y1), (x2, y2), overlay_color, -1)

        frame = cv2.addWeighted(zone_overlay, 0.4, frame, 0.6, 0)

        cv2.imshow(f"People Flow - {camera_id}", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Saving Heatmap
    os.makedirs("logs", exist_ok=True)
    conn = init_db()

    # Save traffic log
    with open(f"logs/traffic_log_{camera_id}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "in", "out", "camera_id", "posture", "alert"])
        writer.writeheader()
        for entry in counter.history:
            entry["camera_id"] = camera_id
            writer.writerow(entry)
    
    # Save zone counts
    with open(f"logs/zone_counts_{camera_id}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Zone(Row,Col)", "Count"])
        for row in range(zone_grid[0]):
            for col in range(zone_grid[1]):
                writer.writerow([(row, col), zone_counts[row, col]])

    for entry in counter.history:
        entry["camera_id"] = camera_id
        insert_log(conn, entry)
    conn.close()
    
    cap.release()

def main():
    for camera_id, path in camera_feeds.items():
        process_camera(camera_id, path)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    dashboard_app.run_server(debug=True)
    main()
