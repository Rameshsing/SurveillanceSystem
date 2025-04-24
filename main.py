import os
import cv2
import csv
import numpy as np
from datetime import datetime

from detector import PersonDetector, detect_faces
from tracker import CentroidTracker
from line_counter import LineCounter
from pose_utils import PoseDetector
from posture_classifier import PostureClassifier
from object_detector import ObjectDetector
from alerts import send_email_alert, send_whatsapp_alert

# Initialize reusable components
detector = PersonDetector()
tracker = CentroidTracker()
pose_detector = PoseDetector()
posture_classifier = PostureClassifier()
object_detector = ObjectDetector()
counter = LineCounter(line_position=300)
abandoned_objects = {}


camera_feeds = {
    "store_front": "data/test_videos/front.mp4",
    "back_exit": "data/test_videos/back.mp4"
}

def process_camera(camera_id, path, user_email="recipient@example.com"):
    global abandoned_objects
    zone_counts = np.zeros(zone_grid, dtype=np.int32)
    cap = cv2.VideoCapture(path)

    detector = PersonDetector()
    tracker = CentroidTracker()
    pose_detector = PoseDetector()
    object_detector = ObjectDetector()
    posture_classifier = PostureClassifier()
    counter = LineCounter(line_position=300)
    abandoned_objects = {}

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
        frame_height, frame_width = frame.shape[:2]
        zone_height = frame_height // zone_grid[0]
        zone_width = frame_width // zone_grid[1]

        # Track Person Position in Zones
        for (object_id, (cx, cy)) in tracked.items():
            zone_y = min(cy // zone_height, zone_grid[0] - 1)
            zone_x = min(cx // zone_width, zone_grid[1] - 1)
            zone_counts[zone_y, zone_x] += 1

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

    # Heatmap generation
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for (x, y) in tracker.trail_map:
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
            heatmap[y, x] += 5

    heatmap_colored = cv2.applyColorMap(cv2.GaussianBlur(heatmap, (25, 25), 0), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_frame, 0.6, heatmap_colored, 0.4, 0)
    cv2.imshow(f"Heatmap - {camera_id}", overlay)

    # Zone heatmap generation
    norm_zone = cv2.normalize(zone_counts, None, 0, 255, cv2.NORM_MINMAX)
    zone_heat = cv2.resize(norm_zone.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    zone_heat = cv2.resize(zone_heat, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    zone_heat_colored = cv2.applyColorMap(zone_heat, cv2.COLORMAP_JET)
    
    # Overlay zone heatmap on original frame
    os.makedirs("logs", exist_ok=True)
    cv2.imwrite(f"logs/zone_heatmap_{camera_id}.jpg", zone_heat_colored)
    cv2.imwrite(f"logs/heatmap_{camera_id}.jpg", overlay)
    
    # Save traffic log
    with open(f"logs/traffic_log_{camera_id}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "in", "out", "camera_id", "posture", "alert"])
        writer.writeheader()
        for entry in counter.history:
            entry["camera_id"] = camera_id
            writer.writerow(entry)
    
    # zone counts
    with open(f"logs/zone_counts_{camera_id}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Zone(Row,Col)", "Count"])
        for row in range(zone_grid[0]):
            for col in range(zone_grid[1]):
                writer.writerow([(row, col), zone_counts[row, col]])


    cap.release()

def main():
    for camera_id, path in camera_feeds.items():
        process_camera(camera_id, path)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
