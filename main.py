import os
import cv2
import csv
import numpy as np
from detector import PersonDetector, detect_faces
from tracker import CentroidTracker
from line_counter import LineCounter
from pose_utils import PoseDetector
from posture_classifier import PostureClassifier
from datetime import datetime

camera_feeds = {
    "store_front": "data/test_videos/front.mp4",
    "back_exit": "data/test_videos/back.mp4"
}

for camera_id, path in camera_feeds.items():
    cap = cv2.VideoCapture(path)
    detector = PersonDetector()
    tracker = CentroidTracker()
    pose_detector = PoseDetector()
    posture_classifier = PostureClassifier()
    counter = LineCounter(line_position=300)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame.copy()
        boxes = detector.detect(frame)
        tracked = tracker.update(boxes)
        pose_result = pose_detector.detect_pose(frame)
        frame = pose_detector.draw_landmarks(frame, pose_result)
        counter.update(tracked)

        # Draw bounding boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw tracker points
        for (object_id, (cx, cy)) in tracked.items():
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, str(object_id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Face blur
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y:y+h, x:x+w] = face_roi

        # Line + info overlay
        cv2.line(frame, (0, counter.line_y), (frame.shape[1], counter.line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"IN: {counter.count_in} | OUT: {counter.count_out}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {camera_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Pose detection
        pose_result = pose_detector.detect_pose(frame)
        frame = pose_detector.draw_landmarks(frame, pose_result)

        # Posture classification
        if pose_result.pose_landmarks:
            posture = posture_classifier.classify(pose_result.pose_landmarks)
            cv2.putText(frame, f"Posture: {posture}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if posture == "Lying":
            cv2.putText(frame, "ALERT: Possible Fall!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # Line counter
        cv2.line(frame, (0, counter.line_y), (frame.shape[1], counter.line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"IN: {counter.count_in} | OUT: {counter.count_out}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {camera_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow(f"People Flow - {camera_id}", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Generate heatmap after processing
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for (x, y) in tracker.trail_map:
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
            heatmap[y, x] += 5

    heatmap_colored = cv2.applyColorMap(cv2.GaussianBlur(heatmap, (25, 25), 0), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_frame, 0.6, heatmap_colored, 0.4, 0)
    cv2.imshow(f"Heatmap - {camera_id}", overlay)

    os.makedirs("logs", exist_ok=True)
    cv2.imwrite(f"logs/heatmap_{camera_id}.jpg", overlay)

    with open(f"logs/traffic_log_{camera_id}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "in", "out", "camera_id"])
        writer.writeheader()
        for entry in counter.history:
            entry["camera_id"] = camera_id
            writer.writerow(entry)

    cap.release()

cv2.destroyAllWindows()
