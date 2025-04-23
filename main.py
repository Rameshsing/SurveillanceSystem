import cv2
import csv
import numpy as np
from detector import PersonDetector
from detector import detect_faces
from tracker import CentroidTracker
from line_counter import LineCounter

cap = cv2.VideoCapture("data/test_videos/sample.mp4")
detector = PersonDetector()
tracker = CentroidTracker()
counter = LineCounter(line_position=300)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detector.detect(frame)
    tracked = tracker.update(boxes)
    counter.update(tracked)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for (object_id, (cx, cy)) in tracked.items():
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(frame, str(object_id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.line(frame, (0, counter.line_y), (frame.shape[1], counter.line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"IN: {counter.count_in} | OUT: {counter.count_out}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("People Flow Analytics", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[y:y+h, x:x+w] = face_roi

heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
for (x, y) in tracker.trail_map:
    if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
        heatmap[y, x] += 5

heatmap_colored = cv2.applyColorMap(cv2.GaussianBlur(heatmap, (25, 25), 0), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
cv2.imshow("Heatmap", overlay)

with open("traffic_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["time", "in", "out"])
    writer.writeheader()
    for entry in counter.history:
        writer.writerow(entry)

cap.release()
cv2.imwrite("heatmap_output.jpg", overlay)
cv2.destroyAllWindows()
