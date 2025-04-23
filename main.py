import cv2
from detector import PersonDetector
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

cap.release()
cv2.destroyAllWindows()
