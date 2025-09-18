# core_processing.py
import sys
import cv2
import numpy as np
from datetime import datetime

from detector import PersonDetector, detect_faces
from tracker import CentroidTracker
from line_counter import LineCounter
from pose_utils import PoseDetector
from posture_classifier import PostureClassifier, DemographicsDetector
from object_detector import ObjectDetector
from loitering_detector import LoiteringDetector
from detectors.zone_intrusion import ZoneIntrusionDetector
from db import init_db, insert_log  # logging to SQLite

# Toggle this to show OpenCV windows (requires GUI-enabled OpenCV)
SHOW_WINDOWS = True

class SurveillanceProcessor:
    """
    Reusable, per-process pipeline that:
      - opens a webcam or file
      - processes frames with detectors
      - logs periodic rows to SQLite for the dashboard
    """

    def __init__(self):
        # Reuse heavy components for performance
        self.detector = PersonDetector()
        self.tracker = CentroidTracker()
        self.pose_detector = PoseDetector()
        self.object_detector = ObjectDetector()
        self.zone_detector = ZoneIntrusionDetector()
        self.posture_classifier = PostureClassifier()
        self.demographics_detector = DemographicsDetector()
        self.loitering_detector = LoiteringDetector()
        self.counter = LineCounter(line_position=300)

        # Shared DB connection for inserts
        self.conn = init_db()

    def _process_single_frame(self, frame, camera_id="live_cam"):
        """
        Process one frame and return: processed_frame, alerts_text, posture_label.
        """
        alert_text = ""

        # People detection, tracking, counting
        boxes = self.detector.detect(frame)
        tracked = self.tracker.update(boxes)
        self.counter.update(tracked)

        # Draw detections and track IDs
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (object_id, (cx, cy)) in tracked.items():
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, str(object_id), (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Face blur
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (99, 99), 30)

        # Pose and posture
        pose_result = self.pose_detector.detect_pose(frame)
        frame = self.pose_detector.draw_landmarks(frame, pose_result)
        posture = "Unknown"
        if getattr(pose_result, "pose_landmarks", None):
            posture = self.posture_classifier.classify(pose_result.pose_landmarks)

        # Alerts
        if posture == "Lying":
            cv2.putText(frame, "ALERT: Possible Fall!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            alert_text += "Fall "
        if len(tracked) > 10:
            cv2.putText(frame, "⚠️ Crowd Alert!", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            alert_text += "Crowd "

        # HUD
        cv2.putText(frame, f"Camera: {camera_id}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Posture: {posture}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame, alert_text.strip(), posture

    def process_camera_stream(self, camera_id, video_source, user_email="recipient@example.com"):
        """
        Open a webcam (index like 0 or 1) or a file path and process frames in a loop.
        On Windows, try Media Foundation (MSMF) then DirectShow (DSHOW) for reliability.
        """
        # Prefer Windows backends when available for camera indices
        backend = cv2.CAP_MSMF if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(video_source, backend)

        # Fallback to DirectShow on Windows if MSMF fails
        if not cap.isOpened() and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)

        print(f"[DEBUG] Opening source for {camera_id}: {video_source} (backend={backend})")
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source for {camera_id}: {video_source}")
            return

        # Optional capture properties (may be ignored by drivers)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] {camera_id}: read() returned False after {frames} frames; stopping.")
                break

            frames += 1
            processed_frame, alerts, posture = self._process_single_frame(frame, camera_id)

            # Periodic logging to SQLite (about once per ~30 frames)
            if frames % 30 == 0:
                insert_log(self.conn, {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camera_id": camera_id,
                    "in": self.counter.count_in,
                    "out": self.counter.count_out,
                    "posture": posture,
                    "alert": alerts
                })

            if SHOW_WINDOWS:
                cv2.imshow(f"{camera_id}", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if alerts:
                # Import here to avoid circular import at module load
                from alert_service import send_surveillance_alert
                send_surveillance_alert(alerts, camera_id)

        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Safe in case HighGUI is unavailable in some environments
            pass
