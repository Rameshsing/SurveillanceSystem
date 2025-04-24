import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseDetector:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(static_image_mode=static_image_mode,
                                 min_detection_confidence=min_detection_confidence,
                                 min_tracking_confidence=min_tracking_confidence)

    def detect_pose(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        return result

    def draw_landmarks(self, frame, result):
        if result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        return frame
