import numpy as np
from mediapipe.solutions.pose import PoseLandmark

class PostureClassifier:
    def __init__(self):
        self.prev_state = {}

    def classify(self, landmarks, visibility_threshold=0.5):
        if landmarks is None:
            return "Unknown"

        def get_point(lm):
            return np.array([lm.x, lm.y]) if lm.visibility > visibility_threshold else None

        keypoints = {
            "left_shoulder": get_point(landmarks.landmark[PoseLandmark.LEFT_SHOULDER]),
            "right_shoulder": get_point(landmarks.landmark[PoseLandmark.RIGHT_SHOULDER]),
            "left_hip": get_point(landmarks.landmark[PoseLandmark.LEFT_HIP]),
            "right_hip": get_point(landmarks.landmark[PoseLandmark.RIGHT_HIP]),
            "left_knee": get_point(landmarks.landmark[PoseLandmark.LEFT_KNEE]),
            "right_knee": get_point(landmarks.landmark[PoseLandmark.RIGHT_KNEE]),
        }

        if any(v is None for v in keypoints.values()):
            return "Uncertain"

        shoulders_y = np.mean([keypoints["left_shoulder"][1], keypoints["right_shoulder"][1]])
        hips_y = np.mean([keypoints["left_hip"][1], keypoints["right_hip"][1]])
        knees_y = np.mean([keypoints["left_knee"][1], keypoints["right_knee"][1]])

        shoulder_hip_dist = hips_y - shoulders_y
        hip_knee_dist = knees_y - hips_y
        total_height = knees_y - shoulders_y

        if total_height < 0.15:
            return "Lying"
        elif shoulder_hip_dist < 0.1 and hip_knee_dist > 0.1:
            return "Sitting"
        elif shoulder_hip_dist > 0.1 and hip_knee_dist > 0.1:
            return "Standing"
        else:
            return "Unknown"
