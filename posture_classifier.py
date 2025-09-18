import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Tuple

# MediaPipe Pose landmarks enum
PoseLandmark = mp.solutions.pose.PoseLandmark

class PostureClassifier:
    def __init__(self):
        self.prev_state = {}

    def classify(self, landmarks, visibility_threshold: float = 0.5) -> str:
        """
        Classify posture using vertical relationships of shoulders, hips, and knees
        from MediaPipe pose landmarks (normalized coordinates). Returns one of:
        'Standing', 'Sitting', 'Lying', 'Unknown', or 'Uncertain'.
        """
        if landmarks is None:
            return "Unknown"

        def get_point(lm):
            return np.array([lm.x, lm.y], dtype=np.float32) if lm.visibility > visibility_threshold else None

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

        # Use y at index 1 (not 22)
        shoulders_y = float(np.mean([keypoints["left_shoulder"][2], keypoints["right_shoulder"][2]]))
        hips_y = float(np.mean([keypoints["left_hip"][2], keypoints["right_hip"][2]]))
        knees_y = float(np.mean([keypoints["left_knee"][2], keypoints["right_knee"][2]]))

        shoulder_hip_dist = hips_y - shoulders_y
        hip_knee_dist = knees_y - hips_y
        total_height = knees_y - shoulders_y

        # Heuristics on normalized y-distances (tune as needed)
        if total_height < 0.15:
            return "Lying"
        elif shoulder_hip_dist < 0.1 and hip_knee_dist > 0.1:
            return "Sitting"
        elif shoulder_hip_dist > 0.1 and hip_knee_dist > 0.1:
            return "Standing"
        else:
            return "Unknown"

class DemographicsDetector:
    """
    Age and gender classification using pre-trained Caffe models via OpenCV DNN.
    Expects the following files located in a 'models' directory next to this file:
      - age_deploy.prototxt
      - age_net.caffemodel
      - gender_deploy.prototxt
      - gender_net.caffemodel
    """

    # Standard mean values and input size used for these models
    MODEL_MEAN_VALUES: Tuple[float, float, float] = (78.4263377603, 87.7689143744, 114.895847746)
    INPUT_SIZE: Tuple[int, int] = (227, 227)

    AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    GENDER_LIST = ['Male', 'Female']

    def __init__(self, models_dir: Path | None = None):
        base = Path(__file__).resolve().parent
        models = (base / "models") if models_dir is None else Path(models_dir)

        age_proto = (models / "age_deploy.prototxt").resolve()
        age_weights = (models / "age_net.caffemodel").resolve()
        gender_proto = (models / "gender_deploy.prototxt").resolve()
        gender_weights = (models / "gender_net.caffemodel").resolve()

        for p in (age_proto, age_weights, gender_proto, gender_weights):
            if not p.exists():
                raise FileNotFoundError(f"Missing model file: {p}")

        # Load networks
        self.age_net = cv2.dnn.readNetFromCaffe(str(age_proto), str(age_weights))
        self.gender_net = cv2.dnn.readNetFromCaffe(str(gender_proto), str(gender_weights))

    def detect_age_gender(self, face_bgr: np.ndarray) -> Tuple[str, str]:
        """
        Input: face ROI in BGR color space.
        Returns: (age_bucket, gender_label)
        """
        # Ensure BGR input; resize to required input size
        face_resized = cv2.resize(face_bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # Create blob with documented mean and size; keep BGR channel order
        blob = cv2.dnn.blobFromImage(
            face_resized,
            scalefactor=1.0,
            size=self.INPUT_SIZE,
            mean=self.MODEL_MEAN_VALUES,
            swapRB=False,
            crop=False,
        )

        # Gender prediction
        self.gender_net.setInput(blob)
        g_pred = self.gender_net.forward()
        gender = self.GENDER_LIST[int(np.argmax(g_pred))]

        # Age prediction
        self.age_net.setInput(blob)
        a_pred = self.age_net.forward()
        age = self.AGE_BUCKETS[int(np.argmax(a_pred))]

        return age, gender
