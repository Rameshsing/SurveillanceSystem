from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import cv2
import numpy as np


class ObjectDetector:
    """
    MobileNet-SSD (Caffe) object detector using OpenCV DNN.
    Expects:
      - MobileNetSSD_deploy.prototxt
      - MobileNetSSD_deploy.caffemodel
    in a 'models' directory adjacent to this file by default.
    """

    # Default 21-label set for the common MobileNet-SSD Caffe (PASCAL VOC) deploy
    # Source: PyImageSearch and MobileNet-SSD Caffe references
    DEFAULT_LABELS = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        prototxt_name: str = "MobileNetSSD_deploy.prototxt",
        caffemodel_name: str = "MobileNetSSD_deploy.caffemodel",
        labels: Optional[List[str]] = None,
        backend: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target: int = cv2.dnn.DNN_TARGET_CPU,
    ):
        """
        Initialize detector, loading the Caffe network with absolute paths.
        """
        # Resolve models directory relative to this file if not provided
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"

        proto_path = (models_dir / prototxt_name).resolve()
        weights_path = (models_dir / caffemodel_name).resolve()

        if not proto_path.exists():
            raise FileNotFoundError(
                f"Missing prototxt at: {proto_path}. Ensure the file exists "
                f"or update the path/filename."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing caffemodel at: {weights_path}. Ensure the file exists "
                f"or update the path/filename."
            )

        # Load network
        self.net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))

        # Optional: set backend/target (CPU by default)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)

        # Labels: default VOC set unless a custom list is supplied
        self.labels = labels if labels is not None else list(self.DEFAULT_LABELS)

    def detect_objects(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.6,
        filter_labels: Optional[Iterable[str]] = None,
    ) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Run inference on a BGR frame and return list of (box, label, confidence).
        Box is (x1, y1, x2, y2) in pixel coords.
        """
        (h, w) = frame.shape[:2]

        # Preprocessing per MobileNet-SSD (Caffe): size (300,300), mean 127.5, scale 1/127.5
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,                # 1/127.5
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        detections = self.net.forward()  # shape: [1, 1, N, 7]

        results: List[Tuple[Tuple[int, int, int, int], str, float]] = []

        if detections.ndim != 4 or detections.shape[21] < 7:
            return results  # unexpected output, return empty

        for i in range(detections.shape[22]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < conf_thresh:
                continue

            class_id = int(detections[0, 0, i, 1])

            # Safety check on label index
            if class_id < 0 or class_id >= len(self.labels):
                continue

            label = self.labels[class_id]

            # Optional filtering by label
            if filter_labels is not None and label not in filter_labels:
                continue

            # Extract and clamp box coordinates
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            results.append(((x1, y1, x2, y2), label, confidence))

        return results

    def detect_only(self, frame: np.ndarray, labels: Iterable[str], conf_thresh: float = 0.6):
        """
        Convenience: detect only the given labels.
        """
        return self.detect_objects(frame, conf_thresh=conf_thresh, filter_labels=labels)
