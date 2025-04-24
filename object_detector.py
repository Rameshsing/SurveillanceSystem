import cv2

class ObjectDetector:
    def __init__(self):
        proto = "models/MobileNetSSD_deploy.prototxt"
        weights = "models/MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", 
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
                        "dog", "horse", "motorbike", "person", "pottedplant", 
                        "sheep", "sofa", "train", "tvmonitor", "bag"]

    def detect_objects(self, frame, conf_thresh=0.6):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if confidence > conf_thresh:
                if self.classes[class_id] in ["bag", "backpack", "suitcase"]:  # Add more if needed
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    results.append(((x1, y1, x2, y2), self.classes[class_id], confidence))
        return results
