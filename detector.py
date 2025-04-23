from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        boxes = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls_id = r
            if int(cls_id) == 0 and score > 0.5:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return boxes
