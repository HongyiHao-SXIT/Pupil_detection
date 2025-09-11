from ultralytics import YOLO
import numpy as np

class PupilDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_pupil_center(self, frame):
        results = self.model.predict(source=frame, verbose=False, conf=0.3)
        if len(results) == 0 or results[0].boxes is None:
            return 0.5, 0.5
        boxes = results[0].boxes
        if boxes.conf is None or len(boxes.conf) == 0:
            return 0.5, 0.5

        for i, conf in enumerate(boxes.conf.cpu().numpy()):
            if conf >= 0.80:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                h, w, _ = frame.shape
                return center_x / w, center_y / h

        return 0.5, 0.5

