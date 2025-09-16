from ultralytics import YOLO
from app.models import Detection, BoundingBox
from typing import List
from PIL import Image
import numpy as np

class LogoDetector:
    def __init__(self, weights_path: str = "weights/best.pt", conf: float = 0.25, device: str = "cpu"):
        """
        Инициализация детектора.
        weights_path: путь до best.pt
        conf: порог confidence для детекции
        """
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path)
        self.conf = conf

    def detect(self, image: Image.Image):
        results = self.model.predict(source=image, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = map(int, xyxy)
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max
                        )
                    )
                )
        return detections
