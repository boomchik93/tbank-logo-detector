from ultralytics import YOLO
from models import Detection, BoundingBox
from typing import List
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LogoDetector:
    def __init__(self, weights_path: str = "weights/best.pt", conf: float = 0.25, device: str = "cpu"):
        """
        Инициализация детектора.
        weights_path: путь до best.pt
        conf: порог confidence для детекции
        """
        self.weights_path = weights_path
        self.conf = conf

        logger.info("Загрузка модели YOLO с весами: %s", self.weights_path)
        try:
            self.model = YOLO(self.weights_path)
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.exception("Ошибка при загрузке модели YOLO: %s", str(e))
            raise

    def detect(self, image: Image.Image):
        logger.info("Запуск детекции (conf=%.2f)", self.conf)
        detections = []

        try:
            results = self.model.predict(source=image, conf=self.conf, verbose=False)
            logger.debug("YOLO вернул %d результатов", len(results))
        except Exception as e:
            logger.exception("Ошибка во время предсказания YOLO: %s", str(e))
            return detections

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                logger.debug("Результат без boxes, пропуск...")
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
        logger.info("Детекция завершена: найдено объектов %d", len(detections))
        return detections
