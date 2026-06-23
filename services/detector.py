"""
detector.py
YOLOv11 object detection wrapper.
Returns only the target classes defined in config.
"""

from ultralytics import YOLO
import numpy as np
import config


class Detector:
    def __init__(self):
        self.model = YOLO(str(config.YOLO_MODEL))
        self.model.to(config.DEVICE)
        self.target_ids = set(config.TARGET_CLASSES.keys())

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns a list of dicts:
            {
                "bbox":       [x1, y1, x2, y2],   # ints, pixel coords
                "conf":       float,
                "class_id":   int,
                "class_name": str,
            }
        """
        results = self.model.predict(
            source=frame,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=list(self.target_ids),
            device=config.DEVICE,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.target_ids:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area < config.MIN_BOX_AREA:
                continue

            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "conf":       round(float(box.conf[0]), 2),
                "class_id":   class_id,
                "class_name": config.TARGET_CLASSES[class_id],
            })

        return detections
