"""
tracker.py
ByteTrack multi-object tracker via Ultralytics.

Each tracked object gets a persistent integer ID that survives
occlusion and re-entry for up to TRACK_BUFFER frames.

Output per object:
    {
        "track_id":   int,
        "bbox":       [x1, y1, x2, y2],
        "conf":       float,
        "class_id":   int,
        "class_name": str,
        "centroid":   (cx, cy),
        "track_len":  int,          # number of frames seen
    }
"""

import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import config


class Tracker:
    def __init__(self):
        # We reuse the YOLO model's built-in ByteTrack integration
        self.model = YOLO(str(config.YOLO_MODEL))
        self.model.to(config.DEVICE)

        # History: track_id → list of centroids (for trail drawing)
        self.histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.frame_counts: dict[int, int] = defaultdict(int)

    def track(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection + ByteTrack on a single BGR frame.
        Returns list of tracked object dicts.
        """
        results = self.model.track(
            source=frame,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=list(config.TARGET_CLASSES.keys()),
            tracker="bytetrack.yaml",
            device=config.DEVICE,
            persist=True,       # keeps track state between calls
            verbose=False,
        )[0]

        tracked = []

        if results.boxes is None or results.boxes.id is None:
            return tracked

        for box in results.boxes:
            if box.id is None:
                continue

            track_id  = int(box.id[0])
            class_id  = int(box.cls[0])

            if class_id not in config.TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area < config.MIN_BOX_AREA:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Update history
            self.histories[track_id].append((cx, cy))
            if len(self.histories[track_id]) > 60:   # keep last 60 frames
                self.histories[track_id].pop(0)

            self.frame_counts[track_id] += 1

            tracked.append({
                "track_id":   track_id,
                "bbox":       [x1, y1, x2, y2],
                "conf":       round(float(box.conf[0]), 2),
                "class_id":   class_id,
                "class_name": config.TARGET_CLASSES[class_id],
                "centroid":   (cx, cy),
                "track_len":  self.frame_counts[track_id],
                "history":    list(self.histories[track_id]),
            })

        return tracked

    def reset(self):
        self.histories.clear()
        self.frame_counts.clear()
