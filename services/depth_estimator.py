"""
depth_estimator.py
Monocular depth estimation using Depth Anything V2 (small).

For each tracked object, samples the depth map inside its bounding box
and returns an estimated distance in metres.

No LiDAR required — pure monocular camera.
"""

import cv2
import numpy as np
import torch
from transformers import pipeline


class DepthEstimator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("[INFO] Loading Depth Anything V2...")
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if device == "cuda" else -1,
        )
        # Calibration scalar — maps normalised depth to metres
        # Tuned for dashcam footage (tweak if distances look off)
        self.scale = 20.0
        print("[INFO] Depth Anything V2 ready.")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a BGR frame.
        Returns a depth map (H x W float32) where higher = farther.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgb)
        result   = self.pipe(pil_img)
        depth    = np.array(result["depth"], dtype=np.float32)

        # Resize depth map to match frame size
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        return depth

    def get_object_distance(self, depth_map: np.ndarray,
                             bbox: list[int]) -> float:
        """
        Estimate distance (metres) to an object from its bounding box.
        Samples the centre 50% of the box to avoid edge noise.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape

        # Clamp to frame bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return -1.0

        # Sample inner 50% of the box (more stable than full box)
        mx1 = x1 + (x2 - x1) // 4
        mx2 = x2 - (x2 - x1) // 4
        my1 = y1 + (y2 - y1) // 4
        my2 = y2 - (y2 - y1) // 4

        region = depth_map[my1:my2, mx1:mx2]
        if region.size == 0:
            return -1.0

        # Median is more robust than mean for depth
        median_depth = float(np.median(region))

        # Normalise to 0–1 then convert to metres
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max == d_min:
            return -1.0

        normalised = (median_depth - d_min) / (d_max - d_min)

        # Invert: depth models usually output higher = closer
        # Depth Anything V2 outputs higher = farther, so no inversion needed
        distance_m = normalised * self.scale
        return round(distance_m, 1)
