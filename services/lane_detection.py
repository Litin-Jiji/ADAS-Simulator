"""
lane_detection.py
Lane detection using Canny edge detection + Hough Line Transform.

Pipeline per frame:
  1. Convert to grayscale
  2. Gaussian blur (reduce noise)
  3. Canny edge detection
  4. ROI trapezoid mask (focus on road ahead)
  5. Probabilistic Hough Line Transform
  6. Separate lines into left / right by slope
  7. Fit a single averaged line per side
  8. Compute lane centre offset + departure warning
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class LaneResult:
    left_line:    tuple | None   # (x1, y1, x2, y2)
    right_line:   tuple | None
    offset_cm:    float          # + = right of centre, - = left
    status:       str            # "Centered" | "Drifting Left" | "Drifting Right" | "No Lane"
    warning:      bool           # True when departure threshold exceeded


class LaneDetector:
    def __init__(self):
        # Hough parameters
        self.rho          = 1
        self.theta        = np.pi / 180
        self.threshold    = 40
        self.min_line_len = 40
        self.max_line_gap = 150

        # Canny thresholds
        self.canny_low    = 50
        self.canny_high   = 150

        # Lane departure threshold (cm)
        self.departure_threshold = 25

        # Smoothing: keep last N averaged lines
        self._left_hist:  list = []
        self._right_hist: list = []
        self._hist_len    = 8

    # ── ROI mask ───────────────────────────────────────────────────────────

    def _roi_mask(self, edges: np.ndarray) -> np.ndarray:
        h, w = edges.shape
        # Trapezoid focused on road ahead
        # Top of trapezoid starts at 55% height, narrows to centre
        roi_vertices = np.array([[
            (int(w * 0.05), h),           # bottom-left
            (int(w * 0.42), int(h * 0.58)),  # top-left
            (int(w * 0.58), int(h * 0.58)),  # top-right
            (int(w * 0.95), h),           # bottom-right
        ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)

    # ── Line averaging ─────────────────────────────────────────────────────

    def _avg_line(self, frame: np.ndarray, lines: list) -> tuple | None:
        """Fit a single line through all detected line segments on one side."""
        if not lines:
            return None
        h = frame.shape[0]
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        try:
            poly = np.polyfit(y_coords, x_coords, 1)
        except Exception:
            return None

        y1 = h
        y2 = int(h * 0.58)
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))
        return (x1, y1, x2, y2)

    def _smooth(self, history: list, new_line: tuple | None, max_len: int) -> tuple | None:
        """Temporal smoothing — average over last N frames."""
        if new_line:
            history.append(new_line)
        if len(history) > max_len:
            history.pop(0)
        if not history:
            return None
        arr = np.array(history, dtype=np.float32)
        return tuple(arr.mean(axis=0).astype(int))

    # ── Main detect ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]

        # 1. Grayscale + blur
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Canny edges
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 3. ROI
        masked = self._roi_mask(edges)

        # 4. Hough lines
        raw_lines = cv2.HoughLinesP(
            masked,
            self.rho, self.theta, self.threshold,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap,
        )

        left_lines, right_lines = [], []

        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                # Filter near-horizontal lines (noise)
                if abs(slope) < 0.4:
                    continue
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

        # 5. Average + smooth
        left  = self._smooth(self._left_hist,  self._avg_line(frame, left_lines),  self._hist_len)
        right = self._smooth(self._right_hist, self._avg_line(frame, right_lines), self._hist_len)

        # 6. Compute lane offset
        offset_cm = 0.0
        status    = "No Lane"
        warning   = False

        if left and right:
            # Bottom x of each line = where they hit the bottom of frame
            lane_centre = (left[0] + right[0]) / 2
            frame_centre = w / 2
            # Rough calibration: 1 pixel ≈ 0.3 cm at dashcam distance
            offset_cm = round((lane_centre - frame_centre) * 0.3, 1)

            if abs(offset_cm) < self.departure_threshold:
                status = "Centered"
            elif offset_cm < 0:
                status  = "Drifting Left"
                warning = True
            else:
                status  = "Drifting Right"
                warning = True

        elif left:
            status = "Drifting Right"
            warning = True
        elif right:
            status = "Drifting Left"
            warning = True

        return LaneResult(
            left_line=left,
            right_line=right,
            offset_cm=offset_cm,
            status=status,
            warning=warning,
        )
