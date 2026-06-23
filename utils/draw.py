"""
draw.py
All OpenCV drawing utilities: bounding boxes, labels,
motion trails, stats panel, and the top HUD bar.
"""

import cv2
import numpy as np
import config


# ── Helpers ────────────────────────────────────────────────────────────────

def _alpha_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, alpha: float) -> np.ndarray:
    """Draw a semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _label_above(frame: np.ndarray, text: str, x1: int, y1: int,
                 color: tuple) -> None:
    """Draw a pill-shaped label above a bounding box."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = config.HUD_FONT_SCALE
    thickness  = config.HUD_THICKNESS
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad = 4
    bx1, by1 = x1, max(0, y1 - th - 2 * pad)
    bx2, by2 = x1 + tw + 2 * pad, y1

    # Background pill
    frame[:] = _alpha_rect(frame, bx1, by1, bx2, by2, color, 0.75)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad),
                font, scale, (15, 15, 15), thickness, cv2.LINE_AA)


# ── Public drawing functions ────────────────────────────────────────────────

def draw_tracked_objects(frame: np.ndarray, tracked: list[dict]) -> None:
    """
    Draw bounding box, label, confidence, track ID,
    and motion trail for every tracked object.
    """
    for obj in tracked:
        x1, y1, x2, y2 = obj["bbox"]
        cls   = obj["class_name"]
        tid   = obj["track_id"]
        conf  = obj["conf"]
        color = config.CLASS_COLORS.get(cls, (200, 200, 200))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents (looks more technical)
        corner = 12
        thick  = 3
        for sx, ex, sy, ey in [
            (x1, x1+corner, y1, y1), (x2-corner, x2, y1, y1),
            (x1, x1+corner, y2, y2), (x2-corner, x2, y2, y2),
        ]:
            cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), thick)
        for sx, ex, sy, ey in [
            (x1, x1, y1, y1+corner), (x2, x2, y1, y1+corner),
            (x1, x1, y2-corner, y2), (x2, x2, y2-corner, y2),
        ]:
            cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), thick)

        # Label
        label = f"{cls} #{tid}  {conf:.0%}"
        _label_above(frame, label, x1, y1, color)

        # Motion trail
        history = obj.get("history", [])
        if len(history) > 1:
            for i in range(1, len(history)):
                alpha = i / len(history)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, history[i - 1], history[i],
                         trail_color, 2, cv2.LINE_AA)


def draw_stats_panel(frame: np.ndarray, tracked: list[dict],
                     fps: float, frame_num: int) -> None:
    """
    Bottom-left panel: per-class counts, FPS, frame number.
    """
    h, w = frame.shape[:2]

    # Count per class
    counts: dict[str, int] = {}
    for obj in tracked:
        cls = obj["class_name"]
        counts[cls] = counts.get(cls, 0) + 1

    lines = []
    for cls in ["car", "truck", "bus", "motorcycle", "bicycle", "person"]:
        n = counts.get(cls, 0)
        if n:
            lines.append((cls.capitalize() + "s", str(n),
                          config.CLASS_COLORS.get(cls, (200, 200, 200))))

    lines.append(("FPS", f"{fps:.1f}", (200, 200, 200)))
    lines.append(("Frame", str(frame_num), (160, 160, 160)))

    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.52
    lh     = 22
    pad    = 10
    panel_w = 180
    panel_h = len(lines) * lh + pad * 2

    px, py = 14, h - panel_h - 14
    frame[:] = _alpha_rect(frame, px, py, px + panel_w, py + panel_h,
                           (20, 20, 20), 0.65)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h),
                  (80, 80, 80), 1)

    for i, (key, val, color) in enumerate(lines):
        y = py + pad + i * lh + lh // 2
        cv2.putText(frame, key, (px + 8, y),
                    font, scale, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (px + panel_w - 8, y),
                    font, scale, color, 1, cv2.LINE_AA)
        # Right-align value
        (vw, _), _ = cv2.getTextSize(val, font, scale, 1)
        cv2.putText(frame, val, (px + panel_w - 8 - vw, y),
                    font, scale, color, 1, cv2.LINE_AA)


def draw_hud_bar(frame: np.ndarray, total_tracked: int) -> None:
    """
    Thin top bar: project name + active track count.
    """
    h, w = frame.shape[:2]
    bar_h = 32
    frame[:] = _alpha_rect(frame, 0, 0, w, bar_h, (10, 10, 10), 0.75)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    cv2.putText(frame, "ADAS SIMULATOR  |  Milestone 1: Detection & Tracking",
                (10, 21), font, scale, (100, 200, 255), 1, cv2.LINE_AA)

    right_text = f"Active tracks: {total_tracked}"
    (tw, _), _ = cv2.getTextSize(right_text, font, scale, 1)
    cv2.putText(frame, right_text, (w - tw - 10, 21),
                font, scale, (100, 255, 150), 1, cv2.LINE_AA)
