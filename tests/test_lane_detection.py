"""
tests/test_lane_detection.py
Basic smoke tests for lane detector.
"""
import sys, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from services.lane_detection import LaneDetector


def test_blank_frame_no_crash():
    det    = LaneDetector()
    frame  = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = det.detect(frame)
    assert result.status in ("No Lane", "Drifting Left", "Drifting Right", "Centered")


def test_returns_lane_result():
    det    = LaneDetector()
    frame  = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = det.detect(frame)
    assert hasattr(result, "offset_cm")
    assert hasattr(result, "warning")
    assert hasattr(result, "status")
