"""
video_processor.py
Pipeline: Frame → ROI → Tracker → Lane Detection → Draw → Display / Save
"""

import cv2
import sys
from pathlib import Path

import config
from services.tracker import Tracker
from services.lane_detection import LaneDetector
from utils.draw import draw_tracked_objects, draw_stats_panel, draw_hud_bar, draw_lanes
from utils.fps import FPSCounter


class VideoProcessor:
    def __init__(self, source: str | int, save_output: bool = False):
        self.source       = source
        self.save_output  = save_output
        self.tracker      = Tracker()
        self.lane_detector = LaneDetector()
        self.fps_counter  = FPSCounter()
        self.writer       = None

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open source: {self.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return cap

    def _init_writer(self, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps  = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
        name = Path(str(self.source)).stem if isinstance(self.source, str) else "webcam"
        out  = config.OUTPUTS_DIR / f"{name}_tracked.mp4"
        print(f"[INFO] Saving to {out}")
        return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def run(self):
        cap = self._open_capture()
        if self.save_output:
            self.writer = self._init_writer(cap)

        frame_num = 0
        print("[INFO] Running — press Q to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of stream.")
                break

            frame_num += 1
            h, w = frame.shape[:2]

            # ── ROI for tracker: skip top 35% (sky) and bottom 10% (bonnet) ──
            roi_top    = int(h * 0.35)
            roi_bottom = int(h * 0.90)
            roi_frame  = frame[roi_top:roi_bottom, :]

            # ── Tracking ───────────────────────────────────────────────────
            tracked_roi = self.tracker.track(roi_frame)
            fps         = self.fps_counter.tick()

            # Shift y-coords back to full-frame space
            tracked = []
            for obj in tracked_roi:
                obj["bbox"][1] += roi_top
                obj["bbox"][3] += roi_top
                if obj.get("history"):
                    obj["history"] = [(x, y + roi_top) for x, y in obj["history"]]
                cx, cy = obj["centroid"]
                obj["centroid"] = (cx, cy + roi_top)
                tracked.append(obj)

            # ── Lane detection (runs on full frame) ────────────────────────
            lane_result = self.lane_detector.detect(frame)

            # ── Draw (order matters) ───────────────────────────────────────
            draw_hud_bar(frame, len(tracked))
            draw_lanes(frame, lane_result)          # lanes first (under boxes)
            draw_tracked_objects(frame, tracked)    # boxes on top
            draw_stats_panel(frame, tracked, fps, frame_num)

            # ── Output ─────────────────────────────────────────────────────
            if self.writer:
                self.writer.write(frame)

            cv2.imshow("ADAS Simulator — M1 + M2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit.")
                break

        cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        self.tracker.reset()
