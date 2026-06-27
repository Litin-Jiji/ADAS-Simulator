"""
video_processor.py
Pipeline: Frame → Tracker → Lane Detection → Depth Estimation → Draw → Output
"""

import cv2
import sys
from pathlib import Path

import config
from services.tracker import Tracker
from services.lane_detection import LaneDetector
from services.depth_estimator import DepthEstimator
from utils.draw import (draw_tracked_objects, draw_stats_panel,
                        draw_hud_bar, draw_lanes,
                        draw_depth_labels, draw_depth_map_thumbnail)
from utils.fps import FPSCounter


class VideoProcessor:
    def __init__(self, source: str | int, save_output: bool = False,
                 enable_depth: bool = True):
        self.source        = source
        self.save_output   = save_output
        self.enable_depth  = enable_depth
        self.tracker       = Tracker()
        self.lane_detector = LaneDetector()
        self.depth_est     = DepthEstimator(device=config.DEVICE) if enable_depth else None
        self.fps_counter   = FPSCounter()
        self.writer        = None

        # Run depth every N frames (saves GPU — depth doesn't need 30fps)
        self.depth_every   = 3
        self._depth_map    = None
        self._depth_frame  = 0

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
        out  = config.OUTPUTS_DIR / f"{name}_adas.mp4"
        print(f"[INFO] Saving to {out}")
        return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def run(self):
        cap = self._open_capture()
        if self.save_output:
            self.writer = self._init_writer(cap)

        frame_num = 0
        print("[INFO] Running — press Q to quit.")
        print(f"[INFO] Depth estimation: {'ON' if self.enable_depth else 'OFF'}")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of stream.")
                break

            frame_num += 1
            h, w = frame.shape[:2]

            # ── Tracking ROI ───────────────────────────────────────────────
            roi_top    = int(h * 0.35)
            roi_bottom = int(h * 0.90)
            roi_frame  = frame[roi_top:roi_bottom, :]

            tracked_roi = self.tracker.track(roi_frame)
            fps         = self.fps_counter.tick()

            tracked = []
            for obj in tracked_roi:
                obj["bbox"][1] += roi_top
                obj["bbox"][3] += roi_top
                if obj.get("history"):
                    obj["history"] = [(x, y + roi_top) for x, y in obj["history"]]
                cx, cy = obj["centroid"]
                obj["centroid"] = (cx, cy + roi_top)
                tracked.append(obj)

            # ── Lane detection ─────────────────────────────────────────────
            lane_result = self.lane_detector.detect(frame)

            # ── Depth estimation (every N frames) ──────────────────────────
            if self.enable_depth and self.depth_est:
                if frame_num % self.depth_every == 0:
                    self._depth_map = self.depth_est.estimate(frame)

                if self._depth_map is not None:
                    for obj in tracked:
                        obj["distance_m"] = self.depth_est.get_object_distance(
                            self._depth_map, obj["bbox"]
                        )

            # ── Draw ───────────────────────────────────────────────────────
            draw_hud_bar(frame, len(tracked))
            draw_lanes(frame, lane_result)
            draw_tracked_objects(frame, tracked)
            draw_depth_labels(frame, tracked)
            if self._depth_map is not None:
                draw_depth_map_thumbnail(frame, self._depth_map)
            draw_stats_panel(frame, tracked, fps, frame_num)

            # ── Output ─────────────────────────────────────────────────────
            if self.writer:
                self.writer.write(frame)

            cv2.imshow("ADAS Simulator — M1+M2+M4", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit.")
                break

        cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        self.tracker.reset()
