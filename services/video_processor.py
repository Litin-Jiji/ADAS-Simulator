"""
video_processor.py
Full ADAS Pipeline:
  Frame → Tracker → Lane Detection → Depth → Collision Risk → Draw → Output
"""

import cv2
import sys
from pathlib import Path

import config
from services.tracker import Tracker
from services.lane_detection import LaneDetector
from services.depth_estimator import DepthEstimator
from services.collision_risk import CollisionRiskAssessor
from utils.draw import (draw_tracked_objects, draw_stats_panel, draw_hud_bar,
                        draw_lanes, draw_depth_labels, draw_depth_map_thumbnail,
                        draw_risk_overlays, draw_collision_warning)
from utils.fps import FPSCounter


class VideoProcessor:
    def __init__(self, source: str | int, save_output: bool = False,
                 enable_depth: bool = True):
        self.source       = source
        self.save_output  = save_output
        self.enable_depth = enable_depth

        self.tracker      = Tracker()
        self.lane_det     = LaneDetector()
        self.risk_assess  = CollisionRiskAssessor()
        self.depth_est    = DepthEstimator(device=config.DEVICE) if enable_depth else None
        self.fps_counter  = FPSCounter()
        self.writer       = None

        self.depth_every  = 4      # run depth every N frames
        self._depth_map   = None
        self._frame_num   = 0

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open: {self.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return cap

    def _init_writer(self, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
        name = Path(str(self.source)).stem if isinstance(self.source, str) else "webcam"
        out  = config.OUTPUTS_DIR / f"{name}_adas.mp4"
        print(f"[INFO] Saving to {out}")
        return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def run(self):
        cap = self._open_capture()
        if self.save_output:
            self.writer = self._init_writer(cap)

        print("[INFO] ADAS running — press Q to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            self._frame_num += 1
            h, w = frame.shape[:2]

            # ── 1. Tracking ────────────────────────────────────────────────
            roi_top    = int(h * 0.35)
            roi_bottom = int(h * 0.90)
            roi_frame  = frame[roi_top:roi_bottom, :]
            tracked_roi = self.tracker.track(roi_frame)
            fps = self.fps_counter.tick()

            tracked = []
            for obj in tracked_roi:
                obj["bbox"][1] += roi_top
                obj["bbox"][3] += roi_top
                if obj.get("history"):
                    obj["history"] = [(x, y + roi_top) for x, y in obj["history"]]
                cx, cy = obj["centroid"]
                obj["centroid"] = (cx, cy + roi_top)
                tracked.append(obj)

            # ── 2. Lane detection ──────────────────────────────────────────
            lane_result = self.lane_det.detect(frame)

            # ── 3. Depth estimation ────────────────────────────────────────
            if self.enable_depth and self.depth_est:
                if self._frame_num % self.depth_every == 0:
                    self._depth_map = self.depth_est.estimate(frame)
                if self._depth_map is not None:
                    for obj in tracked:
                        obj["distance_m"] = self.depth_est.get_object_distance(
                            self._depth_map, obj["bbox"]
                        )

            # ── 4. Collision risk ──────────────────────────────────────────
            risk_results, warning = self.risk_assess.assess(tracked)

            # Attach risk to each tracked obj for stats panel
            risk_map = {r.track_id: r.risk for r in risk_results}
            for obj in tracked:
                obj["risk"] = risk_map.get(obj["track_id"], "LOW")

            # ── 5. Draw (order matters) ────────────────────────────────────
            draw_hud_bar(frame, len(tracked))
            draw_lanes(frame, lane_result)
            draw_tracked_objects(frame, tracked)
            draw_depth_labels(frame, tracked)
            draw_risk_overlays(frame, risk_results)
            draw_collision_warning(frame, warning)
            if self._depth_map is not None:
                draw_depth_map_thumbnail(frame, self._depth_map)
            draw_stats_panel(frame, tracked, fps, self._frame_num)

            # ── 6. Output ──────────────────────────────────────────────────
            if self.writer:
                self.writer.write(frame)

            cv2.imshow("ADAS Simulator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        self.tracker.reset()
