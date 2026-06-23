"""
video_processor.py
Orchestrates the full Milestone 1 pipeline:
  Frame → Tracker → Draw → Display / Save
"""

import cv2
import sys
from pathlib import Path

import config
from services.tracker import Tracker
from utils.draw import draw_tracked_objects, draw_stats_panel, draw_hud_bar
from utils.fps import FPSCounter


class VideoProcessor:
    def __init__(self, source: str | int, save_output: bool = False):
        """
        source      : path to video file, or 0 for webcam
        save_output : write annotated video to outputs/
        """
        self.source      = source
        self.save_output = save_output
        self.tracker     = Tracker()
        self.fps_counter = FPSCounter()
        self.writer      = None

    # ── internal helpers ───────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open source: {self.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return cap

    def _init_writer(self, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
        src_name = Path(str(self.source)).stem if isinstance(self.source, str) else "webcam"
        out_path = config.OUTPUTS_DIR / f"{src_name}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        print(f"[INFO] Saving output to {out_path}")
        return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # ── public ────────────────────────────────────────────────────────────

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

            # ── Core pipeline ──────────────────────────────────────────
            tracked = self.tracker.track(frame)
            fps     = self.fps_counter.tick()

            # ── Draw ───────────────────────────────────────────────────
            draw_hud_bar(frame, len(tracked))
            draw_tracked_objects(frame, tracked)
            draw_stats_panel(frame, tracked, fps, frame_num)

            # ── Output ────────────────────────────────────────────────
            if self.writer:
                self.writer.write(frame)

            cv2.imshow("ADAS Simulator — M1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit.")
                break

        cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        self.tracker.reset()
