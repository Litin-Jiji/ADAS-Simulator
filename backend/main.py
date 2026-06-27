"""
backend/main.py
FastAPI backend for ADAS Simulator.

Endpoints:
  GET  /                    → health check
  GET  /api/status          → current system status
  POST /api/start           → start processing a video
  POST /api/stop            → stop processing
  GET  /api/analytics       → trip analytics summary
  WS   /ws                  → real-time telemetry stream
"""

import asyncio
import json
import time
import cv2
import sys
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.append(str(Path(__file__).parent.parent))

import config
from services.tracker import Tracker
from services.lane_detection import LaneDetector
from services.collision_risk import CollisionRiskAssessor
from utils.fps import FPSCounter

app = FastAPI(title="ADAS Simulator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────

class PipelineState:
    def __init__(self):
        self.running          = False
        self.source           = None
        self.frame_num        = 0
        self.fps              = 0.0
        self.active_tracks    = 0
        self.class_counts     = {}
        self.lane_status      = "No Lane"
        self.lane_offset      = 0.0
        self.collision_risk   = "LOW"
        self.ttc              = 99.0
        self.warning_active   = False
        self.warning_msg      = ""
        self.total_vehicles   = 0
        self.near_misses      = 0
        self.lane_departures  = 0
        self.high_risk_events = 0
        self.start_time       = None
        self.risk_history     = []

pipeline = PipelineState()
connected_clients: list[WebSocket] = []


async def broadcast(data: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connected_clients:
            connected_clients.remove(ws)


async def run_pipeline(source):
    tracker     = Tracker()
    lane_det    = LaneDetector()
    risk_assess = CollisionRiskAssessor()
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        pipeline.running = False
        return

    seen_ids: set = set()

    while pipeline.running:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Video ended.")
            break

        pipeline.frame_num += 1
        h, w = frame.shape[:2]

        # Tracking
        roi_top    = int(h * 0.35)
        roi_bottom = int(h * 0.90)
        tracked_roi = tracker.track(frame[roi_top:roi_bottom, :])
        tracked = []
        for obj in tracked_roi:
            obj["bbox"][1] += roi_top
            obj["bbox"][3] += roi_top
            cx, cy = obj["centroid"]
            obj["centroid"] = (cx, cy + roi_top)
            tracked.append(obj)
            if obj["track_id"] not in seen_ids:
                seen_ids.add(obj["track_id"])
                pipeline.total_vehicles += 1

        fps = fps_counter.tick()

        # Lane
        lane_result = lane_det.detect(frame)
        if lane_result.warning:
            pipeline.lane_departures += 1

        # Distance heuristic (no depth model in backend — saves GPU for frontend)
        for obj in tracked:
            x1, y1, x2, y2 = obj["bbox"]
            obj["distance_m"] = max(1.0, round(120.0 / max(y2 - y1, 1), 1))

        # Risk
        risk_results, warning = risk_assess.assess(tracked)
        if warning.risk == "CRITICAL":
            pipeline.near_misses += 1
        if warning.risk in ("HIGH", "CRITICAL"):
            pipeline.high_risk_events += 1

        counts: dict[str, int] = defaultdict(int)
        for obj in tracked:
            counts[obj["class_name"]] += 1

        pipeline.risk_history.append(warning.risk)
        if len(pipeline.risk_history) > 300:
            pipeline.risk_history.pop(0)

        pipeline.fps            = round(fps, 1)
        pipeline.active_tracks  = len(tracked)
        pipeline.class_counts   = dict(counts)
        pipeline.lane_status    = lane_result.status
        pipeline.lane_offset    = round(lane_result.offset_cm, 1)
        pipeline.collision_risk = warning.risk
        pipeline.ttc            = warning.ttc
        pipeline.warning_active = warning.active
        pipeline.warning_msg    = warning.message

        await broadcast({
            "type":           "telemetry",
            "frame":          pipeline.frame_num,
            "fps":            pipeline.fps,
            "active_tracks":  pipeline.active_tracks,
            "class_counts":   pipeline.class_counts,
            "lane_status":    pipeline.lane_status,
            "lane_offset":    pipeline.lane_offset,
            "collision_risk": pipeline.collision_risk,
            "ttc":            pipeline.ttc if pipeline.ttc < 90 else None,
            "warning_active": pipeline.warning_active,
            "warning_msg":    pipeline.warning_msg,
            "timestamp":      time.time(),
        })

        await asyncio.sleep(0.01)

    cap.release()
    tracker.reset()
    pipeline.running = False


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "ADAS Simulator API v1.0"}


@app.get("/api/status")
async def status():
    elapsed = round(time.time() - pipeline.start_time, 1) if pipeline.start_time else 0
    return {
        "running":        pipeline.running,
        "frame":          pipeline.frame_num,
        "fps":            pipeline.fps,
        "active_tracks":  pipeline.active_tracks,
        "class_counts":   pipeline.class_counts,
        "lane_status":    pipeline.lane_status,
        "collision_risk": pipeline.collision_risk,
        "warning_active": pipeline.warning_active,
        "elapsed_sec":    elapsed,
    }


@app.post("/api/start")
async def start(body: dict):
    if pipeline.running:
        raise HTTPException(400, "Already running")
    source = body.get("source", "videos/dashcam2.mp4")
    pipeline.running          = True
    pipeline.source           = source
    pipeline.frame_num        = 0
    pipeline.total_vehicles   = 0
    pipeline.near_misses      = 0
    pipeline.lane_departures  = 0
    pipeline.high_risk_events = 0
    pipeline.risk_history     = []
    pipeline.start_time       = time.time()
    src = int(source) if str(source) == "0" else source
    asyncio.create_task(run_pipeline(src))
    return {"status": "started", "source": source}


@app.post("/api/stop")
async def stop():
    pipeline.running = False
    return {"status": "stopped"}


@app.get("/api/analytics")
async def analytics():
    elapsed = round(time.time() - pipeline.start_time, 1) if pipeline.start_time else 0
    risk_counts: dict[str, int] = defaultdict(int)
    for r in pipeline.risk_history:
        risk_counts[r] += 1
    return {
        "elapsed_sec":       elapsed,
        "total_frames":      pipeline.frame_num,
        "total_vehicles":    pipeline.total_vehicles,
        "near_misses":       pipeline.near_misses,
        "lane_departures":   pipeline.lane_departures,
        "high_risk_events":  pipeline.high_risk_events,
        "avg_fps":           pipeline.fps,
        "risk_distribution": dict(risk_counts),
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in connected_clients:
            connected_clients.remove(ws)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
