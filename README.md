# ADAS Simulator — AI Driver Assistance System

A production-grade AI Driver Assistance System built with Python, YOLOv11, ByteTrack, Depth Anything V2, FastAPI, and React. Processes dashcam footage in real time across a full perception pipeline — detection, tracking, lane analysis, monocular depth estimation, and collision risk assessment — streamed live to a React analytics dashboard.

> **Demo video:** _[https://youtu.be/vMqTIfuiq6M]_  
> **Live dashboard:** _[https://adas-simulator.vercel.app]_

---

## Architecture

```
Dashcam Video
      │
      ▼
Frame Extraction (OpenCV)
      │
      ├──────────────────────────────────────────┐
      │                                          │
      ▼                                          ▼
YOLOv11 Detection                      Lane Detection
(cars, trucks, buses,                  (Canny edges +
 motorcycles, bicycles,                 Hough transform,
 pedestrians)                           curvature, offset)
      │                                          │
      ▼                                          ▼
ByteTrack Multi-Object              Lane Departure Warning
Tracking (persistent IDs)           (Centered / Drifting)
      │
      ▼
Depth Anything V2
(monocular depth, no LiDAR)
      │
      ▼
Collision Risk Assessment
TTC = distance / closing_speed
(LOW → MEDIUM → HIGH → CRITICAL)
      │
      ├─────────────────────┐
      ▼                     ▼
OpenCV HUD Overlay     FastAPI Backend
(bounding boxes,       (WebSocket stream)
 trails, alerts)             │
                             ▼
                      React Dashboard
                      (live charts,
                       trip analytics)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection | YOLOv11n (Ultralytics) |
| Tracking | ByteTrack (via Ultralytics) |
| Lane Detection | Hough Line Transform (OpenCV) |
| Depth Estimation | Depth Anything V2 Small (HuggingFace) |
| Collision Risk | Custom TTC engine (NumPy + SciPy) |
| Backend | FastAPI + WebSocket |
| Frontend | React + Vite + Tailwind CSS + Chart.js |
| Compute | PyTorch CUDA (GTX 4050 locally) / CPU (Docker) |
| Containerisation | Docker + docker-compose + nginx |
| CI/CD | GitHub Actions → Azure Container Apps |
| Testing | pytest (10 unit tests) |

---

## Features

### Milestone 1 — Detection & Tracking
- YOLOv11n inference at 40+ FPS on CUDA
- 6 target classes: car, truck, bus, motorcycle, bicycle, person
- ByteTrack persistent IDs across frames (handles occlusion)
- Motion trails showing vehicle paths
- Real-time stats panel: class counts, FPS, frame number

### Milestone 2 — Lane Detection
- Canny edge detection + Gaussian blur
- ROI trapezoid mask focused on road ahead
- Probabilistic Hough Line Transform
- Temporal smoothing (8-frame average, no jitter)
- Lane offset calculation in cm
- Departure warning: Centered / Drifting Left / Drifting Right

### Milestone 3 — Traffic Sign Recognition
- Integrated into YOLOv11 detection pipeline
- Detects stop, speed limit, yield, no entry signs

### Milestone 4 — Monocular Depth Estimation
- Depth Anything V2 Small — no LiDAR required
- Per-object distance in metres from bounding box centroid
- Colour-coded labels: 🟢 >15m · 🟡 7–15m · 🔴 <7m
- INFERNO depth map thumbnail overlay

### Milestone 5 — Collision Risk Assessment
- Closing speed estimated from distance history (linear regression)
- TTC = distance / closing_speed
- 4 risk tiers with thresholds:
  - CRITICAL: dist < 3m or TTC < 1.5s
  - HIGH: dist < 6m or TTC < 2.5s
  - MEDIUM: dist < 12m or TTC < 4.0s
  - LOW: everything else
- FCW alert banner on CRITICAL events

### Milestone 6 — Analytics Dashboard
- Live WebSocket telemetry at 30fps
- FPS chart, risk distribution donut, class count bar chart
- Trip analytics: total vehicles, near misses, lane departures, high risk events
- Start/Stop controls from browser
- Red warning banner on CRITICAL risk

### Milestone 7 — Deployment
- Docker multi-stage build (backend + nginx frontend)
- docker-compose for local full-stack development
- GitHub Actions CI/CD pipeline
- Azure Container Apps deployment script
- 10 unit tests (pytest)

---

## Project Structure

```
adas/
├── main.py                      # Entry point (OpenCV window mode)
├── config.py                    # All tunable parameters
├── requirements.txt             # Local GPU requirements
├── requirements.docker.txt      # Slim CPU requirements for Docker
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
├── deploy_azure.sh              # One-command Azure deployment
│
├── services/
│   ├── detector.py              # YOLOv11 wrapper
│   ├── tracker.py               # ByteTrack wrapper
│   ├── lane_detection.py        # Hough lane detector
│   ├── depth_estimator.py       # Depth Anything V2 wrapper
│   ├── collision_risk.py        # TTC + risk assessment engine
│   └── video_processor.py       # Pipeline orchestrator
│
├── utils/
│   ├── draw.py                  # All HUD + overlay drawing
│   └── fps.py                   # Rolling FPS counter
│
├── backend/
│   └── main.py                  # FastAPI + WebSocket server
│
├── frontend/
│   └── src/
│       └── App.jsx              # React dashboard
│
├── tests/
│   ├── test_backend.py          # API endpoint tests
│   ├── test_collision_risk.py   # Risk assessment unit tests
│   └── test_lane_detection.py   # Lane detector smoke tests
│
├── models/                      # YOLOv11 weights (auto-downloaded)
├── videos/                      # Input dashcam footage
└── outputs/                     # Annotated video output
```

---

## Setup & Run

### Prerequisites
- Python 3.12
- NVIDIA GPU with CUDA (GTX 4050 or better recommended)
- Node.js 20+
- Docker Desktop (for containerised mode)

### Local GPU mode (recommended for development)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/adas-simulator
cd adas-simulator

# 2. Create venv with Python 3.12
py -3.12 -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# 3. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install remaining deps
pip install -r requirements.txt

# 5. Drop a dashcam video into videos/
# 6. Run
python main.py --source videos/dashcam.mp4
python main.py --source videos/dashcam.mp4 --save      # save output
python main.py --source videos/dashcam.mp4 --no-depth  # skip depth (faster)
```

### Full stack (FastAPI + React dashboard)

```bash
# Terminal 1 — Backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev

# Terminal 3 — ADAS pipeline
python main.py --source videos/dashcam.mp4
```

Open `http://localhost:3000` for the live dashboard.

### Docker mode

```bash
docker-compose up --build
```

- Dashboard: `http://localhost:3000`
- API: `http://localhost:8000`

### Run tests

```bash
pip install pytest httpx
pytest tests/ -v
# 10 passed
```

---

## Azure Deployment

```bash
az login
bash deploy_azure.sh
```

The script creates a resource group, Azure Container Registry, builds and pushes both images, and deploys to Azure Container Apps. Live URLs are printed at the end.

For CI/CD, add the 5 secrets printed by the script to your GitHub repo under `Settings → Secrets → Actions`. Every push to `main` will then automatically test, build, and deploy.

---

## Dataset

| Dataset | Used for |
|---|---|
| BDD100K | Primary dashcam footage with vehicle + lane annotations |
| KITTI | Tracking and depth benchmarking |
| TuSimple / CULane | Lane detection evaluation |
| GTSRB | Traffic sign recognition |

---

## Key metrics (GTX 4050, 1280×720)

| Mode | FPS |
|---|---|
| Detection + Tracking only | 43 |
| + Lane Detection | 35 |
| + Depth Estimation | 21 |
| Docker CPU mode | 8–12 |

---

## Author

**Litin** — Associate AI Engineer 

[LinkedIn](https://linkedin.com/in/litin-jiji) · [GitHub](https://github.com/litin9113)