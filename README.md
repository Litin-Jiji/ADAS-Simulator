# ADAS Simulator

An AI Driver Assistance System built with Python, YOLOv11, and ByteTrack.
Processes dashcam footage in real time to detect and track vehicles, pedestrians,
and cyclists — the foundation of a full autonomous driving perception stack.

## Architecture

```
Dashcam Video
      │
Frame Extraction
      │
 YOLOv11 Detection  (cars, trucks, buses, motorcycles, bicycles, pedestrians)
      │
 ByteTrack          (persistent IDs across frames)
      │
 HUD Overlay        (bounding boxes, trails, stats panel)
      │
Display / Save
```

## Tech Stack

| Layer     | Technology                     |
|-----------|-------------------------------|
| Detection | YOLOv11n (Ultralytics)        |
| Tracking  | ByteTrack (via Ultralytics)   |
| Vision    | OpenCV 4.9                    |
| Compute   | PyTorch + CUDA (GTX 4050)     |

## Milestones

- [x] **M1** — Detection & Tracking
- [ ] **M2** — Lane Detection (CLRNet)
- [ ] **M3** — Traffic Sign Recognition
- [ ] **M4** — Monocular Depth Estimation (Depth Anything V2)
- [ ] **M5** — Collision Risk & TTC
- [ ] **M6** — Analytics Dashboard (FastAPI + React)
- [ ] **M7** — Docker + Azure Deployment

## Setup

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/adas-simulator
cd adas-simulator

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Drop a dashcam video into videos/
#    (or use --source 0 for webcam)
```

## Run

```bash
# Webcam
python main.py

# Video file
python main.py --source videos/dashcam.mp4

# Video file + save output
python main.py --source videos/dashcam.mp4 --save
```

YOLOv11n weights (~6 MB) download automatically on first run.

## Dataset

Primary: [BDD100K](https://bdd-data.berkeley.edu/) — dashcam videos with
vehicle, pedestrian, lane, and traffic sign annotations.

## Project Structure

```
adas/
├── main.py                  # Entry point
├── config.py                # All tunable parameters
├── requirements.txt
├── services/
│   ├── detector.py          # YOLOv11 wrapper
│   ├── tracker.py           # ByteTrack wrapper
│   └── video_processor.py   # Pipeline orchestrator
├── utils/
│   ├── draw.py              # HUD, boxes, trails, stats
│   └── fps.py               # Rolling FPS counter
├── models/                  # Place .pt weights here
├── videos/                  # Input dashcam footage
└── outputs/                 # Annotated video output
```
