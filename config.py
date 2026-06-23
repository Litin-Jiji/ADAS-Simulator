from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "models"
VIDEOS_DIR  = ROOT / "videos"
OUTPUTS_DIR = ROOT / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────
YOLO_MODEL  = "yolo11n.pt"   # auto-downloaded on first run
DEVICE      = "cuda"         # "cuda" | "cpu" | "mps"

# ── Detection ──────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.30
IOU_THRESHOLD   = 0.45

# COCO classes we care about (id: label)
TARGET_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Colour per class  BGR
CLASS_COLORS = {
    "person":     (0,   200, 255),
    "bicycle":    (0,   255, 180),
    "car":        (0,   255,  80),
    "motorcycle": (255, 180,   0),
    "bus":        (255,  80,   0),
    "truck":      (180,   0, 255),
}

# ── Tracker (ByteTrack via Ultralytics) ────────────────────────────────────
TRACK_BUFFER    = 30   # frames to keep a lost track alive
MIN_BOX_AREA    = 200  # px², ignore tiny detections

# ── Video processing ───────────────────────────────────────────────────────
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
TARGET_FPS      = 30

# ── HUD ────────────────────────────────────────────────────────────────────
HUD_FONT_SCALE  = 0.55
HUD_THICKNESS   = 1
HUD_COLOR       = (220, 220, 220)
HUD_BG_ALPHA    = 0.55
