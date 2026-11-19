"""
config.py
Central configuration for the Traffic Sign Recognition and Road Safety Monitor.
Every other module imports its paths, constants and thresholds from here.
"""
import logging
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
GTSRB_DIR = DATASETS_DIR / "GTSRB"
GTSDB_DIR = DATASETS_DIR / "GTSDB"
TEST_VIDEOS_DIR = DATASETS_DIR / "test_videos"
GTSDB_YOLO_DIR = DATASETS_DIR / "GTSDB_YOLO"   # created by the converter later

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_WEIGHTS = MODELS_DIR / "traffic_sign_cnn.pth"
SIGN_DETECTOR_WEIGHTS = MODELS_DIR / "traffic_sign_yolo.pt"
OBJECT_DETECTOR_WEIGHTS = "yolov8n.pt"  # pretrained COCO; Ultralytics auto-downloads

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANNOTATED_VIDEOS_DIR = OUTPUTS_DIR / "annotated_videos"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

ALL_DIRS = [
    GTSRB_DIR, GTSDB_DIR, TEST_VIDEOS_DIR, MODELS_DIR,
    ANNOTATED_VIDEOS_DIR, PREDICTIONS_DIR, EVALUATION_DIR,
]

# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Traffic sign classifier (GTSRB)
# ---------------------------------------------------------------------------
NUM_SIGN_CLASSES = 43
SIGN_IMG_SIZE = 32                       # CNN input is 32x32 RGB
SIGN_MEAN = (0.3403, 0.3121, 0.3214)     # placeholder; recompute from the training set in preprocessing.py
SIGN_STD = (0.2724, 0.2608, 0.2669)      # placeholder; recompute from the training set in preprocessing.py
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
NUM_WORKERS = 0                          # 0 is safest on Windows

# ---------------------------------------------------------------------------
# Sign detector (GTSDB -> YOLO)
# ---------------------------------------------------------------------------
SIGN_DET_CONF = 0.25
SIGN_DET_IMG_SIZE = 960

# ---------------------------------------------------------------------------
# General object detector (COCO classes used)
# ---------------------------------------------------------------------------
COCO_ROAD_CLASSES = {
    0: "pedestrian",   # COCO 'person'
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_NAMES = {"car", "bus", "truck", "motorcycle"}
VULNERABLE_CLASS_NAMES = {"pedestrian", "bicycle"}
OBJECT_DET_CONF = 0.35
TRACKER_CONFIG = "bytetrack.yaml"        # bundled with Ultralytics
TRAJECTORY_LENGTH = 30                   # frames of history kept per track

# ---------------------------------------------------------------------------
# Risk analysis (all values are in PIXEL units: prototype, uncalibrated)
# ---------------------------------------------------------------------------
RISK_CONFIG = {
    # Proximity is measured as center distance divided by the mean box diagonal,
    # so it scales with how large the objects appear.
    "ped_vehicle_proximity_ratio": 1.5,
    "vehicle_vehicle_proximity_ratio": 0.8,
    "high_density_count": 10,                # objects in a frame
    "sign_approach_ratio": 3.0,              # object-to-sign distance / sign box size
    "persistence_frames": 8,                 # condition must hold this many frames
    "smoothing_window": 10,                  # frames for temporal smoothing
    "medium_score": 0.35,
    "high_score": 0.65,
    "alert_cooldown_seconds": 3.0,
}

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

# ---------------------------------------------------------------------------
# Video / app
# ---------------------------------------------------------------------------
PROCESS_EVERY_N_FRAMES = 1               # raise to speed up on slow CPUs
MAX_UPLOAD_MB = 200
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once. Safe to call repeatedly."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs() -> None:
    """Create every required directory if it does not exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)