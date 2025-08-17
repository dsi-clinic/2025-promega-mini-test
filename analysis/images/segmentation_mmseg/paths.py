from pathlib import Path
import glob
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# ---------- helpers ----------
def require_env_path(key: str) -> Path:
    v = os.environ.get(key)
    if not v:
        raise KeyError(f"Missing required env var: {key}")
    return Path(v)

# ---------- load .env ----------
env_path = find_dotenv()
if not load_dotenv(env_path):
    raise RuntimeError(f"Failed to load .env (tried: {env_path})")

# ---------- core env-backed paths ----------
BASE_PATH            = require_env_path("BASE_PATH")
OUTPUT_FOLDER        = require_env_path("OUTPUT_FOLDER")
PLOTS_FOLDER         = require_env_path("PLOTS_FOLDER")
PREDICTIONS_DIR      = require_env_path("PREDICTIONS_DIR")
ORIGINAL_MAPPING     = require_env_path("ORIGINAL_MAPPING")
MANUAL_MAPPING_DIR   = require_env_path("MANUAL_MAPPING_DIR")
MANUAL_PROCESSED_DIR = require_env_path("MANUAL_PROCESSED_DIR")

# ---------- sizes ----------
TARGET_WIDTH  = int(os.environ.get("TARGET_WIDTH", "584"))
TARGET_HEIGHT = int(os.environ.get("TARGET_HEIGHT", "384"))
TARGET_SIZE   = (TARGET_WIDTH, TARGET_HEIGHT)

# ---------- manual masks ----------
MANUAL_MASKS_DIR         = BASE_PATH / "manual_masks"
MANUAL_THRESHOLD_MAPPING = MANUAL_MASKS_DIR / "image_mapping_thresholded_and_manual.json"

MANUAL_MASK_FOLDERS = [
    Path(p) for p in glob.glob(str(MANUAL_MASKS_DIR / "masks-batch-*"))
    if Path(p).is_dir()
]

# ---------- processed dataset (resized) ----------
PROCESSED_IMAGES_DIR = MANUAL_PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR  = MANUAL_PROCESSED_DIR / "masks"
MAPPING_PROCESSED_TOTAL = (
    MANUAL_PROCESSED_DIR / f"mapping_processed_total_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.json"
)
PROCESSED_DATA_DIR = require_env_path("PROCESSED_DATA_DIR")

# ---------- predictions ----------
OUTPUT_MASKS_BASE_DIR = PREDICTIONS_DIR  # single source of truth

# ---------- SegFormer model locations ----------
EARLY_MODEL = {
    "config": PLOTS_FOLDER / "segformer_masks/512by384/models/early/vis_data/config.py",
    "checkpoint": PLOTS_FOLDER / "segformer_masks/512by384/models/early/iter_1000.pth",
}

LATE_MODEL = {
    "config": PLOTS_FOLDER / "segformer_masks/512by384/models/late/vis_data/config.py",
    "checkpoint": PLOTS_FOLDER / "segformer_masks/512by384/models/late/iter_1000.pth",
}
