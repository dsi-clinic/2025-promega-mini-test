from dotenv import load_dotenv
import os
from pathlib import Path


load_dotenv()

def require_env_path(key):
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Environment variable '{key}' not set.")
    return Path(val)

def require_env_int(key):
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Environment variable '{key}' not set.")
    return int(val)

# Resize dimensions
TARGET_WIDTH  = require_env_int("TARGET_WIDTH")
TARGET_HEIGHT = require_env_int("TARGET_HEIGHT")
TARGET_SIZE   = (TARGET_WIDTH, TARGET_HEIGHT)
TARGET_SUFFIX = f"{TARGET_WIDTH}x{TARGET_HEIGHT}"

# Base paths
BASE_PATH         = require_env_path("BASE_PATH")
OUTPUT_FOLDER     = require_env_path("OUTPUT_FOLDER")
PLOTS_FOLDER      = require_env_path("PLOTS_FOLDER")
LOGS_FOLDER       = require_env_path("LOGS_FOLDER")
NPY_OUTPUTS       = require_env_path("NPY_OUTPUTS")
PREDICTIONS_DIR   = require_env_path("PREDICTIONS_DIR")

# Metadata
META_FILE         = require_env_path("META_FILE")
SURVEY_RESULTS    = require_env_path("SURVEY_RESULTS")

# Mask inputs
MANUAL_MASKS_DIR    = require_env_path("MANUAL_MASKS_DIR")
MANUAL_MASK_FOLDERS = [
    MANUAL_MASKS_DIR / "Manuais",
    MANUAL_MASKS_DIR / "Treshold"
]

# Original image mapping
ORIGINAL_MAPPING = require_env_path("ORIGINAL_MAPPING")

# Processed resized image output (from original image mapping)
PREPROCESSED_DIR     = require_env_path("PREPROCESSED_DIR")
PROCESSED_DATA_DIR   = require_env_path("PROCESSED_DATA_DIR")
PREPROCESSED_JSON_DIR = PROCESSED_DATA_DIR
OUTPUT_MASKS_BASE_DIR = PREDICTIONS_DIR


# Manual processed data and splits
MANUAL_MAPPING_DIR       = require_env_path("MANUAL_MAPPING_DIR")
MANUAL_PROCESSED_DIR     = require_env_path("MANUAL_PROCESSED_DIR")
PROCESSED_IMAGES_DIR     = MANUAL_PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR      = MANUAL_PROCESSED_DIR / "masks"
MAPPING_PROCESSED_TOTAL  = require_env_path("MAPPING_PROCESSED_TOTAL")
MANUAL_SPLITS_DIR        = MANUAL_PROCESSED_DIR / "split"

# SegFormer
# Model configs and checkpoints
EARLY_MODEL = {
    "config": PLOTS_FOLDER / "segformer_masks/20250505_145514/vis_data/config.py",
    "checkpoint": PLOTS_FOLDER / "segformer_masks/20250505_145514/iter_1000_038_2.pth",
}

LATE_MODEL = {
    "config": PLOTS_FOLDER / "segformer_masks/20250505_154220/vis_data/config.py",
    "checkpoint": PLOTS_FOLDER / "segformer_masks/20250505_154220/iter_1000_2430_2.pth",
}
