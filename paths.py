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


# Manual processed data and splits
MANUAL_MAPPING_DIR       = require_env_path("MANUAL_MAPPING_DIR")
MANUAL_PROCESSED_DIR     = require_env_path("MANUAL_PROCESSED_DIR")
PROCESSED_IMAGES_DIR     = MANUAL_PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR      = MANUAL_PROCESSED_DIR / "masks"
MAPPING_PROCESSED_TOTAL  = require_env_path("MAPPING_PROCESSED_TOTAL")
PROCESSED_DATA_DIR = require_env_path("PROCESSED_DATA_DIR")
MANUAL_SPLITS_DIR        = MANUAL_PROCESSED_DIR / "split"

