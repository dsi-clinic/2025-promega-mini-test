# paths.py (ROOT)
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load nearest .env from current working dir upward
load_dotenv(find_dotenv(usecwd=True), override=True)

def require_env_path(key: str) -> Path:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return Path(v).expanduser()

def optional_env_path(key: str) -> Path | None:
    v = os.getenv(key)
    return Path(v).expanduser() if v else None

def require_env_int(key: str) -> int:
    v = os.getenv(key)
    if not v or not str(v).isdigit():
        raise RuntimeError(f"Missing/invalid int env var: {key}={v!r}")
    return int(v)

# ---- Canonical base ----
BASE_PATH         = require_env_path("BASE_PATH")
OUTPUT_FOLDER     = require_env_path("OUTPUT_FOLDER")
PLOTS_FOLDER      = require_env_path("PLOTS_FOLDER")
LOGS_FOLDER       = require_env_path("LOGS_FOLDER")
NPY_OUTPUTS       = require_env_path("NPY_OUTPUTS")
PREDICTIONS_DIR   = require_env_path("PREDICTIONS_DIR")
SURVEY_RESULTS    = require_env_path("SURVEY_RESULTS")
MANUAL_MASKS_DIR  = require_env_path("MANUAL_MASKS_DIR")
META_FILE         = require_env_path("META_FILE")

# The raw, first-stage image mapping JSON
RAW_IMAGE_MAPPING_JSON = require_env_path("RAW_IMAGE_MAPPING_JSON")

# Optional model config/checkpoint from env (used by mmseg; can be None)
CONFIG_FILE_PATH    = optional_env_path("CONFIG_FILE_PATH")
CHECKPOINT_FILE_PATH= optional_env_path("CHECKPOINT_FILE_PATH")

# ---- Sizes ----
TARGET_WIDTH   = require_env_int("TARGET_WIDTH")
TARGET_HEIGHT  = require_env_int("TARGET_HEIGHT")
TARGET_SIZE    = (TARGET_WIDTH, TARGET_HEIGHT)
TARGET_SUFFIX  = f"{TARGET_WIDTH}x{TARGET_HEIGHT}"

# ---- Training layout ----
TRAIN_RESIZED_DIR          = require_env_path("TRAIN_RESIZED_DIR")
TRAIN_MANUAL_MAPPING_DIR   = require_env_path("TRAIN_MANUAL_MAPPING_DIR")
TRAIN_MANUAL_PROCESSED_DIR = require_env_path("TRAIN_MANUAL_PROCESSED_DIR")
TRAIN_SPLITS_DIR           = require_env_path("TRAIN_SPLITS_DIR")

# ---- Inference layout ----
INFER_RESIZED_DIR        = require_env_path("INFER_RESIZED_DIR")
INFER_MAPPING_TOTAL_JSON = require_env_path("INFER_MAPPING_TOTAL_JSON")
INFER_AUTO_PROCESSED_DIR = INFER_RESIZED_DIR / "auto_processed"

# ---- Shared artifacts (centralized file names) ----
MANUAL_THRESHOLD_MAPPING = require_env_path("MANUAL_THRESHOLD_MAPPING")

# ---- Surveys & metabolites (with sensible defaults) ----
METABOLITE_DATA_DIR     = optional_env_path("METABOLITE_DATA_DIR") or (BASE_PATH / "metabolite_data")
METABOLITE_SOURCE_XLSX  = optional_env_path("METABOLITE_SOURCE_XLSX") or (METABOLITE_DATA_DIR / "metabolite_data_07_23_25.xlsx")
METABOLITE_MAP_JSON     = optional_env_path("METABOLITE_MAP_JSON") or (METABOLITE_DATA_DIR / "metabolite_map.json")
SURVEY_AGGREGATED_JSON  = optional_env_path("SURVEY_AGGREGATED_JSON") or (SURVEY_RESULTS / "organoid_surveys_aggregated.json")

# ---- Back-compat shims (do NOT use in new code) ----
PREPROCESSED_DIR        = TRAIN_RESIZED_DIR
PROCESSED_PARENT_DIR    = OUTPUT_FOLDER
PROCESSED_DATA_DIR      = INFER_AUTO_PROCESSED_DIR
MANUAL_MAPPING_DIR      = TRAIN_MANUAL_MAPPING_DIR
MANUAL_PROCESSED_DIR    = TRAIN_MANUAL_PROCESSED_DIR
MANUAL_SPLITS_DIR       = TRAIN_SPLITS_DIR
MAPPING_PROCESSED_TOTAL = INFER_MAPPING_TOTAL_JSON
