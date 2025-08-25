from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load nearest .env (repo root)
load_dotenv(find_dotenv(usecwd=True), override=False)

def require_env_path(key: str) -> Path:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return Path(v).expanduser()

def require_env_int(key: str) -> int:
    v = os.getenv(key)
    if not v or not v.isdigit():
        raise RuntimeError(f"Missing/invalid int env var: {key}={v!r}")
    return int(v)

# Canonical vars
BASE_PATH               = require_env_path("BASE_PATH")
OUTPUT_FOLDER           = require_env_path("OUTPUT_FOLDER")
PLOTS_FOLDER            = require_env_path("PLOTS_FOLDER")
LOGS_FOLDER             = require_env_path("LOGS_FOLDER")
NPY_OUTPUTS             = require_env_path("NPY_OUTPUTS")
PREDICTIONS_DIR         = require_env_path("PREDICTIONS_DIR")
SURVEY_RESULTS          = require_env_path("SURVEY_RESULTS")
META_FILE               = require_env_path("META_FILE")

TARGET_WIDTH            = require_env_int("TARGET_WIDTH")
TARGET_HEIGHT           = require_env_int("TARGET_HEIGHT")
TARGET_SIZE             = (TARGET_WIDTH, TARGET_HEIGHT)
TARGET_SUFFIX           = f"{TARGET_WIDTH}x{TARGET_HEIGHT}"

PREPROCESSED_DIR        = require_env_path("PREPROCESSED_DIR")
PROCESSED_PARENT_DIR    = require_env_path("PROCESSED_PARENT_DIR")
PROCESSED_DATA_DIR      = require_env_path("PROCESSED_DATA_DIR")
MANUAL_MASKS_DIR        = require_env_path("MANUAL_MASKS_DIR")
MANUAL_MAPPING_DIR      = require_env_path("MANUAL_MAPPING_DIR")
MANUAL_PROCESSED_DIR    = require_env_path("MANUAL_PROCESSED_DIR")
MANUAL_SPLITS_DIR       = require_env_path("MANUAL_SPLITS_DIR")
MAPPING_PROCESSED_TOTAL = require_env_path("MAPPING_PROCESSED_TOTAL")
