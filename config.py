# config.py - Consolidated configuration (merged paths.py and config.py)
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

def _get_path(key: str, default_rel: str | None = None) -> Path:
    """Get path from env with optional default relative to repo root"""
    v = os.getenv(key)
    if v:
        return Path(v).expanduser()
    if default_rel is not None:
        return repo_root() / default_rel
    raise RuntimeError(f"Missing required env var: {key}")

def _get_int(key: str, default: int | None = None) -> int:
    """Get int from env with optional default"""
    v = os.getenv(key)
    if v is None:
        if default is None:
            raise RuntimeError(f"Missing required env var: {key}")
        return default
    return int(v)

def repo_root() -> Path:
    """Find repository root containing paths.py and .env"""
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p.parent / ".git").exists() or (p.parent / ".env").exists():
            return p.parent
        p = p.parent
    return Path.cwd()

ROOT = repo_root()

# ---- Canonical base paths ----
BASE_PATH         = require_env_path("BASE_PATH")
OUTPUT_FOLDER     = require_env_path("OUTPUT_FOLDER")
PLOTS_FOLDER      = require_env_path("PLOTS_FOLDER")
LOGS_FOLDER       = require_env_path("LOGS_FOLDER")
NPY_OUTPUTS       = require_env_path("NPY_OUTPUTS")
PREDICTIONS_DIR   = require_env_path("PREDICTIONS_DIR")
SURVEY_RESULTS    = require_env_path("SURVEY_RESULTS")
MANUAL_MASKS_DIR  = require_env_path("MANUAL_MASKS_DIR")
META_FILE         = require_env_path("META_FILE")
ALL_DATA_JSON     = OUTPUT_FOLDER / "all_data.json"

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


# ---- Legacy aliases for backwards compatibility ----
ORIGINAL_MAPPING = RAW_IMAGE_MAPPING_JSON

# ---- Validation ----
def validate_config():
    """Validate configuration and create necessary directories"""
    try:
        # Create essential directories
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
        LOGS_FOLDER.mkdir(parents=True, exist_ok=True)
        
        # Validate size parameters
        assert TARGET_WIDTH > 0 and TARGET_HEIGHT > 0, "Invalid target dimensions"
        
        # Validate critical paths
        if not BASE_PATH.exists():
            raise RuntimeError(f"BASE_PATH does not exist: {BASE_PATH}")
            
        return True
    except Exception as e:
        raise RuntimeError(f"Configuration validation failed: {e}")

# Legacy function name for backwards compatibility
def sanity_check():
    """Legacy alias for validate_config()"""
    return validate_config()
