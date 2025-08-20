# src/proj/config.py
from __future__ import annotations
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

# Load .env once, do not override explicit env
load_dotenv(find_dotenv(), override=False)

def repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p.parent / "pyproject.toml").exists() or (p.parent / ".git").exists():
            return p.parent
        p = p.parent
    return Path.cwd()

ROOT = repo_root()

def _get_path(key: str, default_rel: str | None = None) -> Path:
    v = os.getenv(key)
    if v:
        return Path(v).expanduser()
    if default_rel is not None:
        return ROOT / default_rel
    raise RuntimeError(f"Missing required env var: {key}")

def _get_int(key: str, default: int | None = None) -> int:
    v = os.getenv(key)
    if v is None:
        if default is None:
            raise RuntimeError(f"Missing required env var: {key}")
        return default
    return int(v)

# ---- Canonical config (single source of truth) ----
BASE_PATH      = _get_path("BASE_PATH", "data")
OUTPUT_FOLDER  = _get_path("OUTPUT_FOLDER", "output")
PLOTS_FOLDER   = _get_path("PLOTS_FOLDER", "plots")
LOGS_FOLDER    = _get_path("LOGS_FOLDER", "logs")
NPY_OUTPUTS    = _get_path("NPY_OUTPUTS", "npy_outputs")
PREDICTIONS_DIR= _get_path("PREDICTIONS_DIR", "predictions")

META_FILE      = _get_path("META_FILE", "Sample-Tracing.xlsx")
SURVEY_RESULTS = _get_path("SURVEY_RESULTS", "results_surveys")

TARGET_WIDTH   = _get_int("TARGET_WIDTH", 512)
TARGET_HEIGHT  = _get_int("TARGET_HEIGHT", 384)
TARGET_SIZE    = (TARGET_WIDTH, TARGET_HEIGHT)
TARGET_SUFFIX  = f"{TARGET_WIDTH}x{TARGET_HEIGHT}"

ORIGINAL_MAPPING   = _get_path("ORIGINAL_MAPPING", f"output/image_mapping.json")
PREPROCESSED_DIR   = _get_path("PREPROCESSED_DIR", f"output/processed_dataset_{TARGET_SUFFIX}")
PROCESSED_PARENT_DIR=_get_path("PROCESSED_PARENT_DIR", "output")
PROCESSED_DATA_DIR = _get_path("PROCESSED_DATA_DIR", f"{PREPROCESSED_DIR}/auto_processed")

MANUAL_MASKS_DIR      = _get_path("MANUAL_MASKS_DIR", "manual_masks")
MANUAL_MAPPING_DIR    = _get_path("MANUAL_MAPPING_DIR", f"{PREPROCESSED_DIR}/manual_mappings")
MANUAL_PROCESSED_DIR  = _get_path("MANUAL_PROCESSED_DIR", f"{MANUAL_MAPPING_DIR}/processed_{TARGET_SUFFIX}")
MANUAL_SPLITS_DIR     = _get_path("MANUAL_SPLITS_DIR", f"{MANUAL_PROCESSED_DIR}/split")
MAPPING_PROCESSED_TOTAL = _get_path("MAPPING_PROCESSED_TOTAL", f"{MANUAL_PROCESSED_DIR}/mapping_processed_total_{TARGET_SUFFIX}.json")

CONFIG_FILE_PATH   = _get_path("CONFIG_FILE_PATH", f"{PLOTS_FOLDER}/segformer_masks/config.py")
CHECKPOINT_FILE_PATH = _get_path("CHECKPOINT_FILE_PATH", f"{PLOTS_FOLDER}/segformer_masks/iter_1000.pth")

# Quick sanity (fail fast on obvious mistakes)
def sanity_check():
    assert TARGET_WIDTH > 0 and TARGET_HEIGHT > 0
    # Optional: assert META_FILE.exists(), etc. If too strict for dev, comment out.
