from __future__ import annotations
from pathlib import Path
import glob

# import all canonical paths from root
from paths import (
    BASE_PATH, OUTPUT_FOLDER, PLOTS_FOLDER, LOGS_FOLDER, NPY_OUTPUTS,
    PREDICTIONS_DIR, SURVEY_RESULTS, META_FILE,
    TARGET_WIDTH, TARGET_HEIGHT, TARGET_SIZE, TARGET_SUFFIX,
    PREPROCESSED_DIR, PROCESSED_PARENT_DIR, PROCESSED_DATA_DIR,
    MANUAL_MASKS_DIR, MANUAL_MAPPING_DIR, MANUAL_PROCESSED_DIR, MANUAL_SPLITS_DIR,
    MAPPING_PROCESSED_TOTAL, CONFIG_FILE_PATH, CHECKPOINT_FILE_PATH,
)

# Some existing folders under plots use "512by384" instead of "512x384".
# Keep the current convention to avoid breaking paths. If you standardize later, change this once.
BY_SUFFIX = f"{TARGET_WIDTH}by{TARGET_HEIGHT}"

EARLY_MODEL = {
    "config":     PLOTS_FOLDER / f"segformer_masks/{BY_SUFFIX}/models/early/vis_data/config.py",
    "checkpoint": PLOTS_FOLDER / f"segformer_masks/{BY_SUFFIX}/models/early/iter_1000.pth",
}
LATE_MODEL = {
    "config":     PLOTS_FOLDER / f"segformer_masks/{BY_SUFFIX}/models/late/vis_data/config.py",
    "checkpoint": PLOTS_FOLDER / f"segformer_masks/{BY_SUFFIX}/models/late/iter_1000.pth",
}

MANUAL_THRESHOLD_MAPPING = MANUAL_MASKS_DIR / "image_mapping_thresholded_and_manual.json"
OUTPUT_MASKS_BASE_DIR = PREDICTIONS_DIR
PROCESSED_IMAGES_DIR  = MANUAL_PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR   = MANUAL_PROCESSED_DIR / "masks"

# dynamic discovery (existing behavior preserved)
MANUAL_MASK_FOLDERS = [
    Path(p) for p in glob.glob(str(MANUAL_MASKS_DIR / "masks-batch-*")) if Path(p).is_dir()
]
