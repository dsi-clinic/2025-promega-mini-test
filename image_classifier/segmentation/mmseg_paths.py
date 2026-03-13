# analysis/images/segmentation_mmseg/mmseg_paths.py
from __future__ import annotations
from pathlib import Path
import glob

# Import the root paths module once, then read attributes from it.
import config as ROOT

# Canonical values from root
TARGET_WIDTH = ROOT.TARGET_WIDTH
TARGET_HEIGHT = ROOT.TARGET_HEIGHT
PLOTS_FOLDER = ROOT.PLOTS_FOLDER

TRAIN_MANUAL_PROCESSED_DIR = ROOT.TRAIN_MANUAL_PROCESSED_DIR
TRAIN_SPLITS_DIR = ROOT.TRAIN_SPLITS_DIR
PREDICTIONS_DIR = ROOT.PREDICTIONS_DIR
MANUAL_MASKS_DIR = ROOT.MANUAL_MASKS_DIR

# Optional model cfgs (may or may not exist in .env)
CONFIG_FILE_PATH = getattr(ROOT, "CONFIG_FILE_PATH", None)
CHECKPOINT_FILE_PATH = getattr(ROOT, "CHECKPOINT_FILE_PATH", None)

# Suffix used by your plots directory convention
BY_SUFFIX = f"{TARGET_WIDTH}x{TARGET_HEIGHT}"

# Model locations (prefer env overrides when provided)
EARLY_MODEL = {
    "config": (
        CONFIG_FILE_PATH
        or PLOTS_FOLDER
        / f"segformer_masks/{BY_SUFFIX}/october_early/early/vis_data/config.py"
    ),
    "checkpoint": (
        CHECKPOINT_FILE_PATH
        or PLOTS_FOLDER / f"segformer_masks/{BY_SUFFIX}/october_early/iter_1000.pth"
    ),
}
LATE_MODEL = {
    "config": PLOTS_FOLDER
    / f"segformer_masks/{BY_SUFFIX}/october_late/late/vis_data/config.py",
    "checkpoint": PLOTS_FOLDER
    / f"segformer_masks/{BY_SUFFIX}/october_late/iter_1000.pth",
}

# Where inference writes predicted masks
OUTPUT_MASKS_BASE_DIR = PREDICTIONS_DIR

# Training preprocessed outputs
PROCESSED_IMAGES_DIR = TRAIN_MANUAL_PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR = TRAIN_MANUAL_PROCESSED_DIR / "masks"

# Manual mask discovery not needed?
MANUAL_MASK_FOLDERS = [
    Path(p)
    for p in glob.glob(str(MANUAL_MASKS_DIR / "masks-batch-*"))
    if Path(p).is_dir()
]
