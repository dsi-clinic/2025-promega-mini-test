#!/usr/bin/env python3
"""
5-fold CV per-day study: fix train/val by fold, same test set.
Runs: 5 folds × 3 setups (rgb, overlay, rgb_mask) × 3 models × 11 days = 495.
Output: per_day_study_kfold/<setup>/fold_<f>/<model>/day_<d>/results.json
Use --array_index 0-494 (one job per run).
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Same env/path as run_per_day_study
COMPARISON_RUNS = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
env_file = REPO_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                import os

                os.environ[k] = v
for k, v in [
    ("BASE_PATH", "/net/projects2/promega/data-analysis"),
    ("OUTPUT_FOLDER", "/net/projects2/promega/data-analysis/output"),
]:
    import os

    os.environ.setdefault(k, v)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COMPARISON_RUNS))

from image_classifier.cnn_lstm.load_split_data import load_split_data
from run_per_day_study import (
    DAYS_11,
    filter_ids_with_frames_up_to_day,
    run_per_day,
    run_cnn_lstm_accumulated,
    run_effnet_ts_accumulated,
)

K = 5
SETUPS = ["rgb", "overlay", "rgb_mask"]  # index 0, 1, 2
MODELS = ["per_day", "cnn_lstm", "effnet_ts"]
N_SETUPS = len(SETUPS)
N_MODELS = len(MODELS)
N_DAYS = len(DAYS_11)
TOTAL = K * N_SETUPS * N_MODELS * N_DAYS  # 495


def make_kfold_splits(train_ids, val_ids, series_metadata, k=5, seed=42):
    """Stratified k-fold from train+val. Returns list of (train_ids, val_ids). Test unchanged."""
    rng = np.random.default_rng(seed)
    all_ids = list(train_ids) + list(val_ids)
    good = lambda oid: str(
        series_metadata.get(oid, {}).get("label", "")
    ).strip().lower() in ("good", "acceptable", "accepted")
    good_ids = [i for i in all_ids if good(i)]
    bad_ids = [i for i in all_ids if not good(i)]
    rng.shuffle(good_ids)
    rng.shuffle(bad_ids)
    folds = [[] for _ in range(k)]
    for i, oid in enumerate(good_ids):
        folds[i % k].append(oid)
    for i, oid in enumerate(bad_ids):
        folds[i % k].append(oid)
    out = []
    for val_fold_idx in range(k):
        val_ids_f = folds[val_fold_idx]
        train_ids_f = [oid for j in range(k) if j != val_fold_idx for oid in folds[j]]
        out.append((train_ids_f, val_ids_f))
    return out


def decode_index(idx):
    """idx in [0, TOTAL-1] -> (fold, setup_name, model_type, day)."""
    if idx < 0 or idx >= TOTAL:
        raise ValueError(f"array_index must be in [0, {TOTAL - 1}]")
    rest = idx
    fold = rest // (N_SETUPS * N_MODELS * N_DAYS)
    rest %= N_SETUPS * N_MODELS * N_DAYS
    setup_idx = rest // (N_MODELS * N_DAYS)
    rest %= N_MODELS * N_DAYS
    model_idx = rest // N_DAYS
    day_idx = rest % N_DAYS
    return fold, SETUPS[setup_idx], MODELS[model_idx], DAYS_11[day_idx]


def main():
    parser = argparse.ArgumentParser(
        description="5-fold CV per-day study, one run per array index"
    )
    parser.add_argument(
        "--array_index", type=int, required=True, help=f"0 to {TOTAL - 1}"
    )
    args = parser.parse_args()

    fold, setup_name, model_type, day = decode_index(args.array_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = COMPARISON_RUNS / "per_day_study_kfold" / setup_name / f"fold_{fold}"

    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    folds = make_kfold_splits(train_ids, val_ids, series_metadata, k=K)
    train_ids_f, val_ids_f = folds[fold]

    if setup_name == "overlay":
        image_key, use_rgb_mask, in_channels = "overlay_path", False, 3
    elif setup_name == "rgb_mask":
        image_key, use_rgb_mask, in_channels = "img_path", True, 4
    else:
        image_key, use_rgb_mask, in_channels = "img_path", False, 3
    input_rgb_mask = use_rgb_mask

    print(
        f"kfold run: fold={fold} setup={setup_name} model={model_type} day={day} array_index={args.array_index}"
    )
    print(f"  train={len(train_ids_f)} val={len(val_ids_f)} test={len(test_ids)}")

    if model_type == "per_day":
        run_per_day(
            day,
            train_ids_f,
            val_ids_f,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            image_key=image_key,
            use_rgb_mask=use_rgb_mask,
            in_channels=in_channels,
        )
    elif model_type == "cnn_lstm":
        run_cnn_lstm_accumulated(
            day,
            train_ids_f,
            val_ids_f,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            image_key=image_key,
            input_rgb_mask=input_rgb_mask,
            in_channels=in_channels,
        )
    else:
        run_effnet_ts_accumulated(
            day,
            train_ids_f,
            val_ids_f,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            image_key=image_key,
            input_rgb_mask=input_rgb_mask,
            in_channels=in_channels,
        )
    print("Done.")


if __name__ == "__main__":
    main()
