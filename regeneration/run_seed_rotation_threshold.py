#!/usr/bin/env python3
"""
Thin wrapper around run_threshold_study.py that accepts a custom --split_dir.

Imports the *same* training functions (run_per_day, run_effnet_ts_accumulated)
and CSV export logic — zero model-logic changes.  Only the path to split JSONs
differs from the original script.

Usage:
    python regeneration/run_seed_rotation_threshold.py \
        --split_dir regeneration/seed_rotation_splits/s7/data_splits \
        --output_dir regeneration/seed_rotation_s7
"""
import os
import sys
import csv
import argparse
from pathlib import Path

COMPARISON_RUNS = Path(__file__).resolve().parent.parent / "comparison_runs"
ROOT = Path(__file__).resolve().parent.parent / "2025-promega-mini-test"
if not ROOT.is_dir():
    ROOT = COMPARISON_RUNS.parent

env_file = ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v

for key, default in [
    ("BASE_PATH", "/net/projects2/promega/data-analysis"),
    ("OUTPUT_FOLDER", "/net/projects2/promega/data-analysis/output"),
    ("RAW_IMAGE_DATA", "/net/projects2/promega/data-analysis/raw-images"),
    ("IMAGE_VERIFICATION_FORM", "/net/projects2/promega/data-analysis/image-verification-form.json"),
    ("PLOTS_FOLDER", "/net/projects2/promega/data-analysis/plots"),
    ("LOGS_FOLDER", "/net/projects2/promega/data-analysis/logs"),
    ("NPY_OUTPUTS", "/net/projects2/promega/data-analysis/npy-outputs"),
    ("PREDICTIONS_DIR", "/net/projects2/promega/data-analysis/predictions"),
    ("SURVEY_RESULTS", "/net/projects2/promega/data-analysis/survey-results"),
    ("MANUAL_MASKS_DIR", "/net/projects2/promega/data-analysis/manual-masks"),
    ("META_FILE", "/net/projects2/promega/data-analysis/metadata.json"),
    ("RAW_IMAGE_MAPPING_JSON", "/net/projects2/promega/data-analysis/image-mapping.json"),
    ("TARGET_WIDTH", "512"),
    ("TARGET_HEIGHT", "384"),
    ("TRAIN_RESIZED_DIR", "/net/projects2/promega/data-analysis/train-resized"),
    ("TRAIN_MANUAL_MAPPING_DIR", "/net/projects2/promega/data-analysis/train-mapping"),
    ("TRAIN_MANUAL_PROCESSED_DIR", "/net/projects2/promega/data-analysis/train-processed"),
    ("TRAIN_SPLITS_DIR", "/net/projects2/promega/data-analysis/train-splits"),
    ("INFER_RESIZED_DIR", "/net/projects2/promega/data-analysis/infer-resized"),
    ("INFER_MAPPING_TOTAL_JSON", "/net/projects2/promega/data-analysis/infer-mapping.json"),
    ("MANUAL_THRESHOLD_MAPPING", "/net/projects2/promega/data-analysis/threshold-mapping.json"),
]:
    os.environ.setdefault(key, default)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMPARISON_RUNS))

from analysis.images.cnn_lstm.load_split_data import load_split_data
from run_per_day_study import DAYS_11, run_per_day, run_effnet_ts_accumulated
from export_metrics_csv import collect_rows, CSV_COLUMNS

import torch
import json


def existing_results_with_thresholds(base_dir):
    """Set of (model_type, day) that already have results.json with threshold_results."""
    done = set()
    for model_type in ("per_day", "effnet_ts"):
        path = base_dir / model_type
        if not path.exists():
            continue
        for day_dir in path.iterdir():
            if not day_dir.is_dir():
                continue
            res_file = day_dir / "results.json"
            if not res_file.exists():
                continue
            try:
                with open(res_file) as f:
                    data = json.load(f)
                if "threshold_results" not in data:
                    continue
                day = data.get("day")
                if day is not None:
                    done.add((model_type, float(day)))
            except Exception:
                pass
    return done


def main():
    p = argparse.ArgumentParser(description="Seed-rotation threshold study (custom split dir)")
    p.add_argument("--split_dir", required=True,
                   help="Dir containing both_train_base.json, both_val_base.json, both_test_base.json")
    p.add_argument("--output_dir", required=True,
                   help="Dir for per_day_study_overlay/ and CSV")
    p.add_argument("--models", default="both", choices=["both", "per_day", "effnet_ts"])
    args = p.parse_args()

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    base_dir = output_dir / "per_day_study_overlay"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seed-rotation threshold study")
    print(f"  split_dir : {split_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  models    : {args.models}")

    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    print(f"  data: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_key, use_rgb_mask, in_channels = "overlay_path", False, 3
    input_rgb_mask = use_rgb_mask

    existing = existing_results_with_thresholds(base_dir)
    models_to_run = ("per_day", "effnet_ts") if args.models == "both" else (args.models,)
    for model_type in models_to_run:
        for day in DAYS_11:
            if (model_type, day) in existing:
                print(f"\n--- {model_type} day={day} --- skip (already done)")
                continue
            print(f"\n--- {model_type} day={day} ---")
            if model_type == "per_day":
                run_per_day(
                    day, train_ids, val_ids, test_ids, series_metadata, data, device, base_dir,
                    image_key=image_key, use_rgb_mask=use_rgb_mask, in_channels=in_channels,
                    save_model=False,
                )
            else:
                run_effnet_ts_accumulated(
                    day, train_ids, val_ids, test_ids, series_metadata, data, device, base_dir,
                    image_key=image_key, input_rgb_mask=input_rgb_mask, in_channels=in_channels,
                    save_model=False,
                )

    rows = collect_rows(include_kfold=False, setup_filter="overlay", overlay_base=base_dir)
    out_csv = output_dir / "overlay_threshold_study_results.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {out_csv}")
    else:
        print("No rows to export.")


if __name__ == "__main__":
    main()
