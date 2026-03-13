#!/usr/bin/env python3
"""
Threshold study: overlay only (best setup for both models), per_day + effnet_ts, all 11 days.
Training does NOT use thresholds — we train once per (model, day). Thresholds (0.5, 0.6, 0.7, 0.8, 0.9, optimal)
are applied only at evaluation time to see how they impact predictions (balanced_acc, etc.) each day.
Runs all 22 runs (2 models × 11 days), then exports one CSV: overlay_threshold_study_results.csv.
"""

import os
import sys
import csv
from pathlib import Path

COMPARISON_RUNS = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
# Load .env first
env_file = REPO_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
# Defaults matching submit_threshold_study.slurm so config.py never misses a var
for key, default in [
    ("BASE_PATH", "/net/projects2/promega/data-analysis"),
    ("OUTPUT_FOLDER", "/net/projects2/promega/data-analysis/output"),
    ("RAW_IMAGE_DATA", "/net/projects2/promega/data-analysis/raw-images"),
    (
        "IMAGE_VERIFICATION_FORM",
        "/net/projects2/promega/data-analysis/image-verification-form.json",
    ),
    ("PLOTS_FOLDER", "/net/projects2/promega/data-analysis/plots"),
    ("LOGS_FOLDER", "/net/projects2/promega/data-analysis/logs"),
    ("NPY_OUTPUTS", "/net/projects2/promega/data-analysis/npy-outputs"),
    ("PREDICTIONS_DIR", "/net/projects2/promega/data-analysis/predictions"),
    ("SURVEY_RESULTS", "/net/projects2/promega/data-analysis/survey-results"),
    ("MANUAL_MASKS_DIR", "/net/projects2/promega/data-analysis/manual-masks"),
    ("META_FILE", "/net/projects2/promega/data-analysis/metadata.json"),
    (
        "RAW_IMAGE_MAPPING_JSON",
        "/net/projects2/promega/data-analysis/image-mapping.json",
    ),
    ("TARGET_WIDTH", "512"),
    ("TARGET_HEIGHT", "384"),
    ("TRAIN_RESIZED_DIR", "/net/projects2/promega/data-analysis/train-resized"),
    ("TRAIN_MANUAL_MAPPING_DIR", "/net/projects2/promega/data-analysis/train-mapping"),
    (
        "TRAIN_MANUAL_PROCESSED_DIR",
        "/net/projects2/promega/data-analysis/train-processed",
    ),
    ("TRAIN_SPLITS_DIR", "/net/projects2/promega/data-analysis/train-splits"),
    ("INFER_RESIZED_DIR", "/net/projects2/promega/data-analysis/infer-resized"),
    (
        "INFER_MAPPING_TOTAL_JSON",
        "/net/projects2/promega/data-analysis/infer-mapping.json",
    ),
    (
        "MANUAL_THRESHOLD_MAPPING",
        "/net/projects2/promega/data-analysis/threshold-mapping.json",
    ),
]:
    os.environ.setdefault(key, default)

sys.path.insert(0, str(REPO_ROOT))

from image_classifier.cnn_lstm.load_split_data import load_split_data

# Import runner and config from run_per_day_study
sys.path.insert(0, str(COMPARISON_RUNS))
from run_per_day_study import (
    DAYS_11,
    run_per_day,
    run_effnet_ts_accumulated,
    run_effnet_tchange_accumulated,
)
from export_metrics_csv import collect_rows, CSV_COLUMNS, CSV_COLUMNS_WITH_SPLIT
from export_predictions_csv import export_predictions

import torch
import json


def existing_results_with_thresholds(base_dir):
    """Set of (model_type, day) that already have results.json with threshold_results."""
    done = set()
    for model_type in ("per_day", "effnet_ts", "effnet_tchange"):
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
    import argparse

    p = argparse.ArgumentParser(
        description="Overlay-only threshold study: per_day + effnet_ts, 11 days, then export CSV"
    )
    p.add_argument(
        "--save_model",
        action="store_true",
        help="Save .pth checkpoints (default: no, to keep disk <2GB)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV (default: comparison_runs/overlay_threshold_study_results.csv)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="If set, write per_day_study_overlay and CSV under this dir (for parallel runs)",
    )
    p.add_argument(
        "--models",
        default="both",
        choices=["both", "per_day", "effnet_ts", "effnet_tchange"],
        help="Which models to run (default: both)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results.json already exists (needed for adding train metrics to existing dirs)",
    )
    p.add_argument(
        "--export_predictions",
        action="store_true",
        help="Export per-organoid predictions to predictions_Dy*.csv (organoid_id, split, day, y_true, y_pred, y_score)",
    )
    args = p.parse_args()
    save_model = args.save_model
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        base_dir = output_dir / "per_day_study_overlay"
        out_csv = output_dir / "overlay_threshold_study_results.csv"
    else:
        base_dir = COMPARISON_RUNS / "per_day_study_overlay"
        out_csv = (
            Path(args.output)
            if args.output
            else COMPARISON_RUNS / "overlay_threshold_study_results.csv"
        )
    base_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_key, use_rgb_mask, in_channels = "overlay_path", False, 3
    input_rgb_mask = use_rgb_mask

    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    print(f"Overlay threshold study: 2 models × 11 days, save_model={save_model}")
    print(f"  data: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    existing = set() if args.force else existing_results_with_thresholds(base_dir)
    models_to_run = (
        ("per_day", "effnet_ts") if args.models == "both" else (args.models,)
    )
    for model_type in models_to_run:
        for day in DAYS_11:
            if (model_type, day) in existing:
                print(f"\n--- {model_type} day={day} --- skip")
                continue
            print(f"\n--- {model_type} day={day} ---")
            if model_type == "per_day":
                run_per_day(
                    day,
                    train_ids,
                    val_ids,
                    test_ids,
                    series_metadata,
                    data,
                    device,
                    base_dir,
                    image_key=image_key,
                    use_rgb_mask=use_rgb_mask,
                    in_channels=in_channels,
                    save_model=save_model,
                )
            elif model_type == "effnet_tchange":
                run_effnet_tchange_accumulated(
                    day,
                    train_ids,
                    val_ids,
                    test_ids,
                    series_metadata,
                    data,
                    device,
                    base_dir,
                    image_key=image_key,
                    input_rgb_mask=input_rgb_mask,
                    in_channels=in_channels,
                    save_model=save_model,
                )
            else:
                run_effnet_ts_accumulated(
                    day,
                    train_ids,
                    val_ids,
                    test_ids,
                    series_metadata,
                    data,
                    device,
                    base_dir,
                    image_key=image_key,
                    input_rgb_mask=input_rgb_mask,
                    in_channels=in_channels,
                    save_model=save_model,
                )

    # Export CSV: overlay only, two models, all days × thresholds (test-only, backward compat)
    rows = collect_rows(
        include_kfold=False, setup_filter="overlay", overlay_base=base_dir
    )
    if not rows:
        print("No results to export.")
        return
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")

    # Export all-splits CSV (train+val+test) alongside the test-only CSV
    all_rows = collect_rows(
        include_kfold=False,
        setup_filter="overlay",
        overlay_base=base_dir,
        splits=["train", "val", "test"],
    )
    if all_rows:
        all_csv = out_csv.parent / out_csv.name.replace(".csv", "_all_splits.csv")
        with open(all_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=CSV_COLUMNS_WITH_SPLIT, extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {all_csv}")

    # Optional: export per-organoid predictions (effnet_ts/effnet_tchange only; no img_path)
    if args.export_predictions:
        pred_dir = (output_dir if output_dir else COMPARISON_RUNS) / "predictions"
        written = export_predictions(base_dir, output_dir=pred_dir, model_filter=None)
        if written:
            print(f"Exported {len(written)} prediction CSV(s) to {pred_dir}")


if __name__ == "__main__":
    main()
