#!/usr/bin/env python3
"""
Single entry point: produce the final CSV.
Runs backfill (add threshold_results to existing results, no training) if needed,
then exports overlay (per_day + effnet_ts) to overlay_threshold_study_results.csv.
Final output: comparison_runs/overlay_threshold_study_results.csv
"""

import os
import sys
import csv
from pathlib import Path

COMPARISON_RUNS = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = COMPARISON_RUNS / "overlay_threshold_study_results.csv"

if (REPO_ROOT / ".env").exists():
    with open(REPO_ROOT / ".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
for k, v in [
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
]:
    os.environ.setdefault(k, v)
sys.path.insert(0, str(COMPARISON_RUNS))


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Produce final CSV: overlay_threshold_study_results.csv"
    )
    p.add_argument(
        "--export_only",
        action="store_true",
        help="Skip backfill; only export existing results to CSV",
    )
    args = p.parse_args()

    if not args.export_only:
        # 1) Backfill threshold_results into existing overlay results (inference only)
        try:
            from add_threshold_results_to_existing import main as backfill_main

            print("Backfilling threshold_results (inference only)...")
            backfill_main()
        except Exception as e:
            print("Backfill:", e)

    # 2) Export overlay to CSV
    from export_metrics_csv import collect_rows, CSV_COLUMNS

    rows = collect_rows(include_kfold=False, setup_filter="overlay")
    if not rows:
        print("No overlay results found. Run run_threshold_study.py first.")
        sys.exit(1)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print("Final output:", str(OUT_CSV))


if __name__ == "__main__":
    main()
