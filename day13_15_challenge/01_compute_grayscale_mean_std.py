#!/usr/bin/env python3
"""
Task 1a: Compute mean and std from grayscale image pixels (not overlay) over
training frames for Day 13 and Day 15. Saves to day13_15_challenge/grayscale_mean_std_*.npy.
Run from repo root or from day13_15_challenge; uses 2025-promega-mini-test/data_splits.
"""
import sys
from pathlib import Path

print("Step 1: Loading split data...", flush=True)

CHALLENGE_DIR = Path(__file__).resolve().parent
ROOT = CHALLENGE_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CHALLENGE_DIR))

from analysis.images.cnn_lstm.load_split_data import load_split_data
from dataset_grayscale_norm import compute_grayscale_mean_std

DAYS = [13.0, 15.0]


def main():
    split_dir = ROOT / "data_splits"
    print(f"Split dir: {split_dir}", flush=True)
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    for max_day in DAYS:
        mean_3, std_3 = compute_grayscale_mean_std(train_ids, series_metadata, data, max_day)
        out = {"mean": mean_3, "std": std_3, "max_day": max_day}
        out_path = CHALLENGE_DIR / f"grayscale_mean_std_day{int(max_day)}.npy"
        import numpy as np
        np.save(out_path, out)
        print(f"Day {max_day}: mean={mean_3}, std={std_3} -> {out_path}")


if __name__ == "__main__":
    main()
