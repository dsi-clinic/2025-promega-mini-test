#!/usr/bin/env python3
"""
Task 2: Audit transforms — compute histogram and percentiles (1st, 50th, 99th) on
post-transform tensors for Day 13 vs Day 28 (or Day 30) time-series overlay setup.
Saves summary to day13_15_challenge/audit_transforms_summary.json and optional
histogram plots. No GPU required.
Run: python 02_audit_transforms.py
"""
import sys
from pathlib import Path
import numpy as np

CHALLENGE_DIR = Path(__file__).resolve().parent
ROOT = CHALLENGE_DIR.parent
sys.path.insert(0, str(ROOT))

from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from analysis.images.cnn_lstm.load_split_data import load_split_data
from analysis.images.cnn_lstm.organoid_dataset import OrganoidTimeSeriesDataset


def filter_ids_with_frames_up_to_day(organoid_ids, series_metadata, max_day):
    out = []
    for oid in organoid_ids:
        days = series_metadata.get(oid, {}).get("days", [])
        if any(d <= max_day for d in days):
            out.append(oid)
    return out


def collect_post_transform_stats(dataset, max_samples=50):
    """Collect post-normalization tensor values from dataset (no model)."""
    all_vals = []
    n = min(len(dataset), max_samples)
    for idx in range(n):
        seq, _, _, _, _ = dataset[idx]
        # seq: (T, C, H, W)
        all_vals.append(seq.numpy().ravel())
    if not all_vals:
        return None
    flat = np.concatenate(all_vals)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p1": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "n_pixels": int(flat.size),
        "n_samples": n,
    }


def main():
    print("Step 2: Loading split data...", flush=True)
    split_dir = ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )

    eval_tf = T.Compose([
        T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
    ])

    # Day 13 overlay TS (the problematic setup)
    day13_ids = filter_ids_with_frames_up_to_day(val_ids, series_metadata, 13.0)
    ds13 = OrganoidTimeSeriesDataset(
        day13_ids[:50], series_metadata, data,
        transform=eval_tf, max_day=13.0, image_key="overlay_path",
    )
    stats13 = collect_post_transform_stats(ds13, max_samples=50)

    # Day 28 overlay TS (good day for comparison)
    day28_ids = filter_ids_with_frames_up_to_day(val_ids, series_metadata, 28.0)
    ds28 = OrganoidTimeSeriesDataset(
        day28_ids[:50], series_metadata, data,
        transform=eval_tf, max_day=28.0, image_key="overlay_path",
    )
    stats28 = collect_post_transform_stats(ds28, max_samples=50)

    out = {
        "day13_overlay_ts": stats13,
        "day28_overlay_ts": stats28,
        "note": "Post-dataset normalization (ImageNet mean/std). Compare p1/p50/p99 and min/max for clipping.",
    }
    out_path = CHALLENGE_DIR / "audit_transforms_summary.json"
    import json
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Day 13 (overlay TS):", stats13)
    print("Day 28 (overlay TS):", stats28)
    print("Written to", out_path)

    # Optional: save histogram data for plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for idx, (ds, label) in enumerate([(ds13, "Day 13 overlay TS"), (ds28, "Day 28 overlay TS")]):
            vals = []
            for i in range(min(30, len(ds))):
                seq, _, _, _, _ = ds[i]
                vals.append(seq.numpy().ravel())
            vals = np.concatenate(vals)
            axes[idx].hist(vals.ravel(), bins=80, range=(-3, 3), density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
            axes[idx].set_title(label)
            axes[idx].set_xlabel("Normalized value")
        plt.tight_layout()
        plt.savefig(CHALLENGE_DIR / "audit_transforms_histogram.png", dpi=120)
        plt.close()
        print("Histogram saved to audit_transforms_histogram.png")
    except Exception as e:
        print("Could not save histogram:", e)


if __name__ == "__main__":
    main()
