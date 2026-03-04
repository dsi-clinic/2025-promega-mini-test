#!/usr/bin/env python3
"""
Generate one PNG per organoid: 11 day panels in a row with green contour overlay,
organoid ID as super title. For split organoids, keep only one (e.g. split1).
Output: liya_requested_images/<organoid_id>.png under project root.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

# Run from amanda_temporal root; 2025-promega-mini-test has load_split_data and preprocess
ROOT = Path(__file__).resolve().parent
MINITEST = ROOT / "2025-promega-mini-test"
if str(MINITEST) not in sys.path:
    sys.path.insert(0, str(MINITEST))

from analysis.images.cnn_lstm.load_split_data import _derive_overlay_path, day_str_to_float
from analysis.images.preprocessing.stitched_preprocessing import preprocess_stitched


def _draw_outline_overlay_bgr(img_bgr, mask_bin, color=(0, 255, 0), thickness=2):
    """Draw mask contour on image (BGR)."""
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img_bgr.copy()
    if contours:
        cv2.drawContours(out, contours, contourIdx=-1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def _base_split_key(organoid_id):
    """Key for grouping split organoids: 'BA2 96_2 B5 split1' -> 'BA2 96_2 B5'."""
    return re.sub(r"\s+split\d+\s*$", "", organoid_id.strip(), flags=re.IGNORECASE).strip()


def keep_one_per_split(organoid_ids):
    """When multiple IDs share the same base (e.g. split1, split2), keep only one."""
    seen_base = {}
    out = []
    for oid in organoid_ids:
        base = _base_split_key(oid)
        if base not in seen_base:
            seen_base[base] = oid
            out.append(oid)
    return out


def load_overlay_image(img_path, mask_path, overlay_path=None):
    """Load image with green contour overlay. Prefer overlay_path; else build from image + mask."""
    if overlay_path is None:
        overlay_path = _derive_overlay_path(mask_path) if mask_path else ""
    if overlay_path and Path(overlay_path).exists():
        img = imread(overlay_path)
    elif mask_path and Path(mask_path).exists() and img_path and Path(img_path).exists():
        img = imread(img_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = preprocess_stitched(img, img_path)
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_bin = (mask > 127).astype(np.uint8)
        img = _draw_outline_overlay_bgr(img.astype(np.uint8), mask_bin)
    else:
        if img_path and Path(img_path).exists():
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = preprocess_stitched(img, img_path)
            img = np.ascontiguousarray(img)
        else:
            img = np.zeros((384, 512, 3), dtype=np.uint8)
            img[:] = 128
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def day_label(day_float):
    """Format day for subplot title, e.g. 3.0 -> 'Day 3', 20.5 -> 'Day 20.5'."""
    if day_float == int(day_float):
        return f"Day {int(day_float)}"
    return f"Day {day_float}"


def sanitize_filename(organoid_id):
    """Safe filename: replace spaces and problematic chars."""
    return re.sub(r'[^\w\-.]', '_', organoid_id.strip()).strip("_") or "organoid"


def get_display_label(org_data, votes_override=None, organoid_id=None):
    """
    Show label only for clear majority: 5-0, 0-5, 4-1, or 1-4 (Acceptable vs Not Acceptable); otherwise return "".
    votes_override: optional dict organoid_id -> {"votes": {"Acceptable": n, "Not Acceptable": m}, "label": "..."}.
    """
    entry = votes_override.get(organoid_id, {}) if votes_override and organoid_id else {}
    votes = entry.get("votes") or org_data.get("votes") or {}
    n_accept = int(votes.get("Acceptable", 0))
    n_not = int(votes.get("Not Acceptable", 0))
    total = n_accept + n_not
    if total != 5:
        return ""
    if n_accept >= 4:
        return "Acceptable"
    if n_not >= 4:
        return "Not Acceptable"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate one PNG per organoid (day strip with green overlay)")
    parser.add_argument("--splits_dir", type=Path, default=ROOT / "data_splits",
                        help="Directory containing both_train_amanda_style.json etc.")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "liya_requested_images",
                        help="Output directory for PNGs (default: liya_requested_images)")
    parser.add_argument("--train", default="both_train_amanda_style.json", help="Train split filename")
    parser.add_argument("--val", default="both_val_amanda_style.json", help="Val split filename")
    parser.add_argument("--test", default="both_test_amanda_style.json", help="Test split filename")
    parser.add_argument("--max_day", type=float, default=30.0, help="Include timepoints with day <= this")
    args = parser.parse_args()

    train_path = args.splits_dir / args.train
    val_path = args.splits_dir / args.val
    test_path = args.splits_dir / args.test
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        print(f"Missing split files in {args.splits_dir}")
        sys.exit(1)

    # Load raw splits so we have per-organoid paths (avoids data[] key collision for split1/split2)
    with open(train_path) as f:
        train_split = json.load(f)
    with open(val_path) as f:
        val_split = json.load(f)
    with open(test_path) as f:
        test_split = json.load(f)
    all_splits = {**train_split, **val_split, **test_split}
    all_ids = list(all_splits.keys())
    organoid_ids = keep_one_per_split(all_ids)
    print(f"Organoids after keeping one per split: {len(organoid_ids)} (from {len(all_ids)} total)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir.resolve()}")

    for i, organoid_id in enumerate(organoid_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(organoid_ids)}] {organoid_id}")
        org_data = all_splits.get(organoid_id)
        if not org_data:
            continue
        timepoints = org_data.get("timepoints", {})
        sorted_day_strs = sorted(timepoints.keys(), key=lambda d: day_str_to_float(d))
        panels = []
        for day_str in sorted_day_strs:
            day_float = day_str_to_float(day_str)
            if day_float > args.max_day:
                continue
            tp = timepoints[day_str]
            img_path = tp.get("img_path", "")
            mask_path = tp.get("mask_path", "")
            overlay_path = tp.get("overlay_path") or (_derive_overlay_path(mask_path) if mask_path else "")
            try:
                img = load_overlay_image(img_path, mask_path, overlay_path)
                panels.append((day_float, img))
            except Exception as e:
                print(f"    Skip {day_str}: {e}")
                continue

        if not panels:
            print(f"    No panels for {organoid_id}, skipping")
            continue

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.2))
        if n == 1:
            axes = [axes]
        fig.suptitle(organoid_id, fontsize=12, fontweight="bold", y=1.02)
        for ax, (day, img) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(day_label(day), fontsize=10)
            ax.axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_path = args.output_dir / f"{sanitize_filename(organoid_id)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"Done. Wrote up to {len(organoid_ids)} PNGs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
