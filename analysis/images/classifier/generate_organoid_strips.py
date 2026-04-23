#!/usr/bin/env python3
"""
Generate per-organoid image strips: Day 3 → Day 30, overlay images side by side.

For each organoid in the split CSV, produces a horizontal strip of 11 overlay
images (one per day) saved to an output directory.

Usage:
    python analysis/images/classifier/generate_organoid_strips.py \
        --overlay_dir /net/projects2/promega/2026_04_15_data/intermediate/overlays \
        --splits_csv data/2026_winter_student_splits.csv \
        --out_dir analysis/images/classifier/organoid_strips \
        [--split train|val|test|all] \
        [--max_organoids N]
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DAYS = ["Dy3", "Dy6", "Dy8", "Dy10", "Dy13", "Dy15", "Dy17", "Dy20.5", "Dy24", "Dy26", "Dy28", "Dy30"]

# Overlay filename uses zero-padded two-digit days for single-digit values
DAY_FILENAME_MAP = {
    "Dy3":    "Dy03",
    "Dy6":    "Dy06",
    "Dy8":    "Dy08",
    "Dy10":   "Dy10",
    "Dy13":   "Dy13",
    "Dy15":   "Dy15",
    "Dy17":   "Dy17",
    "Dy20.5": "Dy20.5",
    "Dy24":   "Dy24",
    "Dy26":   "Dy26",
    "Dy28":   "Dy28",
    "Dy30":   "Dy30",
}


def organoid_id_to_filename_base(organoid_id: str) -> str:
    """'BA1 96_1 A10' -> 'BA1_96_1_A10'"""
    return organoid_id.replace(" ", "_")


def find_overlay(overlay_dir: Path, base: str, day_str: str) -> Path | None:
    day_file = DAY_FILENAME_MAP[day_str]
    # e.g. BA1_96_1_Dy03_A10_overlay.png
    # base = 'BA1_96_1_A10'  =>  insert day before well
    # Pattern: {batch}_{day}_{well}_overlay.png
    # base has format BA1_96_1_A10 -> split on last _ to get well
    parts = base.rsplit("_", 1)
    if len(parts) != 2:
        return None
    batch_part, well = parts
    candidate = overlay_dir / f"{batch_part}_{day_file}_{well}_overlay.png"
    return candidate if candidate.exists() else None


def make_strip(overlay_dir: Path, organoid_id: str, split: str, thumb_w: int, thumb_h: int) -> Image.Image | None:
    base = organoid_id_to_filename_base(organoid_id)
    frames = []
    for day in DAYS:
        p = find_overlay(overlay_dir, base, day)
        if p is not None:
            img = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
        else:
            img = Image.new("RGB", (thumb_w, thumb_h), (30, 30, 30))

        # Day label
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, thumb_w, 16], fill=(0, 0, 0, 180))
        draw.text((3, 2), DAY_FILENAME_MAP[day], fill=(255, 255, 255))
        frames.append(img)

    if not any(find_overlay(overlay_dir, base, d) for d in DAYS):
        return None

    gap = 4
    total_w = len(frames) * thumb_w + (len(frames) - 1) * gap
    header_h = 20
    strip = Image.new("RGB", (total_w, thumb_h + header_h), (50, 50, 50))

    draw = ImageDraw.Draw(strip)
    label = f"{organoid_id}  [{split}]"
    draw.text((4, 3), label, fill=(255, 220, 100))

    for i, frame in enumerate(frames):
        x = i * (thumb_w + gap)
        strip.paste(frame, (x, header_h))

    return strip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay_dir", required=True)
    parser.add_argument("--splits_csv", default="data/2026_winter_student_splits.csv")
    parser.add_argument("--out_dir", default="analysis/images/classifier/organoid_strips")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--max_organoids", type=int, default=None)
    parser.add_argument("--thumb_w", type=int, default=160)
    parser.add_argument("--thumb_h", type=int, default=120)
    args = parser.parse_args()

    overlay_dir = Path(args.overlay_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.splits_csv) as f:
        reader = csv.DictReader(f)
        organoids = [(r["organoid_id"], r["split"]) for r in reader]

    if args.split != "all":
        organoids = [(oid, sp) for oid, sp in organoids if sp == args.split]

    if args.max_organoids:
        organoids = organoids[: args.max_organoids]

    print(f"Generating strips for {len(organoids)} organoids -> {out_dir}")
    skipped = 0
    for organoid_id, split in organoids:
        strip = make_strip(overlay_dir, organoid_id, split, args.thumb_w, args.thumb_h)
        if strip is None:
            skipped += 1
            continue
        safe_name = organoid_id.replace(" ", "_")
        out_path = out_dir / f"{safe_name}_strip.png"
        strip.save(out_path)

    print(f"Done. Skipped {skipped} organoids with no overlays found.")
    print(f"Strips saved to {out_dir}/")


if __name__ == "__main__":
    main()
