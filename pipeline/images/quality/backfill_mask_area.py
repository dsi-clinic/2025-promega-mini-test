#!/usr/bin/env python3
"""Backfill ``mask_area_px`` / ``mask_area_um2`` into an existing all_data.json.

Step 11 (``mask_edge_fraction``) now writes these fields into the image mapping,
and the merge (Step 16) copies them into ``record["images"]``. This script
produces the SAME values for an all_data.json that predates that wiring, without
rerunning the image-resize chain (Steps 14-16): each record already carries
``images.mask_path`` and ``images.um_per_px.final``, which is all the area
computation needs.

Idempotent. Writes a ``.bak`` next to the target on first run.

Usage:
    PYTHONPATH=. python pipeline/images/quality/backfill_mask_area.py \
        --all-data data/all_data.json
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.images.quality.mask_edge_fraction import mask_area_px, mask_area_um2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_mask(path: str):
    """Match ``mask_edge_fraction.load_mask``: foreground = pixels > 0."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-data", type=Path, default=Path("data/all_data.json"))
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    with open(args.all_data) as f:
        data = json.load(f)

    if not args.no_backup:
        bak = args.all_data.with_suffix(args.all_data.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(args.all_data, bak)
            logger.info("Backup written: %s", bak)

    done = missing_mask = no_upp = 0
    for rec in data.values():
        im = rec.get("images") or {}
        mp = im.get("mask_path")
        upp = (im.get("um_per_px") or {}).get("final") or {}
        ux, uy = upp.get("x"), upp.get("y")
        if not mp or not Path(mp).exists():
            im["mask_area_px"] = None
            im["mask_area_um2"] = None
            missing_mask += 1
            continue
        area_px = mask_area_px(_load_mask(mp))
        im["mask_area_px"] = area_px
        a2 = mask_area_um2(area_px, ux, uy)
        im["mask_area_um2"] = a2
        if a2 is None:
            no_upp += 1
        done += 1

    with open(args.all_data, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "Backfilled %d records (%d missing mask, %d missing um/px) -> %s",
        done, missing_mask, no_upp, args.all_data,
    )


if __name__ == "__main__":
    main()
