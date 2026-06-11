#!/usr/bin/env python3
"""Organoid size (segmentation area) accessors + a provenance check.

The RehenLab analysis normalizes metabolites by cross-sectional AREA (not
volume): specifically the winsorized mean of the current and previous day's area
(``Average_area_win``). We compute area ourselves from the predicted masks
(``mask_area_um2``, see ``pipeline/images/quality/mask_edge_fraction.py``) and
verify it reproduces the lab's ``Area_win`` column.

Usage:
    PYTHONPATH=. python pipeline/images/quality/organoid_area.py \
        --all-data data/all_data.json \
        --csv data/normalized/CONC_data_organoides_residualized_final.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from pipeline.data_loader import get_mask_area_um2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def organoid_area_um2(record):
    """Segmentation cross-sectional area (um^2) for a record, or None."""
    return get_mask_area_um2(record)


def average_area_um2(curr_record, prev_record):
    """Paper's size denominator: mean of current and previous day's area.

    First day (no previous record) falls back to the current area. Returns None
    if the current area is missing.
    """
    a_curr = get_mask_area_um2(curr_record)
    if a_curr is None:
        return None
    a_prev = get_mask_area_um2(prev_record) if prev_record is not None else None
    if a_prev is None:
        return a_curr
    return (a_curr + a_prev) / 2.0


def _day_key(day_number):
    """all_data day.number (e.g. 3.0, 20.5) -> CSV Day string ('3', '20.5')."""
    n = float(day_number)
    return str(int(n)) if n == int(n) else str(n)


def verify_mask_area(all_data, csv_path, tol=0.03):
    """Assert our ``mask_area_um2`` reproduces the lab's CSV ``Area_win``.

    Joins by (organoid_id, day). ``Area_win`` is the lab's winsorized area, so
    a small fraction of clipped extremes will differ; we assert on the MEDIAN
    relative difference and the log-log correlation, which are robust to that.

    Raises AssertionError on failure. Returns a stats dict.
    """
    rows = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows[(r["Organoid"], str(r["Day"]))] = r

    ours, theirs = [], []
    for rec in all_data.values():
        a = get_mask_area_um2(rec)
        if not a:
            continue
        key = (rec.get("organoid_id"), _day_key((rec.get("day") or {}).get("number")))
        row = rows.get(key)
        if row is None:
            continue
        aw = row.get("Area_win")
        if aw in (None, "", "NA"):
            continue
        ours.append(a)
        theirs.append(float(aw))

    ours = np.array(ours, float)
    theirs = np.array(theirs, float)
    n = len(ours)
    assert n >= 100, f"only {n} joined rows; cannot verify mask area"

    pdiff = np.abs(ours - theirs) / np.abs(theirs)
    median_pdiff = float(np.median(pdiff))
    corr = float(np.corrcoef(np.log(ours), np.log(theirs))[0, 1])
    median_ratio = float(np.median(ours / theirs))
    stats = {
        "n": n, "median_pct_diff": median_pdiff,
        "p90_pct_diff": float(np.percentile(pdiff, 90)),
        "log_corr": corr, "median_ratio": median_ratio,
    }
    logger.info(
        "  joined=%d  median%%diff=%.3f%%  p90=%.2f%%  log_corr=%.4f  median_ratio=%.4f",
        n, 100 * median_pdiff, 100 * stats["p90_pct_diff"], corr, median_ratio,
    )
    assert median_pdiff < tol, (
        f"mask_area_um2 vs Area_win median diff {100*median_pdiff:.2f}% >= {100*tol:.0f}%"
    )
    assert corr > 0.99, f"mask_area_um2 vs Area_win log-corr {corr:.4f} <= 0.99"
    logger.info("verify_mask_area: OK (median diff %.2f%% < %.0f%%)", 100 * median_pdiff, 100 * tol)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-data", type=Path, default=Path("data/all_data.json"))
    ap.add_argument("--csv", type=Path,
                    default=Path("data/normalized/CONC_data_organoides_residualized_final.csv"))
    ap.add_argument("--tol", type=float, default=0.03)
    args = ap.parse_args()
    with open(args.all_data) as f:
        all_data = json.load(f)
    verify_mask_area(all_data, args.csv, tol=args.tol)


if __name__ == "__main__":
    main()
