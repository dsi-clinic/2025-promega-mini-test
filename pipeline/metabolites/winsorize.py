#!/usr/bin/env python3
"""Per-day winsorization of metabolite values, and a provenance check.

Reproduces the RehenLab winsorization (``main_data_analysis.ipynb``): for each
day, clip a feature to its 1st/99th percentile across organoids. This is the
transform behind the Promega ``win`` columns; we recompute it from raw
``concentration_uM`` rather than trusting the ingested ``win`` (which, for
MalateGlo, is a separately-cleaned signal -- see ``verify_winsorization``).

Usage:
    PYTHONPATH=. python pipeline/metabolites/winsorize.py --all-data data/all_data.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LOW_Q = 0.01
HIGH_Q = 0.99

# Stored Promega ``win`` is a faithful (rescaled) winsorization of raw for these.
# MalateGlo's raw sits at the assay noise floor (26% negative), so its stored
# ``win`` is a separately-cleaned signal, NOT a winsorized copy of raw -- it is
# an expected exception to the provenance check.
WELL_BEHAVED = ("GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo", "BCAAGlo")
MALATE = "MalateGlo"


def winsorize_per_day(values, day_labels, low_q=LOW_Q, high_q=HIGH_Q):
    """Clip ``values`` to per-day [low_q, high_q] quantiles.

    Args:
        values: 1-D array-like of floats (NaNs ignored when finding bounds).
        day_labels: same-length sequence; rows sharing a label are one day.
    Returns:
        np.ndarray of winsorized values (same order/length as ``values``).
    """
    values = np.asarray(values, dtype=float)
    day_labels = np.asarray(day_labels)
    out = values.copy()
    for d in np.unique(day_labels):
        mask = day_labels == d
        v = values[mask]
        finite = v[np.isfinite(v)]
        if finite.size < 5:
            continue
        lo = np.quantile(finite, low_q)
        hi = np.quantile(finite, high_q)
        out[mask] = np.clip(v, lo, hi)
    return out


def _collect_raw_and_win(all_data, metabolite):
    """Per-record (raw concentration_uM, stored win, day) for one metabolite."""
    raw, win, days = [], [], []
    for rec in all_data.values():
        md = (rec.get("metabolite") or {}).get(metabolite, {})
        c = md.get("concentration_uM")
        w = md.get("win")
        if c is None or w is None:
            continue
        raw.append(c)
        win.append(w)
        days.append((rec.get("day") or {}).get("id"))
    return np.array(raw, float), np.array(win, float), np.array(days)


def verify_winsorization(all_data, tol=0.03, low_q=LOW_Q, high_q=HIGH_Q):
    """Assert our per-day winsorization reproduces the stored ``win`` columns.

    Stored ``win`` equals (per-metabolite units constant ``k``) x winsorized-raw.
    For each well-behaved metabolite we recompute winsorized-raw, fit ``k`` on the
    non-clipped bulk, and require the median relative residual < ``tol``. MalateGlo
    is asserted to be the documented exception (its stored win does NOT match).

    Raises AssertionError on failure. Returns a per-metabolite report dict.
    """
    report = {}
    for m in WELL_BEHAVED + (MALATE,):
        raw, win, days = _collect_raw_and_win(all_data, m)
        if raw.size < 50:
            logger.warning("  %s: only %d records, skipping", m, raw.size)
            continue
        our = winsorize_per_day(raw, days, low_q, high_q)
        # Per-metabolite units constant from the bulk (avoid div-by-zero rows).
        nz = np.abs(our) > 1e-9
        k = float(np.median(win[nz] / our[nz]))
        resid = np.abs(win[nz] - k * our[nz]) / np.abs(k * our[nz])
        med = float(np.median(resid))
        report[m] = {"k": k, "median_rel_resid": med, "n": int(raw.size)}
        logger.info("  %-12s k=%-9.4f median_rel_resid=%.4f", m, k, med)

    for m in WELL_BEHAVED:
        assert m in report, f"{m} missing from verification"
        assert report[m]["median_rel_resid"] < tol, (
            f"{m}: our winsorization diverges from stored win "
            f"(median rel resid {report[m]['median_rel_resid']:.4f} >= {tol})"
        )
    # Malate is expected to NOT match (different cleaned signal).
    if MALATE in report:
        assert report[MALATE]["median_rel_resid"] > tol, (
            f"{MALATE}: unexpectedly matches stored win; its provenance may have "
            "changed (it was a separately-cleaned noise-floor signal)."
        )
        logger.info("  %s is the documented exception (stored win != winsorized raw).", MALATE)
    logger.info("verify_winsorization: OK (%d well-behaved metabolites within tol=%.3f)",
                len(WELL_BEHAVED), tol)
    return report


WIN_FIELDS = ("concentration_uM", "initial_concentration")


def add_winsorized_fields(all_data, low_q=LOW_Q, high_q=HIGH_Q):
    """Write per-day winsorized values into each record's metabolite block.

    For every metabolite and every field in ``WIN_FIELDS``, clip the value to
    that day's [low_q, high_q] percentile across ALL records (the same global
    per-day winsorization ``verify_winsorization`` checks against the lab's
    ``win``), and store it as ``<field>_win`` (e.g. ``concentration_uM_win``).
    Stored values are winsorized raw concentration in the SAME units as the
    source field (the lab's ``win`` is this x a per-metabolite units constant;
    see ``verify_winsorization``). Records lacking the raw value get ``None``.

    Mutates ``all_data`` in place. Returns the count of values written.
    """
    written = 0
    for m in WELL_BEHAVED + (MALATE,):
        for field in WIN_FIELDS:
            blocks, vals, days = [], [], []
            for rec in all_data.values():
                md = (rec.get("metabolite") or {}).get(m)
                if md is None:
                    continue
                v = md.get(field)
                blocks.append(md)
                vals.append(v if v is not None else np.nan)
                days.append((rec.get("day") or {}).get("id"))
            if not blocks:
                continue
            clipped = winsorize_per_day(vals, days, low_q, high_q)
            for md, wv, raw in zip(blocks, clipped, vals):
                if np.isfinite(raw):
                    md[f"{field}_win"] = float(wv)
                    written += 1
                else:
                    md[f"{field}_win"] = None
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-data", type=Path, default=Path("data/all_data.json"))
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--write", action="store_true",
                    help="Add per-day winsorized <field>_win columns to all_data.json "
                         "(default: verify-only, no write)")
    args = ap.parse_args()
    with open(args.all_data) as f:
        all_data = json.load(f)

    if args.write:
        n = add_winsorized_fields(all_data)
        with open(args.all_data, "w") as f:
            json.dump(all_data, f, indent=2)
        logger.info("Wrote %d winsorized values (%s) to %s",
                    n, "+".join(f"{x}_win" for x in WIN_FIELDS), args.all_data)
        # Re-verify the transform after writing.
        verify_winsorization(all_data, tol=args.tol)
    else:
        verify_winsorization(all_data, tol=args.tol)


if __name__ == "__main__":
    main()
