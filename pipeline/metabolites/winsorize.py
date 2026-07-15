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

# Per-record regeneration proof: the fraction of records whose stored ``win`` we
# reproduce as ``k * per-day-winsorized(concentration_uM)`` within MATCH_TOL.
# MATCH_TOL is relative — small enough to be meaningful, loose enough to absorb
# assay noise, float error, and percentile-boundary interpolation differences.
MATCH_TOL = 0.05
# Minimum acceptable match rate per metabolite. Measured 2026-07 on the shipped
# all_data.json at MATCH_TOL (Glucose .969, Lactate .977, BCAA .965,
# Pyruvate .971, Glutamate .821); floors carry ~5-12 pt margin for data drift.
# Glutamate's raw reads are noisier, so its floor is lower — still far above the
# ~0.07 a non-winsorized signal (MalateGlo) scores.
MATCH_FLOOR = {
    "GlucoseGlo": 0.90,
    "LactateGlo": 0.90,
    "BCAAGlo": 0.90,
    "PyruvateGlo": 0.90,
    "GlutamateGlo": 0.70,
}
# MalateGlo is the documented exception: its stored ``win`` is a separately-
# cleaned noise-floor signal, NOT a winsorization of concentration_uM. It must
# stay far below the well-behaved floors (measured .068 at MATCH_TOL).
MALATE_MATCH_CEIL = 0.25


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


def verify_winsorization(all_data, tol=0.03, low_q=LOW_Q, high_q=HIGH_Q,
                         match_tol=MATCH_TOL):
    """Assert we can regenerate the stored ``win`` columns from source.

    Stored ``win`` equals (per-metabolite units constant ``k``) x winsorized-raw.
    For each well-behaved metabolite we recompute per-day-winsorized raw
    ``concentration_uM``, fit ``k`` on the non-clipped bulk, and check two ways:

    * the **median** relative residual is < ``tol`` (the bulk lines up), and
    * a per-record **match rate** — the fraction of records reproduced within
      ``match_tol`` (relative) — clears that metabolite's ``MATCH_FLOOR``.

    Together these *prove* the winsorized data we use is regenerable from the raw
    reads (not silently drifted). ``MalateGlo`` is asserted to be the documented
    exception: its stored ``win`` is a separately-cleaned noise-floor signal, so
    it must FAIL both checks (median residual > ``tol`` and match rate below
    ``MALATE_MATCH_CEIL``).

    Raises AssertionError on failure. Returns a per-metabolite report dict
    (``k``, ``median_rel_resid``, ``match_rate``, ``match_tol``, ``n``).
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
        match_rate = float(np.mean(resid < match_tol))
        report[m] = {
            "k": k, "median_rel_resid": med,
            "match_rate": match_rate, "match_tol": match_tol, "n": int(raw.size),
        }
        logger.info("  %-12s k=%-9.4f median_rel_resid=%.4f  match_rate=%.1f%% (@%.0f%%)",
                    m, k, med, 100 * match_rate, 100 * match_tol)

    for m in WELL_BEHAVED:
        assert m in report, f"{m} missing from verification"
        assert report[m]["median_rel_resid"] < tol, (
            f"{m}: our winsorization diverges from stored win "
            f"(median rel resid {report[m]['median_rel_resid']:.4f} >= {tol})"
        )
        floor = MATCH_FLOOR[m]
        assert report[m]["match_rate"] >= floor, (
            f"{m}: only {report[m]['match_rate']:.1%} of records regenerate within "
            f"{match_tol:.0%} of the stored win (floor {floor:.0%}) — cannot prove "
            "this metabolite's win is regenerable from concentration_uM"
        )
    # Malate is expected to NOT match (different cleaned signal) — on both checks.
    if MALATE in report:
        assert report[MALATE]["median_rel_resid"] > tol, (
            f"{MALATE}: unexpectedly matches stored win; its provenance may have "
            "changed (it was a separately-cleaned noise-floor signal)."
        )
        assert report[MALATE]["match_rate"] < MALATE_MATCH_CEIL, (
            f"{MALATE}: {report[MALATE]['match_rate']:.1%} of records regenerate "
            f"within {match_tol:.0%} (ceil {MALATE_MATCH_CEIL:.0%}) — it was expected "
            "NOT to be a winsorization of concentration_uM; provenance may have changed."
        )
        logger.info("  %s is the documented exception (stored win != winsorized raw).", MALATE)
    logger.info(
        "verify_winsorization: OK — %d well-behaved metabolites regenerable from "
        "source (median rel resid < %.3f AND per-record match >= floor @%.0f%%)",
        len(WELL_BEHAVED), tol, 100 * match_tol,
    )
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
