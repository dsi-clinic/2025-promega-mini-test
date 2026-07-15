#!/usr/bin/env python3
"""Determine, per metabolite, which winsorization *scope* reproduces the lab ``win``.

We were told MalateGLO's stored ``win`` was winsorized over the **whole dataset**
(all days pooled), whereas the other five metabolites were winsorized **per-day**.
This script tests that empirically. For each metabolite it winsorizes the raw
``concentration_uM`` two ways -- (a) per-day 1st/99th percentile clipping and (b)
a single whole-dataset 1st/99th clip -- and reports the fraction of records for
which the lab ``win`` matches ``k * winsorized_raw`` (``k`` = per-metabolite units
constant, fit on the bulk exactly as ``pipeline.metabolites.verify_winsorization``
does). The scope with the higher match rate is the better-fitting scheme; if
neither clears ``FIT_THRESHOLD`` the metabolite fits *neither* scope.

Reads ``win`` and ``concentration_uM`` straight from ``all_data.json`` (AGENTS.md
rules 3 & 16 -- persisted values are read, never re-derived from elsewhere) and
reuses the winsorization helpers in ``pipeline/metabolites/winsorize.py``.

Run by path (the package name starts with a digit, so ``-m`` won't work):
    make run ARGS="analysis/2026_06_metabolite_pred/verify_winsorize_scope.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/verify_winsorize_scope.py
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from pipeline.metabolites.winsorize import (
    HIGH_Q,
    LOW_Q,
    MALATE,
    WELL_BEHAVED,
    _collect_raw_and_win,
    winsorize_per_day,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Fixed, deterministic evaluation order (rule 17): the five "well-behaved"
# metabolites followed by MalateGlo.
ALL_METABOLITES: tuple[str, ...] = WELL_BEHAVED + (MALATE,)

# A record "matches" if |win - k*winsorized_raw| / |win| < DEFAULT_TOL.
DEFAULT_TOL = 0.03
# A scope is said to *reproduce* win only if its match rate clears this bar;
# below it, the metabolite fits "neither" scope.
FIT_THRESHOLD = 0.5


def winsorize_whole(
    values: np.ndarray | list[float],
    low_q: float = LOW_Q,
    high_q: float = HIGH_Q,
) -> np.ndarray:
    """Clip ``values`` to a single [low_q, high_q] over the POOLED dataset.

    The whole-dataset counterpart to ``winsorize_per_day``: one pair of quantile
    bounds computed across all days at once (NaNs ignored when finding bounds).
    Returns a winsorized copy in the same order/length as ``values``.
    """
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size < 5:
        return v.copy()
    lo = np.quantile(finite, low_q)
    hi = np.quantile(finite, high_q)
    return np.clip(v, lo, hi)


def _match_rate(win: np.ndarray, our: np.ndarray, tol: float) -> tuple[float, float]:
    """Fraction of records where lab ``win`` ~= ``k * our`` within ``tol``.

    ``win`` equals ``k * winsorized_raw`` for a per-metabolite units constant
    ``k`` (see ``verify_winsorization``). We fit ``k`` as the median ratio on the
    non-zero bulk, then count records whose relative residual
    ``|win - k*our| / |win|`` is below ``tol``. Returns ``(match_rate, k)``.
    """
    nz = np.abs(our) > 1e-9
    k = float(np.median(win[nz] / our[nz]))
    pred = k * our
    denom = np.maximum(np.abs(win), 1e-9)
    rel = np.abs(win - pred) / denom
    return float(np.mean(rel < tol)), k


def scope_match_rates(
    all_data: dict,
    metabolite: str,
    tol: float = DEFAULT_TOL,
    low_q: float = LOW_Q,
    high_q: float = HIGH_Q,
) -> dict[str, float | int | str]:
    """Per-day vs whole-dataset match rates of lab ``win`` for one metabolite.

    Returns a report dict with both match rates, the fit ``k`` for each scope,
    the better-fitting scope name, and a verdict ('per-day' / 'whole-dataset' /
    'neither' when the best scope still fails to clear ``FIT_THRESHOLD``).
    """
    raw, win, days = _collect_raw_and_win(all_data, metabolite)
    n0 = int(raw.size)
    per_day = winsorize_per_day(raw, days, low_q, high_q)
    whole = winsorize_whole(raw, low_q, high_q)
    # Rule 11: winsorization clips values, it must never add or drop a record.
    assert per_day.size == n0, f"{metabolite}: per-day changed count {n0} -> {per_day.size}"
    assert whole.size == n0, f"{metabolite}: whole changed count {n0} -> {whole.size}"

    pd_rate, pd_k = _match_rate(win, per_day, tol)
    wd_rate, wd_k = _match_rate(win, whole, tol)
    better = "per-day" if pd_rate >= wd_rate else "whole-dataset"
    best = max(pd_rate, wd_rate)
    verdict = better if best >= FIT_THRESHOLD else "neither"
    return {
        "metabolite": metabolite,
        "n": n0,
        "per_day_rate": pd_rate,
        "whole_rate": wd_rate,
        "per_day_k": pd_k,
        "whole_k": wd_k,
        "better": better,
        "verdict": verdict,
    }


def evaluate_all(all_data: dict, tol: float = DEFAULT_TOL) -> list[dict[str, float | int | str]]:
    """Report per-day vs whole-dataset match rates for every metabolite."""
    return [scope_match_rates(all_data, m, tol) for m in ALL_METABOLITES]


def print_table(rows: list[dict[str, float | int | str]], tol: float) -> None:
    """Print the per-metabolite scope table (match rate = |win - k*raw|/|win| < tol)."""
    logger.info("Winsorization scope check (match if |win - k*raw_win|/|win| < %.3f)", tol)
    logger.info("Scope 'reproduces' win only if its match rate >= %.2f.\n", FIT_THRESHOLD)
    logger.info(
        "  %-13s %6s  %9s  %9s  %-13s  %s",
        "metabolite", "n", "per-day", "whole", "better", "verdict",
    )
    logger.info("  %s", "-" * 66)
    for r in rows:
        logger.info(
            "  %-13s %6d  %9.3f  %9.3f  %-13s  %s",
            r["metabolite"], r["n"], r["per_day_rate"], r["whole_rate"],
            r["better"], r["verdict"],
        )

    per_day_fits = sorted(r["metabolite"] for r in rows if r["verdict"] == "per-day")
    whole_fits = sorted(r["metabolite"] for r in rows if r["verdict"] == "whole-dataset")
    neither = sorted(r["metabolite"] for r in rows if r["verdict"] == "neither")
    logger.info("")
    logger.info("per-day scope:       %s", ", ".join(per_day_fits) or "(none)")
    logger.info("whole-dataset scope: %s", ", ".join(whole_fits) or "(none)")
    logger.info("neither scope fits:  %s", ", ".join(neither) or "(none)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-data", type=Path, default=Path("data/all_data.json"))
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    args = ap.parse_args()

    with open(args.all_data) as f:
        all_data = json.load(f)
    rows = evaluate_all(all_data, args.tol)
    print_table(rows, args.tol)


if __name__ == "__main__":
    main()
