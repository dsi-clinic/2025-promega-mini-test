#!/usr/bin/env python3
"""How many organoids are dropped when delta (growth) features are computed?

Scoped to the IDOR sample (BA1+BA2 col2). For each cohort and each day, build
the metabolite feature matrix with ``include_growth=True`` and report how many
organoids fall out because they lack a usable previous-day value (the row drop
that ``_add_growth_features`` performs, now surfaced via a warning).

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/check_growth_drops.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/check_growth_drops.py
"""

import os
import sys
import warnings

# Sibling modules are imported top-level (see run.py for why).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohorts import ALL_DATA_PATH, build_cohort

from pipeline.data_loader import DAY_ORDER
from pipeline.splits import Splits

COHORTS = ("strong-consensus", "full")


def _n_with_growth(ds, day):
    """(n_without_growth, n_with_growth) for one day on split 'all'."""
    _, _, _, ids_base = ds.get_metabolite_features(
        "all", day, include_growth=False, include_initial=True
    )
    # Silence the drop warning here; we report the delta ourselves.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, _, ids_growth = ds.get_metabolite_features(
            "all", day, include_growth=True, include_initial=True
        )
    return len(ids_base), len(ids_growth)


def main():
    for cohort in COHORTS:
        ds, counts = build_cohort(cohort, ALL_DATA_PATH)
        # Single 'all' split so get_metabolite_features works (mirrors run.py).
        ds.apply_splits(
            Splits.from_dict(
                {oid: "all" for oid in ds.organoid_ids},
                name=f"drops_{cohort}",
                provenance="check_growth_drops",
            ),
            strict=True,
        )
        print(f"\n=== Cohort {cohort}: {len(ds.organoid_ids)} organoids  {counts} ===")
        print(f"{'day':>8}  {'no-growth':>9}  {'with-growth':>11}  {'dropped':>7}")
        for day in DAY_ORDER:
            n_base, n_growth = _n_with_growth(ds, day)
            note = "  (first day: no delta)" if DAY_ORDER.index(day) == 0 else ""
            print(f"{day:>8}  {n_base:>9}  {n_growth:>11}  {n_base - n_growth:>7}{note}")


if __name__ == "__main__":
    main()
