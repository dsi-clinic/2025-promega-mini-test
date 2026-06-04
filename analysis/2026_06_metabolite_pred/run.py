#!/usr/bin/env python3
"""Metabolite-only prediction on our sample (IDOR col2), two label cohorts.

Ports the per-day LightGBM + Logistic Regression modeling from
``analysis/paper_2026_04/metabolites_train.py`` but scopes it to the IDOR col2
set and evaluates by stratified cross-validation (see ``cv.run_cv_for_day``).
Run once per cohort:

  - ``strong-consensus`` (198): supermajority labels (>= 4 of 5 regular votes).
  - ``full`` (248): all col2, ambiguous 3-2 / 2-3 resolved by simple majority.

Outputs:
  - $ANALYSIS_OUTPUT_DIR/metabolite_pred/results_<cohort>.json
  - $ANALYSIS_OUTPUT_DIR/figures/metabolite_pred_<cohort>_LightGBM_vs_LogReg.png

Usage (package name starts with a digit, so run by path, not ``-m``):
    make run ARGS="analysis/2026_06_metabolite_pred/run.py"
    make run ARGS="analysis/2026_06_metabolite_pred/run.py --cohort strong --days Dy30 --skip-lgbm"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/run.py
"""

import argparse
import json
import logging

import numpy as np

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
)
from pipeline.splits import Splits

# Importing the source module reuses its MODEL_SPECS/factories AND triggers its
# module-level warnings.filterwarnings("ignore", UserWarning) — benign, intended
# to quiet sklearn/lightgbm noise. We deliberately do NOT import its _train_one /
# main (those encode the single held-out-split pipeline we are replacing with CV).
from analysis.paper_2026_04.metabolites_train import MODEL_SPECS
from analysis.paper_2026_04.common import plot_balanced_accuracy_by_day

# Sibling modules are imported as top-level (not relative): the package name
# starts with a digit so it can't be imported as ``analysis.2026_06_...``; the
# script is run by path, which puts this directory on sys.path.
from cohorts import ALL_DATA_PATH, build_cohort
from cv import run_cv_for_day

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COHORTS = ("strong-consensus", "full")
_COHORT_ALIASES = {"strong": "strong-consensus", "strong-consensus": "strong-consensus",
                   "full": "full"}

# matches metabolites_train.py so the two PNGs read consistently
_STYLE = {
    "LightGBM":            {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "Logistic Regression": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
}


def _features_all(ds, day):
    """Full (X, y, names, ids) for one day; every organoid is in split 'all'."""
    return ds.get_metabolite_features("all", day, include_growth=True, include_initial=True)


def _run_cohort(cohort, *, specs, days, n_folds, seed):
    ds, counts = build_cohort(cohort, ALL_DATA_PATH)
    logger.info("\n%s\nCohort %s: %d organoids  %s\n%s",
                "=" * 60, cohort, len(ds.organoid_ids), counts, "=" * 60)

    # Single split so get_metabolite_features works; CV folds are formed internally.
    ds.apply_splits(
        Splits.from_dict({oid: "all" for oid in ds.organoid_ids},
                         name=f"cv_all_{cohort}",
                         provenance=f"single-split CV harness, cohort={cohort}"),
        strict=True,
    )

    results = {spec.display: {} for spec in specs}
    for day in days:
        if day not in ds.days:
            logger.info("\nSkipping %s (no data)", day)
            continue
        X, y, names, ids = _features_all(ds, day)
        for spec in specs:
            logger.info("\n%s - %s  (n=%d)", spec.display, day, len(X))
            m = run_cv_for_day(spec, X, y, ids, n_folds=n_folds, seed=seed)
            if m is None:
                continue
            m["feature_names"] = names
            results[spec.display][day] = m
            logger.info(
                "  Balanced Acc (pooled): %.4f | CV mean+/-std: %.4f +/- %.4f | Recall(NA): %.4f",
                m["balanced_accuracy"], m["balanced_accuracy_cv_mean"],
                m["balanced_accuracy_cv_std"], m["recall_not_acceptable"],
            )
    return results


def _print_aggregate(cohort, results, days):
    print(f"\n{'=' * 60}\nAGGREGATE — {cohort}\n{'=' * 60}")
    for display, per_day in results.items():
        if not per_day:
            continue
        bal = [per_day[d]["balanced_accuracy"] for d in days if d in per_day]
        rec = [per_day[d]["recall_not_acceptable"] for d in days if d in per_day]
        print(f"\n{display}:")
        print(f"  Days evaluated:        {len(bal)}")
        print(f"  Avg Balanced Acc:      {np.mean(bal):.1%}")
        print(f"  Best Balanced Acc:     {np.max(bal):.1%}")
        print(f"  Avg Recall (N.A.):     {np.mean(rec):.1%}")


def _write_outputs(cohort, results):
    out_dir = ANALYSIS_OUTPUT_DIR / "metabolite_pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_{cohort}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved results to %s", results_path)

    if results.get("LightGBM") and results.get("Logistic Regression"):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        png = FIGURE_DIR / f"metabolite_pred_{cohort}_LightGBM_vs_LogReg.png"
        plot_balanced_accuracy_by_day(
            {"LightGBM": results["LightGBM"],
             "Logistic Regression": results["Logistic Regression"]},
            day_order=DAY_ORDER,
            output_path=png,
            title=f"Metabolite prediction ({cohort}): Balanced Accuracy by Day",
            style_overrides=_STYLE,
            late_stage_shade_from_day=24,
        )
        logger.info("Saved figure to %s", png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["strong", "strong-consensus", "full", "all"],
                        default="all")
    parser.add_argument("--days", nargs="+", default=None,
                        help="Specific days (e.g. Dy30 Dy24); default all")
    parser.add_argument("--skip-lgbm", action="store_true")
    parser.add_argument("--skip-lr", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    specs = []
    if not args.skip_lgbm:
        specs.append(MODEL_SPECS["lgbm"])
    if not args.skip_lr:
        specs.append(MODEL_SPECS["logreg"])
    if not specs:
        parser.error("nothing to run: both models skipped")

    cohorts = COHORTS if args.cohort == "all" else (_COHORT_ALIASES[args.cohort],)
    days = args.days if args.days else DAY_ORDER

    for cohort in cohorts:
        results = _run_cohort(cohort, specs=specs, days=days,
                              n_folds=args.folds, seed=args.seed)
        _print_aggregate(cohort, results, days)
        _write_outputs(cohort, results)


if __name__ == "__main__":
    main()
