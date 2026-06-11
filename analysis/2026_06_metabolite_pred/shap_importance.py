#!/usr/bin/env python3
"""Out-of-fold SHAP feature importance per model, per day -> text files.

For the headline feature configs (nominal+delta, size-scaled+delta), runs the
same nested CV as ``run.py`` but, via ``run_cv_for_day``'s fold callback,
computes SHAP for each held-out fold using that fold's fitted model
(TreeExplainer for LightGBM, LinearExplainer for Logistic Regression on the
standardized features). Out-of-fold SHAP is concatenated and features are ranked
per day by mean(|SHAP|).

Writes one text file per (cohort, config, model):
    $ANALYSIS_OUTPUT_DIR/metabolite_pred/metabolite_pred_<cohort>_<config>_<model>_shap.txt

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/shap_importance.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/shap_importance.py
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import shap

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER
from pipeline.splits import Splits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import ALL_DATA_PATH, build_cohort
from cv import run_cv_for_day
from analysis.paper_2026_04.metabolites_train import MODEL_SPECS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COHORTS = ("strong-consensus", "full")

# Headline configs only (keeps the file count at 2 configs x 2 cohorts x 2 models
# = 8). Winsorized variants can be added by appending {"winsorize": True} dicts.
HEADLINE_CONFIGS = (
    {"key": "nominal_delta", "label": "Nominal + delta",
     "normalize_by_size": False, "include_growth": True, "winsorize": False},
    {"key": "scaled_delta", "label": "Size-scaled + delta",
     "normalize_by_size": True, "include_growth": True, "winsorize": False},
)


def _features(ds, day, cfg):
    return ds.get_metabolite_features(
        "all", day,
        include_growth=cfg["include_growth"], include_initial=True,
        normalize_by_size=cfg["normalize_by_size"], winsorize=cfg.get("winsorize", False),
    )


def _oof_shap_importance(spec, X, y, ids, names, *, n_folds, seed):
    """{feature: mean|SHAP|} from out-of-fold SHAP, or None if CV can't run."""
    X = np.asarray(X, float)
    oof_abs = np.full(X.shape, np.nan)

    def cb(spec, est, X_tr, X_te, te):
        if spec.name == "lgbm":
            sv = shap.TreeExplainer(est).shap_values(X_te)
        else:
            sv = shap.LinearExplainer(est, X_tr).shap_values(X_te)
        oof_abs[te] = np.abs(np.asarray(sv, dtype=float))

    m = run_cv_for_day(spec, X, y, ids, n_folds=n_folds, seed=seed, fold_callback=cb)
    if m is None:
        return None
    mean_abs = np.nanmean(oof_abs, axis=0)
    return dict(zip(names, mean_abs.tolist()))


def _write_file(out_dir, cohort, cfg, spec, per_day):
    path = out_dir / f"metabolite_pred_{cohort}_{cfg['key']}_{spec.name}_shap.txt"
    lines = [
        "# Out-of-fold SHAP feature importance (mean |SHAP| over held-out folds)",
        f"# cohort={cohort}  config={cfg['key']} ({cfg['label']})  model={spec.display}",
        "# Features ranked per day; larger mean|SHAP| = more influence on the prediction.",
        "",
    ]
    for day in DAY_ORDER:
        if day not in per_day:
            continue
        lines.append(f"== {day} ==")
        ranked = sorted(per_day[day].items(), key=lambda kv: kv[1], reverse=True)
        for rank, (feat, val) in enumerate(ranked, 1):
            lines.append(f"  {rank:2d}. {feat:42s} {val:.5f}")
        lines.append("")
    path.write_text("\n".join(lines))
    logger.info("Saved %s", path)


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["strong", "strong-consensus", "full", "all"], default="all")
    ap.add_argument("--configs", nargs="+", default=None,
                    choices=[c["key"] for c in HEADLINE_CONFIGS])
    ap.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    aliases = {"strong": "strong-consensus", "strong-consensus": "strong-consensus", "full": "full"}
    cohorts = COHORTS if args.cohort == "all" else (aliases[args.cohort],)
    cfgs = ([c for c in HEADLINE_CONFIGS if c["key"] in args.configs]
            if args.configs else HEADLINE_CONFIGS)
    days = args.days if args.days else DAY_ORDER
    specs = [MODEL_SPECS["lgbm"], MODEL_SPECS["logreg"]]

    out_dir = ANALYSIS_OUTPUT_DIR / "metabolite_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cohort in cohorts:
        ds, _ = build_cohort(cohort, ALL_DATA_PATH)
        ds.apply_splits(
            Splits.from_dict({o: "all" for o in ds.organoid_ids},
                             name=f"shap_{cohort}", provenance="shap_importance"),
            strict=True,
        )
        for cfg in cfgs:
            per_day_by_spec = {spec.name: {} for spec in specs}
            for day in days:
                if day not in ds.days:
                    continue
                X, y, names, ids = _features(ds, day, cfg)
                for spec in specs:
                    imp = _oof_shap_importance(spec, X, y, ids, names,
                                               n_folds=args.folds, seed=args.seed)
                    if imp is not None:
                        per_day_by_spec[spec.name][day] = imp
                logger.info("  [%s/%s] %s done", cohort, cfg["key"], day)
            for spec in specs:
                _write_file(out_dir, cohort, cfg, spec, per_day_by_spec[spec.name])


if __name__ == "__main__":
    main()
