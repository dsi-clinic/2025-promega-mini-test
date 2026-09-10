#!/usr/bin/env python3
"""Met-only 10×4-fold repeated CV: LGBM + LogReg, mirroring combined_kfold fold structure.

Key design: SEED=1, all_org_ids computed from full 139 labeled organoids upfront,
folds split on the full list then subsetted per day — exactly matching combined_kfold.py.
This makes these results directly comparable to met_nan/met_nan_lgbm in
combined_results_kfold_series_idor_139.json.

Differences from met_classifier_comparison.py:
  - SEED 1 (not 42)
  - fold splits on all 139 upfront (not per-day subset)
  - fold_seed = rep_seed + fold_i * 97  (not rep_seed + fold_i)
  - training uses inner_tr_oids (85% of train split) to match combined_kfold
  - no SVM/MLP; no met_raw/met_drop variants

Output:
  analysis_output/images/met_lgbm_logreg_kfold_139.json

Usage:
    python3 -m analysis.paper_2026_04.met_lgbm_logreg_kfold
    sbatch analysis/paper_2026_04/submit_met_lgbm_logreg_kfold.slurm
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    LABEL_TO_INT,
    OrganoidDataset,
    idor_ba1_ba2_filters,
    require_complete_series,
)

from .metabolites_train import _features_for_day_all as _met_features_all
from .combined_kfold import _filter_fold

warnings.filterwarnings("ignore", category=UserWarning)

SEED      = 1
N_FOLDS   = 4
N_REPEATS = 10
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "images" / "met_lgbm_logreg_kfold_139.json"

LGBM_PARAM_GRID = {
    "max_depth":         [3, 6],
    "num_leaves":        [15, 31],
    "min_child_samples": [5, 10],
    "learning_rate":     [0.05, 0.1],
    "n_estimators":      [100, 300],
}

LOGREG_GRID = {
    "C":        [0.01, 0.1, 1.0, 10.0],
    "penalty":  ["l1", "l2"],
    "max_iter": [1000],
}


def _train_lgbm_fold(X_tr, y_tr, X_te, fold_seed: int) -> Optional[np.ndarray]:
    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None
    spw = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
    model = lgb.LGBMClassifier(
        objective="binary", scale_pos_weight=spw,
        random_state=fold_seed, verbosity=-1, n_jobs=1,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    grid = GridSearchCV(model, LGBM_PARAM_GRID, cv=inner_cv,
                        scoring="f1", n_jobs=-1, refit=True)
    grid.fit(X_tr, y_tr)
    return grid.predict_proba(X_te)[:, 1]


def _train_logreg_fold(X_tr, y_tr, X_te, fold_seed: int) -> Optional[np.ndarray]:
    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None
    imputer = SimpleImputer(strategy="median")
    X_tr_i = imputer.fit_transform(X_tr)
    X_te_i = imputer.transform(X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_i)
    X_te_s = scaler.transform(X_te_i)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    model = LogisticRegression(class_weight="balanced", solver="saga",
                               random_state=fold_seed)
    grid = GridSearchCV(model, LOGREG_GRID, cv=inner_cv,
                        scoring="f1_weighted", n_jobs=-1, refit=True)
    grid.fit(X_tr_s, y_tr)
    return grid.predict_proba(X_te_s)[:, 1]


def run_day(
    day: str,
    ds: OrganoidDataset,
    all_org_ids: List[str],
    all_labels: np.ndarray,
    n_folds: int,
    n_repeats: int,
    verbose: bool,
) -> Optional[dict]:
    X_met, y_met, _, met_ids = _met_features_all(ds, day, malate_mode="nan")
    if len(X_met) == 0:
        if verbose:
            print(f"  [{day}] no met data, skipping")
        return None

    mod_keys = ["met_nan_lgbm", "met_nan_logreg"]
    repeat_bas: Dict[str, List[float]] = {k: [] for k in mod_keys}
    repeat_cms: Dict[str, List]        = {k: [] for k in mod_keys}

    for rep in range(n_repeats):
        rep_seed = SEED + rep * 1000
        outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
        oof: Dict[str, np.ndarray] = {k: np.full(len(all_org_ids), np.nan) for k in mod_keys}

        for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(all_org_ids, all_labels)):
            fold_seed = rep_seed + fold_i * 97
            tr_oids = [all_org_ids[i] for i in tr_idx]
            te_oids = [all_org_ids[i] for i in te_idx]

            # Mirror combined_kfold: carve out 15% val from train (for img early-stopping parity)
            sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=fold_seed)
            inner_tr_idx, _ = next(sss.split(tr_oids, all_labels[tr_idx]))
            inner_tr_oids = [tr_oids[i] for i in inner_tr_idx]

            X_tr_m, y_tr_m, _ = _filter_fold(X_met, y_met, met_ids, inner_tr_oids)
            X_te_m, y_te_m, valid_te_m = _filter_fold(X_met, y_met, met_ids, te_oids)

            if len(X_tr_m) == 0 or len(X_te_m) == 0:
                continue

            p_lgbm = _train_lgbm_fold(X_tr_m, y_tr_m, X_te_m, fold_seed)
            if p_lgbm is not None:
                for oid, prob in zip(valid_te_m, p_lgbm):
                    oof["met_nan_lgbm"][all_org_ids.index(oid)] = prob

            p_logreg = _train_logreg_fold(X_tr_m, y_tr_m, X_te_m, fold_seed)
            if p_logreg is not None:
                for oid, prob in zip(valid_te_m, p_logreg):
                    oof["met_nan_logreg"][all_org_ids.index(oid)] = prob

            if verbose:
                print(f"  [{day}] rep={rep+1}/{n_repeats}  fold={fold_i+1}/{n_folds}  "
                      f"tr={len(X_tr_m)}  te={len(X_te_m)}")

        for k in mod_keys:
            valid = ~np.isnan(oof[k])
            if valid.sum() < 2:
                continue
            yt = all_labels[valid]
            yp = (oof[k][valid] >= 0.5).astype(int)
            if len(np.unique(yt)) < 2:
                continue
            repeat_bas[k].append(float(balanced_accuracy_score(yt, yp)))
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            repeat_cms[k].append(cm.tolist())

    results = {}
    for k in mod_keys:
        bas = repeat_bas[k]
        if bas:
            results[k] = {
                "balanced_accuracy_mean":     float(np.mean(bas)),
                "balanced_accuracy_std":      float(np.std(bas)),
                "n_repeats":                  len(bas),
                "n_folds":                    n_folds,
                "repeat_balanced_accuracies": bas,
                "repeat_confusion_matrices":  repeat_cms[k],
            }
    return results or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",      nargs="+", default=None)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds",   type=int, default=N_FOLDS)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    ds = OrganoidDataset(
        ALL_DATA_PATH, splits=None,
        filters=[*idor_ba1_ba2_filters(), require_complete_series(drop_stitched=False)],
    )
    all_org_ids = [o for o in ds.organoid_ids if ds.organoid_label(o) in LABEL_TO_INT]
    all_labels  = np.array([LABEL_TO_INT[ds.organoid_label(o)] for o in all_org_ids])
    print(f"Organoids: {len(all_org_ids)}  "
          f"({all_labels.sum()} NAcc, {(all_labels==0).sum()} Acc)")
    print(f"Protocol: {args.n_repeats}×{args.n_folds}-fold  SEED={SEED}")

    days = args.days or list(DAY_ORDER)

    all_results = {}
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        with open(OUTPUT_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming from {OUTPUT_PATH}  ({len(all_results)} days already done)")

    for day in days:
        if day in all_results:
            print(f"[{day}] already done, skipping")
            continue
        if day not in ds.days:
            print(f"[{day}] no data in dataset, skipping")
            continue
        print(f"\n[{day}] running {args.n_repeats}×{args.n_folds}-fold ...")
        day_res = run_day(day, ds, all_org_ids, all_labels,
                          args.n_folds, args.n_repeats, args.verbose)
        if day_res:
            all_results[day] = day_res
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved → {OUTPUT_PATH}")

    print("\n\n=== Summary (mean BA) ===")
    header = f"{'Day':<10}{'lgbm':>12}{'logreg':>12}"
    print(header)
    for day in DAY_ORDER:
        if day not in all_results:
            continue
        r = all_results[day]
        row = f"{day:<10}"
        for k in ["met_nan_lgbm", "met_nan_logreg"]:
            v = r.get(k)
            cell = f"{v['balanced_accuracy_mean']:.3f}" if v else "—"
            row += f"{cell:>12}"
        print(row)


if __name__ == "__main__":
    main()
