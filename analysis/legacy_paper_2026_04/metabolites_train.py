#!/usr/bin/env python3
"""
Reproduce metabolite model results: LightGBM and Logistic Regression per day.

Trains per-day classifiers using the paper's configuration:
  - Features: concentrations + initial_concentrations + growth (day-to-day diffs)
  - LightGBM with GridSearchCV (StratifiedGroupKFold, 3 folds)
  - Logistic Regression with StandardScaler
  - Threshold tuning on validation set
  - Final refit on train+val, evaluate on test

Outputs:
  - analysis/outputs/metabolites/results.json (all metrics per model per day)
  - analysis/outputs/figures/LightGBM_vs_Logistic_Regression.png

Usage:
    make run ARGS="-m analysis.metabolites.train"
    make run ARGS="-m analysis.metabolites.train --days Dy30"
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    CONDITIONAL_METABOLITES,
    DAY_ORDER,
    FIGURE_DIR,
    OrganoidDataset,
    REQUIRED_METABOLITES,
    get_day_int_floor,
)
from pipeline.splits import Splits

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "metabolites"

# LightGBM hyperparameter grid (compact, matching student code)
LGBM_PARAM_GRID = {
    "max_depth": [3, 6],
    "num_leaves": [31, 47, 63],
    "min_child_samples": [10, 20],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 500],
}

# Logistic regression grid (student code included l1 + l2)
LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "max_iter": [1000],
}

# Threshold grids — student code used different ranges per model
LGBM_THRESHOLD_GRID = np.linspace(0.3, 0.7, 9)    # LightGBM: 0.3–0.7, 9 points
LR_THRESHOLD_GRID = np.linspace(0.1, 0.9, 17)      # LR: 0.1–0.9, 17 points


def _get_growth_features(
    ds: OrganoidDataset, split: str, day: str, org_ids: List[str]
) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
    """Compute growth (delta) features for the given organoids at the given day."""
    day_idx = DAY_ORDER.index(day) if day in DAY_ORDER else -1
    if day_idx <= 0:
        return None, [], org_ids

    prev_day = DAY_ORDER[day_idx - 1]

    day_num = get_day_int_floor(day)
    prev_day_num = get_day_int_floor(prev_day)

    active_mets = list(REQUIRED_METABOLITES)
    for met, cond_fn in CONDITIONAL_METABOLITES.items():
        if day_num is not None and cond_fn(day_num):
            active_mets.append(met)

    prev_mets = list(REQUIRED_METABOLITES)
    for met, cond_fn in CONDITIONAL_METABOLITES.items():
        if prev_day_num is not None and cond_fn(prev_day_num):
            prev_mets.append(met)

    growth_mets = [m for m in active_mets if m in prev_mets]
    growth_names = [f"{m}_growth" for m in growth_mets]

    rows = []
    kept_ids = []
    for org_id in org_ids:
        info = ds._organoids[org_id]
        curr_rec = info["records"].get(day)
        prev_rec = info["records"].get(prev_day)
        if curr_rec is None or prev_rec is None:
            continue
        curr_mets = curr_rec.get("metabolite", {})
        prev_mets_data = prev_rec.get("metabolite", {})

        row = []
        skip = False
        for m in growth_mets:
            c_curr = curr_mets.get(m, {}).get("concentration_uM")
            c_prev = prev_mets_data.get(m, {}).get("concentration_uM")
            if c_curr is None or c_prev is None:
                skip = True
                break
            row.append(c_curr - c_prev)
        if skip:
            continue
        rows.append(row)
        kept_ids.append(org_id)

    if not rows:
        return None, [], org_ids

    return np.array(rows, dtype=float), growth_names, kept_ids


def get_features_with_growth(
    ds: OrganoidDataset, split: str, day: str
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Get metabolite features + growth features for split+day."""
    X_base, y_base, feat_names, org_ids = ds.get_metabolite_features(
        split, day, include_growth=False, include_initial=True
    )

    growth_arr, growth_names, kept_ids = _get_growth_features(ds, split, day, org_ids)

    if growth_arr is not None and len(kept_ids) > 0:
        # Filter base features to match kept_ids
        id_to_idx = {oid: i for i, oid in enumerate(org_ids)}
        keep_mask = [id_to_idx[oid] for oid in kept_ids]
        X_base = X_base[keep_mask]
        y_base = y_base[keep_mask]
        org_ids = kept_ids

        X = np.hstack([X_base, growth_arr])
        feat_names = feat_names + growth_names
    else:
        X = X_base

    return X, y_base, feat_names, org_ids


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute full metrics suite."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    result = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "tpr_acceptable": round(tpr, 4),
        "tnr_not_acceptable": round(tnr, 4),
        "sensitivity": round(tpr, 4),
        "specificity": round(tnr, 4),
        "precision_acceptable": round(prec[0], 4),
        "recall_acceptable": round(rec[0], 4),
        "f1_acceptable": round(f1[0], 4),
        "precision_not_acceptable": round(prec[1], 4),
        "recall_not_acceptable": round(rec[1], 4),
        "f1_not_acceptable": round(f1[1], 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": len(y_true),
        "n_positive": int(sum(y_true == 1)),
        "n_negative": int(sum(y_true == 0)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        result["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)

    return result


def train_lgbm_day(ds, day, verbose=True):
    """Train LightGBM for one day: GridSearchCV → threshold tune → refit → test."""
    import lightgbm as lgb

    X_train, y_train, feat_names, ids_train = get_features_with_growth(ds, "train", day)
    X_val, y_val, _, ids_val = get_features_with_growth(ds, "val", day)
    X_test, y_test, _, ids_test = get_features_with_growth(ds, "test", day)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    # Class weights
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    base_model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        verbosity=-1,
        n_jobs=1,
    )

    # Phase 1: GridSearchCV on train
    groups = np.array([oid for oid in ids_train])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        base_model,
        LGBM_PARAM_GRID,
        cv=cv,
        scoring="f1",  # f1 for Not Acceptable (pos_label=1)
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train, groups=groups)

    best_params = grid.best_params_
    if verbose:
        print(f"  Best params: {best_params}")

    # Phase 2: Threshold tuning on validation (f1 for Not Acceptable)
    if len(X_val) > 0 and len(np.unique(y_val)) > 1:
        val_probs = grid.predict_proba(X_val)[:, 1]
        best_threshold = 0.5
        best_f1 = 0.0
        for t in LGBM_THRESHOLD_GRID:
            preds = (val_probs >= t).astype(int)
            f = f1_score(y_val, preds, pos_label=1, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_threshold = t
    else:
        best_threshold = 0.5

    if verbose:
        print(f"  Best threshold: {best_threshold:.2f}")

    # Phase 3: Refit on train+val, evaluate on test
    X_trainval = np.vstack([X_train, X_val]) if len(X_val) > 0 else X_train
    y_trainval = np.concatenate([y_train, y_val]) if len(y_val) > 0 else y_train

    n_pos_tv = sum(y_trainval == 1)
    n_neg_tv = sum(y_trainval == 0)
    spw_tv = n_neg_tv / n_pos_tv if n_pos_tv > 0 else 1.0

    final_model = lgb.LGBMClassifier(
        **best_params,
        objective="binary",
        scale_pos_weight=spw_tv,
        random_state=SEED,
        verbosity=-1,
        n_jobs=1,
    )
    final_model.fit(X_trainval, y_trainval)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    metrics = compute_metrics(y_test, test_preds, test_probs)
    metrics["threshold"] = best_threshold
    metrics["best_params"] = best_params
    metrics["feature_names"] = feat_names

    # Feature importance
    importances = final_model.feature_importances_
    feat_imp = sorted(
        zip(feat_names, importances), key=lambda x: x[1], reverse=True
    )
    metrics["feature_importance"] = [
        {"feature": f, "importance": int(imp)} for f, imp in feat_imp
    ]

    return metrics


def train_logreg_day(ds, day, verbose=True):
    """Train Logistic Regression for one day.

    Key differences from LightGBM (matching student code):
    - CV scoring: f1_weighted (not f1 for minority class)
    - Threshold grid: 0.1–0.9 (17 points)
    - Threshold metric: f1_weighted
    - Solver: saga (supports l1 + l2)
    """
    X_train, y_train, feat_names, ids_train = get_features_with_growth(ds, "train", day)
    X_val, y_val, _, ids_val = get_features_with_growth(ds, "val", day)
    X_test, y_test, _, ids_test = get_features_with_growth(ds, "test", day)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, X_train.shape[1]))
    X_test_s = scaler.transform(X_test)

    groups = np.array([oid for oid in ids_train])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    # Student code used saga solver (supports l1+l2) and f1_weighted scoring
    base_model = LogisticRegression(
        class_weight="balanced",
        random_state=SEED,
        solver="saga",
    )

    grid = GridSearchCV(
        base_model,
        LR_PARAM_GRID,
        cv=cv,
        scoring="f1_weighted",  # student code default for LR
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train_s, y_train, groups=groups)

    best_params = grid.best_params_
    if verbose:
        print(f"  Best params: {best_params}")

    # Threshold tuning on validation — f1_weighted, wider grid (0.1–0.9)
    if len(X_val_s) > 0 and len(np.unique(y_val)) > 1:
        val_probs = grid.predict_proba(X_val_s)[:, 1]
        best_threshold = 0.5
        best_score = 0.0
        for t in LR_THRESHOLD_GRID:
            preds = (val_probs >= t).astype(int)
            score = f1_score(y_val, preds, average="weighted", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = t
    else:
        best_threshold = 0.5

    if verbose:
        print(f"  Best threshold: {best_threshold:.2f}")

    # Refit on train+val
    X_trainval = np.vstack([X_train_s, X_val_s]) if len(X_val_s) > 0 else X_train_s
    y_trainval = np.concatenate([y_train, y_val]) if len(y_val) > 0 else y_train

    final_model = LogisticRegression(
        **best_params,
        class_weight="balanced",
        random_state=SEED,
        solver="saga",
    )
    final_model.fit(X_trainval, y_trainval)

    test_probs = final_model.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    metrics = compute_metrics(y_test, test_preds, test_probs)
    metrics["threshold"] = best_threshold
    metrics["best_params"] = best_params
    metrics["feature_names"] = feat_names

    return metrics


def plot_lgbm_vs_lr(results: dict, output_path: Path):
    """Reproduce Figure 8: LightGBM vs Logistic Regression balanced accuracy."""
    import matplotlib.pyplot as plt

    days = []
    lgbm_ba = []
    lr_ba = []

    for day in DAY_ORDER:
        lgbm = results.get("lgbm", {}).get(day)
        lr = results.get("logreg", {}).get(day)
        if lgbm is not None and lr is not None:
            days.append(day)
            lgbm_ba.append(lgbm["balanced_accuracy"])
            lr_ba.append(lr["balanced_accuracy"])

    x = range(len(days))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, lgbm_ba, "o-", label="LightGBM", color="#1f77b4", linewidth=2)
    ax.plot(x, lr_ba, "s--", label="Logistic Regression", color="#ff7f0e", linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Day")
    ax.set_title("LightGBM vs Logistic Regression: Balanced Accuracy by Day")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    # Shade late-stage region (Day 24+)
    late_start = None
    for i, d in enumerate(days):
        n = get_day_int_floor(d)
        if n is not None and n >= 24 and late_start is None:
            late_start = i
    if late_start is not None:
        ax.axvspan(late_start - 0.5, len(days) - 0.5, alpha=0.1, color="gray")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def print_comparison_table(results: dict):
    """Print Table 3: LR vs LightGBM aggregate comparison."""
    for model_name in ["logreg", "lgbm"]:
        model_results = results.get(model_name, {})
        if not model_results:
            continue

        accs = []
        bal_accs = []
        recall_nas = []
        days_zero_recall = 0
        best_bal_acc = 0.0

        for day in DAY_ORDER:
            m = model_results.get(day)
            if m is None:
                continue
            accs.append(m["accuracy"])
            bal_accs.append(m["balanced_accuracy"])
            r_na = m["recall_not_acceptable"]
            recall_nas.append(r_na)
            if r_na == 0.0:
                days_zero_recall += 1
            best_bal_acc = max(best_bal_acc, m["balanced_accuracy"])

        n_days = len(accs)
        display = "LightGBM" if model_name == "lgbm" else "Logistic Regression"
        print(f"\n{display}:")
        print(f"  Avg Accuracy:       {np.mean(accs):.1%}")
        print(f"  Avg Balanced Acc:   {np.mean(bal_accs):.1%}")
        print(f"  Avg Recall (N.A.):  {np.mean(recall_nas):.1%}")
        print(f"  Days Recall_NA = 0: {days_zero_recall}/{n_days}")
        print(f"  Best Balanced Acc:  {best_bal_acc:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None,
                        help="Specific days to train (e.g. Dy30 Dy24)")
    parser.add_argument("--skip-lr", action="store_true",
                        help="Skip logistic regression")
    parser.add_argument("--skip-lgbm", action="store_true",
                        help="Skip LightGBM")
    args = parser.parse_args()

    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical())
    print(ds.summary())

    days_to_train = args.days if args.days else DAY_ORDER

    results = {"lgbm": {}, "logreg": {}}

    for day in days_to_train:
        if day not in ds.days:
            print(f"\nSkipping {day} (no data)")
            continue

        if not args.skip_lgbm:
            print(f"\n{'='*50}")
            print(f"LightGBM - {day}")
            print(f"{'='*50}")
            m = train_lgbm_day(ds, day)
            if m:
                results["lgbm"][day] = m
                print(f"  Accuracy:     {m['accuracy']:.4f}")
                print(f"  Balanced Acc: {m['balanced_accuracy']:.4f}")
                print(f"  Recall (NA):  {m['recall_not_acceptable']:.4f}")

        if not args.skip_lr:
            print(f"\n{'='*50}")
            print(f"Logistic Regression - {day}")
            print(f"{'='*50}")
            m = train_logreg_day(ds, day)
            if m:
                results["logreg"][day] = m
                print(f"  Accuracy:     {m['accuracy']:.4f}")
                print(f"  Balanced Acc: {m['balanced_accuracy']:.4f}")
                print(f"  Recall (NA):  {m['recall_not_acceptable']:.4f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'results.json'}")

    # Print comparison table (Table 3)
    print(f"\n{'='*60}")
    print("AGGREGATE COMPARISON (Table 3)")
    print(f"{'='*60}")
    print_comparison_table(results)

    # Plot Figure 8
    if results["lgbm"] and results["logreg"]:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_lgbm_vs_lr(results, FIGURE_DIR / "LightGBM_vs_Logistic_Regression.png")


if __name__ == "__main__":
    main()
