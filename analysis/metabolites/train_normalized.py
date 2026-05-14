#!/usr/bin/env python3
"""
Train metabolite models using normalized CONC + EXCH CSVs.

Matches analysis/metabolites/train.py behavior but swaps input data:
  - Uses winsorized + volume-normalized metabolite values (*_win_vol_norm)
  - Concatenates CONC (concentration) and EXCH (exchange rate) features
  - Uses same train/val/test splits CSV and day ordering

Outputs:
  - analysis/outputs/metabolites_normalized/results.json
  - analysis/outputs/figures/LightGBM_vs_Logistic_Regression_normalized.png

Usage:
    make run ARGS="-m analysis.metabolites.train_normalized"
    make run ARGS="-m analysis.metabolites.train_normalized --days Dy30"
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
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER, FIGURE_DIR

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
SPLITS_CSV = "data/splits/canonical_2026_winter.csv"
CONC_CSV = "data/normalized/CONC_data_organoides_residualized_final.csv"
EXCH_CSV = "data/normalized/EXCH_data_organoides_residualized_final.csv"
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "metabolites_normalized"

METABOLITES = [
    "GlucoseGlo",
    "GlutamateGlo",
    "LactateGlo",
    "PyruvateGlo",
    "MalateGlo",
    "BCAAGlo",
]

NORMALIZED_SUFFIX = "_win_vol_norm"

LGBM_PARAM_GRID = {
    "max_depth": [3, 6],
    "num_leaves": [31, 47, 63],
    "min_child_samples": [10, 20],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 500],
}

LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "max_iter": [1000],
}

LGBM_THRESHOLD_GRID = np.linspace(0.3, 0.7, 9)
LR_THRESHOLD_GRID = np.linspace(0.1, 0.9, 17)


def _normalize_organoid_id(value: str) -> str:
    if value is None:
        return ""
    return "_".join(str(value).split())


def _day_to_label(day_value: int) -> Optional[str]:
    if pd.isna(day_value):
        return None
    day_num = int(day_value)
    if day_num == 21:
        return "Dy20_5"
    return f"Dy{day_num:02d}"


def _load_splits() -> Dict[str, str]:
    splits_df = pd.read_csv(SPLITS_CSV)
    splits_df["organoid_id"] = splits_df["organoid_id"].map(_normalize_organoid_id)
    return dict(zip(splits_df["organoid_id"], splits_df["split"]))


def _get_feature_columns(df: pd.DataFrame, prefix: str) -> Tuple[List[str], List[str]]:
    cols = []
    names = []
    for met in METABOLITES:
        col = f"{met}{NORMALIZED_SUFFIX}"
        if col in df.columns:
            cols.append(col)
            names.append(f"{prefix}_{met}")
    return cols, names


def _load_normalized_data() -> pd.DataFrame:
    conc_df = pd.read_csv(CONC_CSV)
    exch_df = pd.read_csv(EXCH_CSV)

    conc_df["Organoid"] = conc_df["Organoid"].map(_normalize_organoid_id)
    exch_df["Organoid"] = exch_df["Organoid"].map(_normalize_organoid_id)

    conc_cols, conc_names = _get_feature_columns(conc_df, "CONC")
    exch_cols, exch_names = _get_feature_columns(exch_df, "EXCH")

    conc_keep = ["Organoid", "Day", "Classification"] + conc_cols
    exch_keep = ["Organoid", "Day", "Classification"] + exch_cols

    conc_df = conc_df[conc_keep].copy()
    exch_df = exch_df[exch_keep].copy()

    conc_df = conc_df.rename(columns=dict(zip(conc_cols, conc_names)))
    exch_df = exch_df.rename(columns=dict(zip(exch_cols, exch_names)))

    merged = conc_df.merge(
        exch_df,
        on=["Organoid", "Day", "Classification"],
        how="inner",
    )

    merged["day_label"] = merged["Day"].map(_day_to_label)
    merged = merged[merged["day_label"].isin(DAY_ORDER)].copy()

    splits = _load_splits()
    merged["split"] = merged["Organoid"].map(splits)
    merged = merged[merged["split"].notna()].copy()

    merged["label"] = merged["Classification"].map(
        {"Good": 0, "Bad": 1, "Acceptable": 0, "Not Acceptable": 1}
    )

    merged = merged[merged["label"].notna()].copy()

    return merged


def _get_features(df: pd.DataFrame, split: str, day: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    subset = df[(df["split"] == split) & (df["day_label"] == day)].copy()
    if subset.empty:
        return np.empty((0, 0)), np.array([]), [], []
    feature_cols = [
        c
        for c in subset.columns
        if c.startswith("CONC_") or c.startswith("EXCH_")
    ]
    X = subset[feature_cols].to_numpy(dtype=float)
    y = subset["label"].to_numpy(dtype=int)
    ids = subset["Organoid"].tolist()
    return X, y, feature_cols, ids


def _impute_with_train_stats(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feat_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if X_train.size == 0:
        return X_train, X_val, X_test, feat_names

    all_nan_mask = np.all(np.isnan(X_train), axis=0)
    if np.any(all_nan_mask):
        keep = ~all_nan_mask
        X_train = X_train[:, keep]
        X_val = X_val[:, keep] if X_val.size else X_val
        X_test = X_test[:, keep] if X_test.size else X_test
        feat_names = [f for f, k in zip(feat_names, keep) if k]

    with np.errstate(all="ignore"):
        medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    X_train = np.where(np.isnan(X_train), medians, X_train)
    if X_val.size:
        X_val = np.where(np.isnan(X_val), medians, X_val)
    if X_test.size:
        X_test = np.where(np.isnan(X_test), medians, X_test)

    return X_train, X_val, X_test, feat_names


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
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


def train_lgbm_day(df: pd.DataFrame, day: str, verbose: bool = True) -> Optional[dict]:
    import lightgbm as lgb

    X_train, y_train, feat_names, ids_train = _get_features(df, "train", day)
    X_val, y_val, _, ids_val = _get_features(df, "val", day)
    X_test, y_test, _, ids_test = _get_features(df, "test", day)

    if len(X_train) == 0 or len(X_test) == 0:
        return None
    if len(np.unique(y_train)) < 2:
        return None

    X_train, X_val, X_test, feat_names = _impute_with_train_stats(
        X_train, X_val, X_test, feat_names
    )

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

    groups = np.array([oid for oid in ids_train])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        base_model,
        LGBM_PARAM_GRID,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train, groups=groups)

    best_params = grid.best_params_
    if verbose:
        print(f"  Best params: {best_params}")

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

    importances = final_model.feature_importances_
    feat_imp = sorted(
        zip(feat_names, importances), key=lambda x: x[1], reverse=True
    )
    metrics["feature_importance"] = [
        {"feature": f, "importance": int(imp)} for f, imp in feat_imp
    ]

    return metrics


def train_logreg_day(df: pd.DataFrame, day: str, verbose: bool = True) -> Optional[dict]:
    X_train, y_train, feat_names, ids_train = _get_features(df, "train", day)
    X_val, y_val, _, ids_val = _get_features(df, "val", day)
    X_test, y_test, _, ids_test = _get_features(df, "test", day)

    if len(X_train) == 0 or len(X_test) == 0:
        return None
    if len(np.unique(y_train)) < 2:
        return None

    X_train, X_val, X_test, feat_names = _impute_with_train_stats(
        X_train, X_val, X_test, feat_names
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, X_train.shape[1]))
    X_test_s = scaler.transform(X_test)

    groups = np.array([oid for oid in ids_train])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    base_model = LogisticRegression(
        class_weight="balanced",
        random_state=SEED,
        solver="saga",
    )

    grid = GridSearchCV(
        base_model,
        LR_PARAM_GRID,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train_s, y_train, groups=groups)

    best_params = grid.best_params_
    if verbose:
        print(f"  Best params: {best_params}")

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

    late_start = None
    for i, d in enumerate(days):
        if d in ("Dy24", "Dy28", "Dy30") and late_start is None:
            late_start = i
    if late_start is not None:
        ax.axvspan(late_start - 0.5, len(days) - 0.5, alpha=0.1, color="gray")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def print_comparison_table(results: dict):
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

    df = _load_normalized_data()
    days_to_train = args.days if args.days else DAY_ORDER

    results = {"lgbm": {}, "logreg": {}}

    for day in days_to_train:
        if day not in DAY_ORDER:
            print(f"\nSkipping {day} (not in day order)")
            continue

        if not args.skip_lgbm:
            print(f"\n{'='*50}")
            print(f"LightGBM - {day}")
            print(f"{'='*50}")
            m = train_lgbm_day(df, day)
            if m:
                results["lgbm"][day] = m
                print(f"  Accuracy:     {m['accuracy']:.4f}")
                print(f"  Balanced Acc: {m['balanced_accuracy']:.4f}")
                print(f"  Recall (NA):  {m['recall_not_acceptable']:.4f}")

        if not args.skip_lr:
            print(f"\n{'='*50}")
            print(f"Logistic Regression - {day}")
            print(f"{'='*50}")
            m = train_logreg_day(df, day)
            if m:
                results["logreg"][day] = m
                print(f"  Accuracy:     {m['accuracy']:.4f}")
                print(f"  Balanced Acc: {m['balanced_accuracy']:.4f}")
                print(f"  Recall (NA):  {m['recall_not_acceptable']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'results.json'}")

    print(f"\n{'='*60}")
    print("AGGREGATE COMPARISON (Table 3)")
    print(f"{'='*60}")
    print_comparison_table(results)

    if results["lgbm"] and results["logreg"]:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_lgbm_vs_lr(
            results,
            FIGURE_DIR / "LightGBM_vs_Logistic_Regression_normalized.png",
        )


if __name__ == "__main__":
    main()
