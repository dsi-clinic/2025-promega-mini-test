#!/usr/bin/env python3
"""
Metabolite Organoid Trajectory Classification

Goal
-----
Compare late-day trajectory features vs full-history features for predicting
the final Day30 label at two target days: Dy28 and Dy30.

Each sample is (Organoid, Target_Day). Features are the metabolite trajectory
up to that target day, flattened into one row.

Experiments (4 variants)
------------------------
1. traj_late_Dy28      : Late-only trajectory → predict at Dy28
    - Uses days: Dy24, Dy28

2. traj_allhist_Dy28   : All-history trajectory → predict at Dy28
    - Uses all available days ≤ 28

3. traj_late_Dy30      : Late-only trajectory → predict at Dy30
    - Uses days: Dy24, Dy28, Dy30

4. traj_allhist_Dy30   : All-history trajectory → predict at Dy30
    - Uses all available days ≤ 30

Label is the final organoid label ("Acceptable" / "Not Acceptable").

Usage
-----
    # Run all four experiments
    python train_metabolites_trajectory.py

    # Run a single variant
    python train_metabolites_trajectory.py --variant traj_late_Dy28

    # Run multiple variants
    python train_metabolites_trajectory.py --variant traj_late_Dy28 --variant traj_allhist_Dy28
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

SEED = 42


# ---------------------------------------------------------------------
# Utilities: seeding, loading, growth features
# ---------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def json_to_df(json_data: Dict) -> pd.DataFrame:
    """
    Convert JSON split data to DataFrame with metabolite features.

    Expected JSON structure (same as existing per-day script):
        {
            "Organoid_ID_1": {
                "label": "Acceptable" / "Not Acceptable",
                "batch": "BA1 96_1",
                "timepoints": {
                    "Dy03": {
                        "img_path": "...",
                        "mask_path": "...",
                        "metabolites": {
                            "GlucoseGlo_concentration_uM": float,
                            ...
                        }
                    },
                    "Dy06": { ... },
                    ...
                }
            },
            ...
        }
    """
    rows = []
    for org_id, info in json_data.items():
        label = info.get("label")
        batch = info.get("batch")
        timepoints = info.get("timepoints", {})

        for day_name, tp in timepoints.items():
            row = {
                "ID": org_id,
                "batch": batch,
                "label": label,
                "DY": day_name,
                "img_path": tp.get("img_path"),
                "mask_path": tp.get("mask_path"),
            }

            for k, v in tp.get("metabolites", {}).items():
                row[k] = v

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["day"] = df["DY"].str.extract(r"(\d+)").astype(int)
    return df


def compute_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add growth features (difference between consecutive timepoints) per organoid.
    Assumes columns like GlucoseGlo_concentration_uM exist.
    """
    df = df.copy()
    if "day" not in df.columns:
        df["day"] = df["DY"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["ID", "day"])

    # Identify metabolite concentration columns by suffix
    conc_cols = [
        c for c in df.columns if c.endswith("_concentration_uM") and df[c].dtype != "O"
    ]

    for col in conc_cols:
        growth_col = col.replace("_concentration_uM", "_growth")
        df[growth_col] = df.groupby("ID")[col].diff()

    return df


# ---------------------------------------------------------------------
# Cleaning / scaling
# ---------------------------------------------------------------------
def clean_and_scale_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Clean NaNs/constants and scale data based on X_train statistics.
    Applies same transformations to X_val and X_test if provided.
    """
    # Drop all-NaN columns
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"  Dropping all-NaN columns: {all_nan_cols}")
        X_train = X_train.drop(columns=all_nan_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in all_nan_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(
                columns=[c for c in all_nan_cols if c in X_test.columns]
            )

    # Drop constant columns
    constant_cols = [
        col for col in X_train.columns if X_train[col].nunique(dropna=True) <= 1
    ]
    if constant_cols:
        print(f"  Dropping constant columns: {constant_cols}")
        X_train = X_train.drop(columns=constant_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(
                columns=[c for c in constant_cols if c in X_test.columns]
            )

    # Drop near-constant columns
    near_constant_cols = []
    for col in X_train.columns:
        col_std = X_train[col].std(skipna=True)
        if np.isfinite(col_std) and col_std < 1e-6:
            near_constant_cols.append(col)
    if near_constant_cols:
        print(f"  Dropping near-constant columns: {near_constant_cols}")
        X_train = X_train.drop(columns=near_constant_cols)
        if X_val is not None:
            X_val = X_val.drop(
                columns=[c for c in near_constant_cols if c in X_val.columns]
            )
        if X_test is not None:
            X_test = X_test.drop(
                columns=[c for c in near_constant_cols if c in X_test.columns]
            )

    # Fill NaNs
    if X_train.isna().any().any():
        print("  Filling remaining NaNs with 0")
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val_scaled = None
    if X_val is not None and X_val.shape[1] > 0:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )

    X_test_scaled = None
    if X_test is not None and X_test.shape[1] > 0:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------
# Calibration + threshold tuning
# ---------------------------------------------------------------------
def save_calibration_diagnostic(
    model_dir: Path, val_scores: np.ndarray, y_val_bin: np.ndarray
) -> None:
    """
    Save a simple calibration diagnostic:
    - calibration_bins.csv
    - calibration_curve.png
    Based on VALIDATION predictions.
    """
    val_scores = np.asarray(val_scores)
    y_val_bin = np.asarray(y_val_bin)

    if len(val_scores) == 0:
        print("No validation data for calibration diagnostic.")
        return

    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(val_scores, bins) - 1  # 0..9
    records = []
    mean_preds = []
    frac_pos = []

    for b in range(10):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        scores_bin = val_scores[mask]
        y_bin = y_val_bin[mask]
        mean_pred = float(scores_bin.mean())
        frac_positive = float(y_bin.mean())
        count = int(mask.sum())
        left = float(bins[b])
        right = float(bins[b + 1])

        records.append(
            {
                "bin_left": left,
                "bin_right": right,
                "mean_pred": mean_pred,
                "frac_positive": frac_positive,
                "count": count,
            }
        )
        mean_preds.append(mean_pred)
        frac_pos.append(frac_positive)

    calib_df = pd.DataFrame(records)
    calib_path_csv = model_dir / "calibration_bins.csv"
    calib_df.to_csv(calib_path_csv, index=False)
    print(f"Saved calibration bins to {calib_path_csv}")

    # Plot reliability curve
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if len(mean_preds) > 0:
        plt.plot(mean_preds, frac_pos, "o-", label="Model")
    plt.xlabel("Mean predicted probability (Acceptable)")
    plt.ylabel("Empirical fraction Acceptable")
    plt.title("Calibration Curve (Validation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    calib_path_png = model_dir / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(calib_path_png, dpi=150)
    plt.close()
    print(f"Saved calibration curve to {calib_path_png}")


def tune_threshold(scores: np.ndarray, y_true_bin: np.ndarray) -> Tuple[float, float]:
    """
    Grid-search threshold in [0.1, 0.9] to maximize F1 for Acceptable (1).
    """
    scores = np.asarray(scores)
    y_true_bin = np.asarray(y_true_bin)

    thresholds = np.linspace(0.1, 0.9, 17)
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred_bin = (scores >= t).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


# ---------------------------------------------------------------------
# Trajectory dataset construction
# ---------------------------------------------------------------------
def get_allowed_days_for_variant(
    target_day: int, mode: str, all_days_sorted: List[int]
) -> List[int]:
    """
    Compute which day numbers to include for a given (target_day, mode).

    mode = "late"    : small set of late days
    mode = "allhist" : all days <= target_day
    """
    if mode == "late":
        if target_day == 28:
            candidate_days = [24, 28]
        elif target_day == 30:
            candidate_days = [24, 28, 30]
        else:
            # Fallback: just last two days <= target_day
            candidate_days = [d for d in all_days_sorted if d <= target_day][-2:]
        allowed = [
            d for d in candidate_days if d in all_days_sorted and d <= target_day
        ]
    elif mode == "allhist":
        allowed = [d for d in all_days_sorted if d <= target_day]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(allowed)


def build_trajectory_dataset(
    df: pd.DataFrame,
    target_day: int,
    mode: str,
    all_days_sorted: List[int],
    metabolite_cols: List[str],
    growth_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build a trajectory dataset for a given split and variant.

    - Filters to rows with day <= target_day.
    - For each organoid, flattens allowed days into a single row.
    - One row per (organoid, target_day).

    Returns:
        X      : features DataFrame
        y      : labels (string)
        groups : groups (ID)
        ids    : organoid IDs (same as groups)
    """
    df = df.copy()
    if df.empty:
        return (
            pd.DataFrame(),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
        )

    df = df[df["day"] <= target_day].copy()
    if df.empty:
        return (
            pd.DataFrame(),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
        )

    allowed_days = get_allowed_days_for_variant(target_day, mode, all_days_sorted)
    print(f"  Allowed days for target={target_day}, mode={mode}: {allowed_days}")

    ids = sorted(df["ID"].unique())
    rows = []

    for org_id in ids:
        df_id = df[df["ID"] == org_id]
        if df_id.empty:
            continue

        # Skip organoids that have no measurements at any allowed day
        has_any_day = False
        for d in allowed_days:
            if (df_id["day"] == d).any():
                has_any_day = True
                break
        if not has_any_day:
            continue

        row = {
            "ID": org_id,
            "target_day": target_day,
        }
        # Label is final label, same across days
        row["label"] = df_id["label"].iloc[0]

        for d in allowed_days:
            df_id_d = df_id[df_id["day"] == d]
            if df_id_d.empty:
                # Create NaNs for all features at this day
                for col in metabolite_cols:
                    row[f"{col}_Dy{d}"] = np.nan
                for col in growth_cols:
                    row[f"{col}_Dy{d}"] = np.nan
            else:
                r = df_id_d.iloc[0]
                for col in metabolite_cols:
                    row[f"{col}_Dy{d}"] = r.get(col, np.nan)
                for col in growth_cols:
                    row[f"{col}_Dy{d}"] = r.get(col, np.nan)

        rows.append(row)

    if not rows:
        return (
            pd.DataFrame(),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
        )

    traj_df = pd.DataFrame(rows)

    y = traj_df["label"]
    groups = traj_df["ID"]
    ids_out = traj_df["ID"]
    X = traj_df.drop(columns=["label", "ID"])

    return X, y, groups, ids_out


# ---------------------------------------------------------------------
# Per-organoid predictions and metrics
# ---------------------------------------------------------------------
def save_organoid_predictions(
    output_path: Path,
    ids: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """
    Save per-organoid predictions to CSV.
    """
    label_map = {"Acceptable": 1, "Not Acceptable": 0}

    ids = np.asarray(ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    organoid_results = []
    for idx in range(len(ids)):
        org_id = ids[idx]
        true_label_str = y_true[idx]
        pred_label_str = y_pred[idx]

        true_label = label_map.get(true_label_str, 0)
        pred_label = label_map.get(pred_label_str, 0)
        pred_prob = float(y_score[idx])
        correct = bool(pred_label == true_label)

        if true_label == 1 and pred_label == 1:
            cm_category = "TP"
        elif true_label == 0 and pred_label == 1:
            cm_category = "FP"
        elif true_label == 1 and pred_label == 0:
            cm_category = "FN"
        else:
            cm_category = "TN"

        organoid_results.append(
            {
                "Organoid_ID": org_id,
                "True_Label": true_label,
                "Predicted_Probability": pred_prob,
                "Predicted_Label": pred_label,
                "Correct": correct,
                "CM_Category": cm_category,
            }
        )

    organoid_preds_df = pd.DataFrame(organoid_results)
    organoid_preds_df.to_csv(output_path, index=False)
    print(f"  Saved organoid predictions to {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray, classes: List[str], title: str, out_path: Path
):
    """
    Simple 2x2 confusion matrix plot.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix to {out_path}")


# ---------------------------------------------------------------------
# Core training for one variant
# ---------------------------------------------------------------------
def run_variant(
    variant_name: str,
    target_day: int,
    mode: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_days_sorted: List[int],
    metabolite_cols: List[str],
    growth_cols: List[str],
    output_root: Path,
) -> None:
    """
    Train and evaluate one trajectory variant.
    """

    print("\n" + "=" * 60)
    print(f"Training Trajectory Model: {variant_name}")
    print(f"Target day: {target_day} | Mode: {mode}")
    print("=" * 60)

    set_seed(SEED)

    # Build trajectory datasets
    X_train, y_train, groups_train, ids_train = build_trajectory_dataset(
        train_df, target_day, mode, all_days_sorted, metabolite_cols, growth_cols
    )
    X_val, y_val, groups_val, ids_val = build_trajectory_dataset(
        val_df, target_day, mode, all_days_sorted, metabolite_cols, growth_cols
    )
    X_test, y_test, groups_test, ids_test = build_trajectory_dataset(
        test_df, target_day, mode, all_days_sorted, metabolite_cols, growth_cols
    )

    print(
        f"  Train rows: {len(X_train)}, Val rows: {len(X_val)}, Test rows: {len(X_test)}"
    )

    if X_train.empty or X_val.empty or X_test.empty:
        print("  Not enough data for this variant (empty split). Skipping.")
        return

    # Clean + scale
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = clean_and_scale_data(
        X_train, X_val, X_test
    )

    if X_train_scaled.shape[1] == 0:
        print("  No features left after cleaning. Skipping.")
        return

    # Class weights and scale_pos_weight on train
    classes = np.unique(y_train)
    if len(classes) < 2:
        print("  Only one class in training data. Skipping.")
        return

    class_weights_balanced = compute_class_weight(
        "balanced", classes=classes, y=y_train
    )
    class_weight_dict = {
        cls: float(w) for cls, w in zip(classes, class_weights_balanced)
    }

    # scale_pos_weight for "Acceptable"
    pos_label_str = "Acceptable"
    y_train_arr = pd.Series(y_train).to_numpy()
    pos = (y_train_arr == pos_label_str).sum()
    neg = (y_train_arr != pos_label_str).sum()
    ratio_balanced = (neg / pos) if pos > 0 else 1.0

    base_model = LGBMClassifier(
        random_state=SEED,
        verbose=-1,
        n_jobs=1,
        scale_pos_weight=ratio_balanced,
        class_weight=class_weight_dict,
        boosting_type="gbdt",
    )

    param_grid = {
        "max_depth": [3, 6],
        "num_leaves": [31, 63],
        "min_child_samples": [10, 20],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 500],
    }

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=4,
        verbose=0,
    )

    print("  Running GridSearchCV for hyperparameters...")
    grid.fit(X_train_scaled, y_train, groups=groups_train)
    best_params = grid.best_params_
    print(f"  Best Params: {best_params}")

    best_model = grid.best_estimator_

    # Threshold tuning on VAL
    print("  Tuning threshold on validation set...")
    val_proba = best_model.predict_proba(X_val_scaled)
    classes_order = list(best_model.classes_)
    if pos_label_str in classes_order:
        pos_idx = classes_order.index(pos_label_str)
        val_scores = val_proba[:, pos_idx]
    else:
        # fallback to second column
        val_scores = val_proba[:, 1]

    y_val_bin = (pd.Series(y_val) == pos_label_str).astype(int).to_numpy()
    if len(np.unique(y_val_bin)) < 2:
        print("  Validation set has only one class. Using default threshold=0.5")
        threshold = 0.5
        best_val_f1 = None
    else:
        threshold, best_val_f1 = tune_threshold(val_scores, y_val_bin)
        print(f"  Tuned threshold: {threshold:.3f} (VAL F1={best_val_f1:.3f})")

    # Calibration
    variant_dir = output_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_calibration_diagnostic(variant_dir, val_scores, y_val_bin)

    # Final training on TRAIN+VAL with best_params
    print("  Re-fitting model on TRAIN+VAL with best hyperparameters...")
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    pd.concat([groups_train, groups_val], axis=0)

    # Rebuild weights on combined
    classes_comb = np.unique(y_combined)
    weights_comb = compute_class_weight("balanced", classes=classes_comb, y=y_combined)
    weight_dict_comb = {cls: float(w) for cls, w in zip(classes_comb, weights_comb)}

    y_comb_arr = pd.Series(y_combined).to_numpy()
    pos_comb = (y_comb_arr == pos_label_str).sum()
    neg_comb = (y_comb_arr != pos_label_str).sum()
    ratio_comb = (neg_comb / pos_comb) if pos_comb > 0 else 1.0

    # Re-clean+scale for combined vs test, to avoid data leakage from val
    X_comb_scaled, _, X_test_scaled_final, _ = clean_and_scale_data(
        X_combined, X_test=X_test
    )

    final_model = LGBMClassifier(
        random_state=SEED,
        verbose=-1,
        n_jobs=1,
        scale_pos_weight=ratio_comb,
        class_weight=weight_dict_comb,
        boosting_type="gbdt",
        **best_params,
    )
    final_model.fit(X_comb_scaled, y_combined)

    # Test predictions
    proba_test = final_model.predict_proba(X_test_scaled_final)
    classes_final = list(final_model.classes_)
    if pos_label_str in classes_final:
        pos_idx_final = classes_final.index(pos_label_str)
        y_score_test = proba_test[:, pos_idx_final]
    else:
        y_score_test = proba_test[:, 1]

    y_pred_test = np.where(y_score_test >= threshold, pos_label_str, "Not Acceptable")

    # Metrics
    if len(np.unique(y_test)) > 1:
        y_true_bin = (pd.Series(y_test) == pos_label_str).astype(int).to_numpy()
        roc_auc = float(roc_auc_score(y_true_bin, y_score_test))
    else:
        roc_auc = None

    accuracy = float(accuracy_score(y_test, y_pred_test))
    report = classification_report(
        y_test, y_pred_test, output_dict=True, zero_division=0
    )

    precision_accept = float(report.get("Acceptable", {}).get("precision", 0.0))
    recall_accept = float(report.get("Acceptable", {}).get("recall", 0.0))
    f1_accept = float(report.get("Acceptable", {}).get("f1-score", 0.0))

    precision_notaccept = float(report.get("Not Acceptable", {}).get("precision", 0.0))
    recall_notaccept = float(report.get("Not Acceptable", {}).get("recall", 0.0))
    f1_notaccept = float(report.get("Not Acceptable", {}).get("f1-score", 0.0))

    # Confusion matrix with fixed label order
    label_order = ["Not Acceptable", "Acceptable"]
    cm = confusion_matrix(y_test, y_pred_test, labels=label_order)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(
        f"  Test ROC AUC: {roc_auc:.3f}"
        if roc_auc is not None
        else "  Test ROC AUC: N/A"
    )
    print(f"  Test Accuracy: {accuracy:.3f}")
    print(f"  Test F1 (Acceptable): {f1_accept:.3f}")
    print(f"  Test Recall (Acceptable): {recall_accept:.3f}")
    print(f"  Test Precision (Acceptable): {precision_accept:.3f}")
    print(f"  Threshold used: {threshold:.3f}")
    print(f"  Confusion matrix (labels={label_order}):\n{cm}")

    # Save per-organoid predictions
    preds_path = variant_dir / "organoid_predictions.csv"
    save_organoid_predictions(preds_path, ids_test, y_test, y_pred_test, y_score_test)

    # Save confusion matrix plot
    cm_path = variant_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        cm, label_order, f"Confusion Matrix - {variant_name}", cm_path
    )

    # Save metrics JSON
    metrics = {
        "variant": variant_name,
        "target_day": int(target_day),
        "test_accuracy": accuracy,
        "test_f1_acceptable": f1_accept,
        "test_f1_notacceptable": f1_notaccept,
        "test_recall_acceptable": recall_accept,
        "test_recall_notacceptable": recall_notaccept,
        "test_precision_acceptable": precision_accept,
        "test_precision_notacceptable": precision_notaccept,
        "test_specificity": specificity,
        "test_roc_auc": roc_auc,
        "threshold_used": float(threshold),
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        },
        "best_params": best_params,
        "val_f1_at_threshold": best_val_f1,
    }
    metrics_path = variant_dir / "metrics_test.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Save a one-row CSV summary for convenience
    summary_df = pd.DataFrame(
        [
            {
                "Variant": variant_name,
                "Target_Day": target_day,
                "Test_Accuracy": accuracy,
                "Test_F1_Acceptable": f1_accept,
                "Test_Recall_Acceptable": recall_accept,
                "Test_Precision_Acceptable": precision_accept,
                "Test_Specificity": specificity,
                "Test_ROC_AUC": roc_auc,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Threshold_Used": float(threshold),
            }
        ]
    )
    summary_path = variant_dir / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved results summary to {summary_path}")

    print("=" * 60)
    print(f"Finished variant: {variant_name}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Metabolite Trajectory Classifiers (Late vs All-History)."
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Variant(s) to run. Choices: "
            "traj_late_Dy28, traj_allhist_Dy28, traj_late_Dy30, traj_allhist_Dy30. "
            "If not specified, runs all four."
        ),
    )
    args = parser.parse_args()

    VARIANTS = {
        "traj_late_Dy28": {"target_day": 28, "mode": "late"},
        "traj_allhist_Dy28": {"target_day": 28, "mode": "allhist"},
        "traj_late_Dy30": {"target_day": 30, "mode": "late"},
        "traj_allhist_Dy30": {"target_day": 30, "mode": "allhist"},
    }

    variants_to_run = args.variant if args.variant else list(VARIANTS.keys())

    # Paths (same JSON split structure as per-day script)
    train_data_path = Path("data_splits/both_train_base.json")
    val_data_path = Path("data_splits/both_val_base.json")
    test_data_path = Path("data_splits/both_test_base.json")
    output_root = Path("analysis/metabolites/classifier/outputs_metabolites_trajectory")

    print("\n" + "=" * 60)
    print("Loading data splits...")
    print("=" * 60)

    with open(train_data_path, "r") as f:
        train_data_json = json.load(f)
    with open(val_data_path, "r") as f:
        val_data_json = json.load(f)
    with open(test_data_path, "r") as f:
        test_data_json = json.load(f)

    train_df = json_to_df(train_data_json)
    val_df = json_to_df(val_data_json)
    test_df = json_to_df(test_data_json)

    print(f"Train rows (per-day): {len(train_df)}")
    print(f"Val rows   (per-day): {len(val_df)}")
    print(f"Test rows  (per-day): {len(test_df)}")

    print("\nComputing growth features...")
    train_df = compute_growth_features(train_df)
    val_df = compute_growth_features(val_df)
    test_df = compute_growth_features(test_df)

    # Global lists of days and feature names for consistent trajectories
    df_all = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    all_days_sorted = sorted(df_all["day"].unique())

    metabolite_cols = [
        c
        for c in df_all.columns
        if c.endswith("_concentration_uM") and df_all[c].dtype != "O"
    ]
    growth_cols = [c for c in df_all.columns if c.endswith("_growth")]

    print("\n" + "!" * 60)
    print("Trajectory models use VALIDATION ONLY for threshold tuning.")
    print("No test data is used for threshold or hyperparameter selection.")
    print("!" * 60 + "\n")

    for variant_name in variants_to_run:
        if variant_name not in VARIANTS:
            print(f"Unknown variant '{variant_name}'. Skipping.")
            continue

        cfg = VARIANTS[variant_name]
        run_variant(
            variant_name=variant_name,
            target_day=cfg["target_day"],
            mode=cfg["mode"],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            all_days_sorted=all_days_sorted,
            metabolite_cols=metabolite_cols,
            growth_cols=growth_cols,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()