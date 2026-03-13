#!/usr/bin/env python3
"""
Metabolite Organoid Quality Classification (CPU Version)
Trains per-day classifiers using LightGBM with GridSearchCV.

THRESHOLD STRATEGY: Validation-based (legacy approach)
  - Phase 1: GridSearchCV on train data
  - Phase 2: Tune threshold on validation predictions (0.3-0.7 grid)
  - Phase 3: Refit on train+val, evaluate on test

CLASS IMBALANCE: scale_pos_weight emphasizes "Not Acceptable" (minority/positive class)
METRICS: 
  - Internal optimization uses pos_label="Not Acceptable" (1)
  - Reporting/Output uses pos_label="Acceptable" (1) to match legacy format

Usage:
    python3 train_metabolites_cpu.py --imbalance scale_pos_weight --scoring recall_notacceptable
    python3 train_metabolites_cpu.py --imbalance both --scoring f1_notacceptable
"""

import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    make_scorer,
)
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

SEED = 42
N_FOLDS = 3

# Internal Training Labels (Optimization Target)
TRAIN_LABEL_POS = "Not Acceptable"
TRAIN_LABEL_NEG = "Acceptable"

# Output/Reporting Labels (Legacy Format)
REPORT_LABEL_POS = "Acceptable"
REPORT_LABEL_NEG = "Not Acceptable"


def set_seed(seed=SEED):
    np.random.seed(seed)


def json_to_df(json_data):
    """Convert JSON split data to DataFrame with metabolite features."""
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

    return pd.DataFrame(rows)


def compute_growth_features(df):
    """Add growth features (difference between consecutive timepoints)."""
    df = df.copy()
    df["day"] = df["DY"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["ID", "day"])

    df["glucose_growth"] = df.groupby("ID")["GlucoseGlo_concentration_uM"].diff()
    df["glutamate_growth"] = df.groupby("ID")["GlutamateGlo_concentration_uM"].diff()
    df["LactateGlo_growth"] = df.groupby("ID")["LactateGlo_concentration_uM"].diff()
    df["PyruvateGlo_growth"] = df.groupby("ID")["PyruvateGlo_concentration_uM"].diff()
    df["MalateGlo_growth"] = df.groupby("ID")["MalateGlo_concentration_uM"].diff()

    return df


def compute_second_order_growth_features(df):
    """Add second-order growth features (acceleration: diff of diff)."""
    df = df.copy()
    if "day" not in df.columns:
        df["day"] = df["DY"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["ID", "day"])

    first_order_cols = [
        "glucose_growth", "glutamate_growth", "LactateGlo_growth",
        "PyruvateGlo_growth", "MalateGlo_growth",
    ]
    for col in first_order_cols:
        if col in df.columns:
            df[f"{col}_accel"] = df.groupby("ID")[col].diff()

    return df


def save_organoid_predictions(selected_test_df, y_test, y_pred, y_score_na, output_path):
    """
    Save per-organoid predictions to CSV.
    Legacy Format: Acceptable=1 (Pos), Not Acceptable=0 (Neg)
    
    y_score_na: P(Not Acceptable) - internal positive class probability
    """
    label_map = {REPORT_LABEL_POS: 1, REPORT_LABEL_NEG: 0}
    y_score_acc = 1.0 - y_score_na

    organoid_results = []
    for idx in range(len(selected_test_df)):
        org_id = selected_test_df.iloc[idx]["ID"]
        true_label_str = selected_test_df.iloc[idx]["label"]
        true_label = label_map.get(true_label_str, 0)
        
        pred_label_str = y_pred[idx]
        pred_label = label_map.get(pred_label_str, 0)
        
        pred_prob = float(y_score_acc[idx])
        correct = pred_label == true_label

        if true_label == 1 and pred_label == 1:
            cm_category = "TP"
        elif true_label == 0 and pred_label == 1:
            cm_category = "FP"
        elif true_label == 1 and pred_label == 0:
            cm_category = "FN"
        else:
            cm_category = "TN"

        organoid_results.append({
            "Organoid_ID": org_id,
            "True_Label": true_label,
            "Predicted_Probability": pred_prob,
            "Predicted_Label": pred_label,
            "Correct": correct,
            "CM_Category": cm_category,
        })

    organoid_preds_df = pd.DataFrame(organoid_results)
    organoid_preds_df.to_csv(output_path, index=False)
    print(f"  Saved organoid predictions to {output_path}")


def prepare_data_for_day(df, day_num, cols_to_drop_base):
    """Prepare features, labels, and groups for a specific day."""
    df_day = df.copy()
    cols_to_drop = cols_to_drop_base.copy()

    if day_num <= 10 and "MalateGlo_concentration_uM" in df_day.columns:
        cols_to_drop.append("MalateGlo_concentration_uM")

    growth_features = [
        "glucose_growth", "glutamate_growth", "LactateGlo_growth",
        "PyruvateGlo_growth", "MalateGlo_growth",
    ]
    accel_features = [f"{g}_accel" for g in growth_features]
    if day_num == 3:
        cols_to_drop.extend([g for g in growth_features + accel_features if g in df_day.columns])
    elif day_num <= 6:
        # Second-order needs at least 3 timepoints; drop accel at Dy06
        cols_to_drop.extend([g for g in accel_features if g in df_day.columns])
        if day_num == 6 and "MalateGlo_growth" in df_day.columns:
            pass  # keep first-order at Dy06
    elif day_num == 13 and "MalateGlo_growth" in df_day.columns:
        cols_to_drop.append("MalateGlo_growth")
        if "MalateGlo_growth_accel" in df_day.columns:
            cols_to_drop.append("MalateGlo_growth_accel")

    df_day = df_day.drop(columns=[c for c in cols_to_drop if c in df_day.columns])

    X = df_day.drop(columns=["label", "ID"])
    y = df_day["label"]
    groups = df_day["ID"]

    return X, y, groups


def clean_data(X_train, X_val=None, X_test=None):
    """Clean NaNs/constants without scaling."""
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        X_train = X_train.drop(columns=all_nan_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in all_nan_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in all_nan_cols if c in X_test.columns])

    constant_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=True) <= 1]
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

    near_constant_cols = []
    for col in X_train.columns:
        col_std = X_train[col].std(skipna=True)
        if np.isfinite(col_std) and col_std < 1e-6:
            near_constant_cols.append(col)
    if near_constant_cols:
        X_train = X_train.drop(columns=near_constant_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in near_constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in near_constant_cols if c in X_test.columns])

    if X_train.isna().any().any():
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)

    return X_train, X_val, X_test


def get_scoring_function(scoring):
    """Get sklearn scorer for GridSearchCV."""
    if scoring == "f1_weighted":
        return "f1_weighted"
    elif scoring == "f1_notacceptable":
        return make_scorer(f1_score, pos_label=TRAIN_LABEL_POS)
    elif scoring == "recall_notacceptable":
        return make_scorer(recall_score, pos_label=TRAIN_LABEL_POS)
    else:
        raise ValueError(f"Unknown scoring: {scoring}")


def tune_threshold_on_val(scores_acc, labels_str, scoring):
    """
    Tune threshold on VALIDATION predictions.
    scores_acc: P(Acceptable) from model.predict_proba
    labels_str: string labels ("Acceptable" / "Not Acceptable")
    
    Returns: (best_threshold, best_score)
    """
    scores_acc = np.asarray(scores_acc)
    labels_bin = (np.asarray(labels_str) == TRAIN_LABEL_NEG).astype(int)
    
    thresholds = np.linspace(0.3, 0.7, 9)
    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred_bin = (scores_acc >= t).astype(int)

        if scoring == "f1_weighted":
            score = f1_score(labels_bin, y_pred_bin, average="weighted", zero_division=0)
        elif scoring == "f1_notacceptable":
            score = f1_score(labels_bin, y_pred_bin, pos_label=0, zero_division=0)
        elif scoring == "recall_notacceptable":
            score = recall_score(labels_bin, y_pred_bin, pos_label=0, zero_division=0)
        else:
            score = f1_score(labels_bin, y_pred_bin, zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


def save_calibration_diagnostic(model_dir, val_scores_all, y_val_bin_all):
    """Save calibration diagnostic plots and CSV."""
    val_scores_all = np.asarray(val_scores_all)
    y_val_bin_all = np.asarray(y_val_bin_all)

    if len(val_scores_all) == 0:
        print("No validation data for calibration diagnostic.")
        return

    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(val_scores_all, bins) - 1
    records = []
    mean_preds = []
    frac_pos = []

    for b in range(10):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        scores_bin = val_scores_all[mask]
        y_bin = y_val_bin_all[mask]
        mean_pred = float(scores_bin.mean())
        frac_positive = float(y_bin.mean())
        count = int(mask.sum())
        left = float(bins[b])
        right = float(bins[b + 1])

        records.append({
            "bin_left": left,
            "bin_right": right,
            "mean_pred": mean_pred,
            "frac_positive": frac_positive,
            "count": count,
        })
        mean_preds.append(mean_pred)
        frac_pos.append(frac_positive)

    calib_df = pd.DataFrame(records)
    calib_path_csv = Path(model_dir) / "calibration_bins.csv"
    calib_df.to_csv(calib_path_csv, index=False)
    print(f"Saved calibration bins to {calib_path_csv}")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if len(mean_preds) > 0:
        plt.plot(mean_preds, frac_pos, "o-", label="Model")
    plt.xlabel("Mean predicted probability (Acceptable)")
    plt.ylabel("Empirical fraction Acceptable")
    plt.title("Calibration Curve (Validation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    calib_path_png = Path(model_dir) / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(calib_path_png, dpi=150)
    plt.close()
    print(f"Saved calibration curve to {calib_path_png}")


def build_model(imbalance, y_train, print_weights=False):
    """Build LightGBM model with specified imbalance handling."""
    classes = np.unique(y_train)
    
    if len(classes) < 2:
        class_weight_dict = None
        scale_pos_weight_val = 1.0
    else:
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}
        
        y_arr = np.asarray(y_train)
        n_pos = (y_arr == TRAIN_LABEL_POS).sum()
        n_neg = (y_arr == TRAIN_LABEL_NEG).sum()
        scale_pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
        
        if print_weights and imbalance in ["scale_pos_weight", "both"]:
            print(f"    scale_pos_weight = {scale_pos_weight_val:.2f} (NA={n_pos}, Acc={n_neg})")

    if imbalance == "class_weight":
        return LGBMClassifier(
            random_state=SEED, verbose=-1, n_jobs=1, device="cpu",
            class_weight=class_weight_dict, boosting_type="gbdt"
        )
    elif imbalance == "scale_pos_weight":
        return LGBMClassifier(
            random_state=SEED, verbose=-1, n_jobs=1, device="cpu",
            scale_pos_weight=scale_pos_weight_val, boosting_type="gbdt"
        )
    elif imbalance == "both":
        return LGBMClassifier(
            random_state=SEED, verbose=-1, n_jobs=1, device="cpu",
            class_weight=class_weight_dict,
            scale_pos_weight=scale_pos_weight_val, boosting_type="gbdt"
        )
    else:
        raise ValueError(f"Unknown imbalance method: {imbalance}")


def train_metabolite_classifier_per_day(
    train_df, val_df, test_df, output_dir, model_name, imbalance, scoring
):
    """Train LightGBM classifier for each day."""
    set_seed()

    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    unique_days = sorted(np.unique(train_df.DY))
    per_day_info = {}

    print(f"\n{'=' * 60}")
    print(f"Training Metabolite Classifier (CPU)")
    print(f"Model name: {model_name}")
    print(f"Imbalance: {imbalance} | Scoring: {scoring}")
    print(f"CV Folds: {N_FOLDS}")
    print(f"Threshold: validation-based, grid 0.3-0.7 (9 steps)")
    print(f"{'=' * 60}\n")

    cols_to_drop_base = [
        "DY", "batch", "img_path", "mask_path",
        "MalateGlo_initial_concentration", "GlucoseGlo_initial_concentration",
        "GlutamateGlo_initial_concentration", "LactateGlo_initial_concentration",
        "PyruvateGlo_initial_concentration", "day",
    ]

    # ========== PHASE 1: GridSearchCV on train ==========
    print("[PHASE 1] Hyperparameter tuning on TRAIN data")
    
    for days in unique_days:
        day_train = train_df[train_df["DY"] == days].copy()

        if len(day_train) == 0:
            print(f"  {days}: no training data, skipping.")
            continue

        day_num = int(re.search(r"\d+", days).group())
        n_acc = (day_train["label"] == TRAIN_LABEL_NEG).sum()
        n_na = (day_train["label"] == TRAIN_LABEL_POS).sum()
        
        print(f"\n  [{days}] Train n={len(day_train)} (Acc={n_acc}, NA={n_na})")

        X_train, y_train, groups_train = prepare_data_for_day(
            day_train, day_num, cols_to_drop_base
        )
        
        X_train_clean, _, _ = clean_data(X_train)

        if X_train_clean.shape[1] == 0:
            print(f"    No features left after cleaning; skipping {days}.")
            continue

        _ = build_model(imbalance, y_train, print_weights=True)
        model = build_model(imbalance, y_train)

        param_grid = {
            "max_depth": [3, 6],
            "num_leaves": [31, 47, 63],
            "min_child_samples": [10, 20],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
        }

        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        scoring_fn = get_scoring_function(scoring)

        grid = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring_fn, n_jobs=4, verbose=0
        )
        grid.fit(X_train_clean, y_train, groups=groups_train)

        best_params = grid.best_params_
        print(f"    Best Params: {best_params}")

        per_day_info[days] = {
            "day_num": day_num,
            "best_params": best_params,
        }

    # ========== PHASE 2: Tune threshold on VALIDATION ==========
    print(f"\n[PHASE 2] Threshold tuning on VALIDATION data (grid: 0.3-0.7)")
    
    thresholds_per_day = {}
    all_val_scores = []
    all_val_labels = []
    
    for days in unique_days:
        if days not in per_day_info:
            continue
            
        day_num = per_day_info[days]["day_num"]
        day_train = train_df[train_df["DY"] == days].copy()
        day_val = val_df[val_df["DY"] == days].copy()

        if len(day_val) == 0:
            print(f"  {days}: no val data, using default threshold=0.5")
            thresholds_per_day[days] = 0.5
            continue

        X_train, y_train, _ = prepare_data_for_day(day_train, day_num, cols_to_drop_base)
        X_val, y_val, _ = prepare_data_for_day(day_val, day_num, cols_to_drop_base)

        X_train_clean, X_val_clean, _ = clean_data(X_train, X_val=X_val)

        model = build_model(imbalance, y_train)
        model.set_params(**per_day_info[days]["best_params"])
        model.fit(X_train_clean, y_train)

        val_proba = model.predict_proba(X_val_clean)
        classes_order = list(model.classes_)
        
        if TRAIN_LABEL_NEG in classes_order:
            acc_idx = classes_order.index(TRAIN_LABEL_NEG)
            scores_acc = val_proba[:, acc_idx]
        else:
            scores_acc = val_proba[:, 0]

        y_val_arr = y_val.to_numpy() if hasattr(y_val, 'to_numpy') else np.asarray(y_val)
        
        if len(np.unique(y_val_arr)) > 1:
            threshold, thresh_score = tune_threshold_on_val(scores_acc, y_val_arr, scoring)
            print(f"  {days}: threshold={threshold:.3f}, {scoring}={thresh_score:.3f}")
            
            labels_bin = (y_val_arr == TRAIN_LABEL_NEG).astype(int)
            all_val_scores.extend(scores_acc.tolist())
            all_val_labels.extend(labels_bin.tolist())
        else:
            threshold = 0.5
            print(f"  {days}: single-class val, using threshold=0.5")

        thresholds_per_day[days] = threshold
        per_day_info[days]["threshold"] = threshold

    save_calibration_diagnostic(model_dir, all_val_scores, all_val_labels)

    # ========== PHASE 3: Refit on train+val, evaluate on TEST ==========
    print(f"\n[PHASE 3] Final refit on TRAIN+VAL, evaluate on TEST")
    
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    for days in unique_days:
        if days not in per_day_info:
            continue

        info = per_day_info[days]
        day_num = info["day_num"]
        threshold = info.get("threshold", 0.5)

        day_trainval = trainval_df[trainval_df["DY"] == days].copy()
        day_test = test_df[test_df["DY"] == days].copy()

        if len(day_test) == 0:
            print(f"  {days}: no test data, skipping.")
            continue

        print(f"\n  [{days}] Trainval n={len(day_trainval)}, Test n={len(day_test)}")

        X_trainval, y_trainval, _ = prepare_data_for_day(
            day_trainval, day_num, cols_to_drop_base
        )
        X_test, y_test, _ = prepare_data_for_day(day_test, day_num, cols_to_drop_base)

        X_trainval_clean, _, X_test_clean = clean_data(X_trainval, X_test=X_test)

        if X_trainval_clean.shape[1] == 0:
            print(f"    No features; skipping {days}.")
            continue

        final_model = build_model(imbalance, y_trainval)
        final_model.set_params(**info["best_params"])
        final_model.fit(X_trainval_clean, y_trainval)

        # Feature importance
        feature_importance = final_model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": X_trainval_clean.columns,
            "importance": feature_importance,
            "importance_normalized": feature_importance / feature_importance.sum()
            if feature_importance.sum() > 0 else 0,
        }).sort_values("importance", ascending=False)

        day_dir = model_dir / days
        day_dir.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(day_dir / "feature_importance.csv", index=False)
        info["feature_importance_df"] = importance_df
        
        top_n = min(20, len(importance_df))
        plt.figure(figsize=(10, 8))
        top_importance = importance_df.head(top_n)
        plt.barh(range(top_n), top_importance["importance"].values, color="steelblue")
        plt.yticks(range(top_n), top_importance["feature"].values)
        plt.xlabel("Importance (split count)", fontweight="bold")
        plt.ylabel("Feature", fontweight="bold")
        plt.title(f"Top {top_n} Feature Importances - {days}", fontweight="bold", fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(day_dir / "feature_importance_top20.png", dpi=150)
        plt.close()

        # Test predictions
        test_proba = final_model.predict_proba(X_test_clean)
        classes_order = list(final_model.classes_)
        
        if TRAIN_LABEL_NEG in classes_order:
            acc_idx = classes_order.index(TRAIN_LABEL_NEG)
            y_score_acc = test_proba[:, acc_idx]
        else:
            y_score_acc = test_proba[:, 0]
        
        y_score_na = 1.0 - y_score_acc

        y_pred_test = np.where(y_score_acc >= threshold, REPORT_LABEL_POS, REPORT_LABEL_NEG)

        y_test_arr = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)
        
        # ROC-AUC & PR-AUC
        if len(np.unique(y_test_arr)) > 1:
            y_true_bin_acc = (y_test_arr == REPORT_LABEL_POS).astype(int)
            roc_auc = roc_auc_score(y_true_bin_acc, y_score_acc)
            pr_auc = average_precision_score(y_true_bin_acc, y_score_acc)
        else:
            roc_auc = None
            pr_auc = None

        accuracy = accuracy_score(y_test_arr, y_pred_test)
        report = classification_report(y_test_arr, y_pred_test, output_dict=True, zero_division=0)

        precision_accept = report.get(REPORT_LABEL_POS, {}).get("precision", 0)
        precision_notaccept = report.get(REPORT_LABEL_NEG, {}).get("precision", 0)
        recall_accept = report.get(REPORT_LABEL_POS, {}).get("recall", 0)
        recall_notaccept = report.get(REPORT_LABEL_NEG, {}).get("recall", 0)
        f1_accept = report.get(REPORT_LABEL_POS, {}).get("f1-score", 0)
        f1_notaccept = report.get(REPORT_LABEL_NEG, {}).get("f1-score", 0)

        cm = confusion_matrix(y_test_arr, y_pred_test, labels=[REPORT_LABEL_POS, REPORT_LABEL_NEG])
        
        if cm.shape == (2, 2):
            tp = cm[0, 0]
            fn = cm[0, 1]
            fp = cm[1, 0]
            tn = cm[1, 1]
        else:
             tp, fn, fp, tn = 0, 0, 0, 0

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"    Recall_NA={recall_notaccept:.3f}, Prec_NA={precision_notaccept:.3f}, "
              f"F1_NA={f1_notaccept:.3f}, Spec={specificity:.3f}, Thr={threshold:.2f}")

        save_organoid_predictions(
            day_test.reset_index(drop=True), y_test_arr, y_pred_test, y_score_na,
            day_dir / "organoid_predictions.csv"
        )

        metrics = {
            "day": days,
            "day_no": day_num,
            "test_accuracy": float(accuracy),
            "test_f1_acceptable": float(f1_accept),
            "test_f1_notacceptable": float(f1_notaccept),
            "test_recall_acceptable": float(recall_accept),
            "test_recall_notacceptable": float(recall_notaccept),
            "test_precision_acceptable": float(precision_accept),
            "test_precision_notacceptable": float(precision_notaccept),
            "test_specificity": float(specificity),
            "test_sensitivity": float(sensitivity),
            "test_roc_auc": float(roc_auc) if roc_auc else None,
            "test_pr_auc": float(pr_auc) if pr_auc else None,
            "best_params": info["best_params"],
            "threshold_used": float(threshold),
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
            "seed": SEED,
            "threshold_grid": {"min": 0.3, "max": 0.7, "n": 9},
            "imbalance_method": imbalance,
            "scoring_metric": scoring,
        }

        with open(day_dir / "metrics_test.json", "w") as f:
            json.dump(metrics, f, indent=2)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix - {days}")
        plt.colorbar()
        plt.xticks([0, 1], [REPORT_LABEL_POS, REPORT_LABEL_NEG], rotation=45)
        plt.yticks([0, 1], [REPORT_LABEL_POS, REPORT_LABEL_NEG])
        thresh_cm = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                         color="white" if cm[i, j] > thresh_cm else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(day_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        results_summary.append({
            "Day": days, "Day_No": day_num,
            "Test_Accuracy": accuracy,
            "Test_F1_Acceptable": f1_accept,
            "Test_F1_NotAcceptable": f1_notaccept,
            "Test_Recall_Acceptable": recall_accept,
            "Test_Recall_NotAcceptable": recall_notaccept,
            "Test_Precision_Acceptable": precision_accept,
            "Test_Precision_NotAcceptable": precision_notaccept,
            "Test_Specificity": specificity,
            "Test_Sensitivity": sensitivity,
            "Test_ROC_AUC": roc_auc,
            "Test_PR_AUC": pr_auc,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Threshold_Used": threshold,
        })

    if not results_summary:
        print("\nWarning: No results to summarize")
        return

    summary_df = pd.DataFrame(results_summary).sort_values("Day_No")
    summary_df.to_csv(model_dir / "results_summary.csv", index=False)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(summary_df[["Day", "Test_F1_NotAcceptable", "Test_Recall_NotAcceptable", 
                       "Test_Specificity", "TP", "FP", "FN", "TN", "Threshold_Used"]].to_string(index=False))

    # --- Metrics by Day Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(summary_df["Day_No"], summary_df["Test_F1_NotAcceptable"], "o-", color="orange")
    axes[0, 0].set_title("Test F1 Score (Not Acceptable)")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    axes[0, 1].plot(summary_df["Day_No"], summary_df["Test_F1_Acceptable"], "o-", color="blue")
    axes[0, 1].set_title("Test F1 Score (Acceptable)")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    axes[1, 0].plot(summary_df["Day_No"], summary_df["Test_Specificity"], "o-", color="purple")
    axes[1, 0].set_title("Test Specificity (TNR)")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Specificity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # PR-AUC plot (replacing ROC-AUC)
    auc_data = summary_df.dropna(subset=["Test_PR_AUC"])
    if len(auc_data) > 0:
        axes[1, 1].plot(auc_data["Day_No"], auc_data["Test_PR_AUC"], "o-", color="green")
        axes[1, 1].set_title("Test PR-AUC")
        axes[1, 1].set_xlabel("Day")
        axes[1, 1].set_ylabel("PR-AUC")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(model_dir / "metrics_by_day.png", dpi=150)
    plt.close()
    print(f"Saved metrics plot to {model_dir / 'metrics_by_day.png'}")

    # --- Feature Importance Summary ---
    all_features = set()
    for days in unique_days:
        if days in per_day_info and "feature_importance_df" in per_day_info[days]:
            feat_df = per_day_info[days]["feature_importance_df"]
            all_features.update(feat_df["feature"].values)

    feature_summary_data = []
    for feature in all_features:
        row = {"feature": feature}
        importance_values = []

        for days in unique_days:
            if days not in per_day_info or "feature_importance_df" not in per_day_info[days]:
                continue
            day_num = per_day_info[days]["day_num"]
            feat_df = per_day_info[days]["feature_importance_df"]
            feature_row = feat_df[feat_df["feature"] == feature]

            if len(feature_row) > 0:
                importance = feature_row["importance"].values[0]
                row[f"Day_{day_num}_importance"] = importance
                importance_values.append(importance)
            else:
                row[f"Day_{day_num}_importance"] = 0.0

        if importance_values:
            row["avg_importance"] = np.mean(importance_values)
            row["total_importance"] = np.sum(importance_values)
            row["num_days_used"] = len(importance_values)
        else:
            row["avg_importance"] = 0.0
            row["total_importance"] = 0.0
            row["num_days_used"] = 0

        feature_summary_data.append(row)

    if feature_summary_data:
        feature_summary_df = pd.DataFrame(feature_summary_data)
        feature_summary_df = feature_summary_df.sort_values("avg_importance", ascending=False)
        feature_summary_df.to_csv(model_dir / "feature_importance_summary.csv", index=False)
        
        top_features = feature_summary_df.head(15)
        day_cols = [c for c in feature_summary_df.columns if c.startswith("Day_") and c.endswith("_importance")]
        
        if len(day_cols) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            x = np.arange(len(top_features))
            width = 0.8 / len(day_cols)

            for i, day_col in enumerate(day_cols):
                day_num = day_col.replace("Day_", "").replace("_importance", "")
                offset = (i - len(day_cols) / 2) * width + width / 2
                ax.bar(x + offset, top_features[day_col].values, width, label=f"Day {day_num}")

            ax.set_xlabel("Feature", fontweight="bold", fontsize=12)
            ax.set_ylabel("Importance", fontweight="bold", fontsize=12)
            ax.set_title("Top 15 Features: Importance Across Days", fontweight="bold", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(top_features["feature"].values, rotation=45, ha="right")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(model_dir / "feature_importance_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"\n{'=' * 60}")
    print(f"Training Complete! Results saved to {model_dir}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Metabolite Classifiers (CPU Version)"
    )
    parser.add_argument(
        "--imbalance",
        choices=["class_weight", "scale_pos_weight", "both"],
        default="class_weight",
        help="Class imbalance handling method (default: class_weight)",
    )
    parser.add_argument(
        "--scoring",
        choices=["f1_weighted", "f1_notacceptable", "recall_notacceptable"],
        default="f1_notacceptable",
        help="Scoring metric for CV and threshold tuning (default: f1_notacceptable)",
    )
    parser.add_argument(
        "--growth-order",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Growth rate feature order: 0=none, 1=first-order diffs, 2=first+second-order (default: 1)",
    )

    args = parser.parse_args()

    train_data_path = "data_splits/both_train_base.json"
    val_data_path = "data_splits/both_val_base.json"
    test_data_path = "data_splits/both_test_base.json"
    output_dir = "analysis/metabolites/LightGBM/outputs_metabolites"

    print(f"\n{'=' * 60}")
    print("Loading data splits...")
    print(f"{'=' * 60}")

    with open(train_data_path, "r") as f:
        train_data_json = json.load(f)
    with open(val_data_path, "r") as f:
        val_data_json = json.load(f)
    with open(test_data_path, "r") as f:
        test_data_json = json.load(f)

    train_df = json_to_df(train_data_json)
    val_df = json_to_df(val_data_json)
    test_df = json_to_df(test_data_json)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if args.growth_order >= 1:
        print("\nComputing first-order growth features...")
        train_df = compute_growth_features(train_df)
        val_df = compute_growth_features(val_df)
        test_df = compute_growth_features(test_df)
    else:
        print("\nSkipping growth features (--growth-order 0)")
        # Still need the 'day' column for prepare_data_for_day
        for df in [train_df, val_df, test_df]:
            df["day"] = df["DY"].str.extract(r"(\d+)").astype(int)

    if args.growth_order >= 2:
        print("Computing second-order growth features (acceleration)...")
        train_df = compute_second_order_growth_features(train_df)
        val_df = compute_second_order_growth_features(val_df)
        test_df = compute_second_order_growth_features(test_df)

    growth_suffix = f"_g{args.growth_order}"
    model_name = f"lgbm_cpu_{args.imbalance}_{args.scoring}{growth_suffix}"

    train_metabolite_classifier_per_day(
        train_df, val_df, test_df, output_dir,
        model_name=model_name,
        imbalance=args.imbalance,
        scoring=args.scoring,
    )


if __name__ == "__main__":
    main()
