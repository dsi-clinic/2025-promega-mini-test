#!/usr/bin/env python3
"""
Metabolite Organoid Quality Classification
Trains per-day classifiers using LightGBM with metabolite features.

Usage:
    python train_metabolites.py                                     # Runs default preset (per_day_noscale_main)
    python train_metabolites.py --preset per_day_noscale_classweight # Runs specific preset
    python train_metabolites.py --preset per_day_noscale_main --cv_scoring balanced_accuracy # Override preset setting
    python train_metabolites.py --preset per_day_noscale_main --preset per_day_balacc # Run multiple presets

Presets:
    - per_day_noscale_main (DEFAULT): No scaling, both weights, f1_weighted
    - per_day_noscale_classweight: No scaling, class_weight_only, f1_weighted
    - per_day_scale_pos: Scaled, scale_pos_only, f1_weighted
    - per_day_balacc: Scaled, both weights, balanced_accuracy
    - per_day_f1_notaccept: Scaled, both weights, f1_notaccept
    - per_day_baseline: Scaled, both weights, f1_weighted (Original baseline)
"""

import os
import json
import re
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, make_scorer
)
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

SEED = 42


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

            # Add metabolites
            for k, v in tp.get("metabolites", {}).items():
                row[k] = v

            rows.append(row)

    return pd.DataFrame(rows)


def compute_growth_features(df):
    """Add growth features (difference between consecutive timepoints)."""
    df = df.copy()
    df['day'] = df['DY'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(['ID', 'day'])

    # Compute growth features by organoid ID
    df['glucose_growth'] = df.groupby('ID')['GlucoseGlo_concentration_uM'].diff()
    df['glutamate_growth'] = df.groupby('ID')['GlutamateGlo_concentration_uM'].diff()
    df['LactateGlo_growth'] = df.groupby('ID')['LactateGlo_concentration_uM'].diff()
    df['PyruvateGlo_growth'] = df.groupby('ID')['PyruvateGlo_concentration_uM'].diff()
    df['MalateGlo_growth'] = df.groupby('ID')['MalateGlo_concentration_uM'].diff()

    return df


def save_organoid_predictions(selected_test_df, y_test, y_pred, y_score, output_path):
    """
    Save per-organoid predictions to CSV in the same format as multimodal.
    """
    label_map = {"Acceptable": 1, "Not Acceptable": 0}

    organoid_results = []
    for idx in range(len(selected_test_df)):
        org_id = selected_test_df.iloc[idx]['ID']
        true_label_str = selected_test_df.iloc[idx]['label']
        true_label = label_map.get(true_label_str, 0)
        pred_label_str = y_pred[idx]
        pred_label = label_map.get(pred_label_str, 0)
        pred_prob = float(y_score[idx])
        correct = (pred_label == true_label)

        if true_label == 1 and pred_label == 1:
            cm_category = 'TP'
        elif true_label == 0 and pred_label == 1:
            cm_category = 'FP'
        elif true_label == 1 and pred_label == 0:
            cm_category = 'FN'
        else:
            cm_category = 'TN'

        organoid_results.append({
            'Organoid_ID': org_id,
            'True_Label': true_label,
            'Predicted_Probability': pred_prob,
            'Predicted_Label': pred_label,
            'Correct': correct,
            'CM_Category': cm_category
        })

    organoid_preds_df = pd.DataFrame(organoid_results)
    organoid_preds_df.to_csv(output_path, index=False)
    print(f"  Saved organoid predictions to {output_path}")


def prepare_data_for_day(df, day_num, cols_to_drop_base):
    """
    Helper to prepare X, y, groups for a specific day.
    Returns: X, y, groups
    """
    df_day = df.copy()
    cols_to_drop = cols_to_drop_base.copy()

    # For days <= 10, also drop Malate concentration
    if day_num <= 10 and 'MalateGlo_concentration_uM' in df_day.columns:
        cols_to_drop.append('MalateGlo_concentration_uM')

    # Drop growth features for day 3 (no previous timepoint)
    growth_features = ['glucose_growth', 'glutamate_growth', 'LactateGlo_growth',
                       'PyruvateGlo_growth', 'MalateGlo_growth']
    if day_num == 3:
        cols_to_drop.extend([g for g in growth_features if g in df_day.columns])
    elif day_num == 13 and 'MalateGlo_growth' in df_day.columns:
        cols_to_drop.append('MalateGlo_growth')

    df_day = df_day.drop(columns=[c for c in cols_to_drop if c in df_day.columns])

    X = df_day.drop(columns=["label", "ID"])
    y = df_day["label"]
    groups = df_day["ID"]

    return X, y, groups


def clean_and_scale_data(X_train, X_val=None, X_test=None):
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
            X_test = X_test.drop(columns=[c for c in all_nan_cols if c in X_test.columns])

    # Drop constant columns
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"  Dropping constant columns: {constant_cols}")
        X_train = X_train.drop(columns=constant_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

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
            X_val = X_val.drop(columns=[c for c in near_constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in near_constant_cols if c in X_test.columns])

    # Fill NaNs
    if X_train.isna().any().any():
        print("  Filling remaining NaNs with 0")
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def clean_data_no_scaling(X_train, X_val=None, X_test=None):
    """
    Clean NaNs/constants but DO NOT scale.
    Returns raw cleaned DataFrames.
    """
    # Drop all-NaN columns
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"  Dropping all-NaN columns: {all_nan_cols}")
        X_train = X_train.drop(columns=all_nan_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in all_nan_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in all_nan_cols if c in X_test.columns])

    # Drop constant columns
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"  Dropping constant columns: {constant_cols}")
        X_train = X_train.drop(columns=constant_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

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
            X_val = X_val.drop(columns=[c for c in near_constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in near_constant_cols if c in X_test.columns])

    # Fill NaNs
    if X_train.isna().any().any():
        print("  Filling remaining NaNs with 0")
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)

    return X_train, X_val, X_test


def save_calibration_diagnostic(model_dir, val_scores_all, y_val_bin_all):
    """
    Save a simple calibration diagnostic:
    - calibration_bins.csv
    - calibration_curve.png
    Based on VALIDATION predictions (pooled across all days).
    """
    val_scores_all = np.asarray(val_scores_all)
    y_val_bin_all = np.asarray(y_val_bin_all)

    if len(val_scores_all) == 0:
        print("No validation data for calibration diagnostic.")
        return

    # 10 bins in [0, 1]
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(val_scores_all, bins) - 1  # 0..9
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
            "count": count
        })
        mean_preds.append(mean_pred)
        frac_pos.append(frac_positive)

    calib_df = pd.DataFrame(records)
    calib_path_csv = Path(model_dir) / "calibration_bins.csv"
    calib_df.to_csv(calib_path_csv, index=False)
    print(f"Saved calibration bins to {calib_path_csv}")

    # Plot reliability curve
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    if len(mean_preds) > 0:
        plt.plot(mean_preds, frac_pos, 'o-', label='Model')
    plt.xlabel('Mean predicted probability (Acceptable)')
    plt.ylabel('Empirical fraction Acceptable')
    plt.title('Calibration Curve (Validation)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    calib_path_png = Path(model_dir) / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(calib_path_png, dpi=150)
    plt.close()
    print(f"Saved calibration curve to {calib_path_png}")


def train_metabolite_classifier_per_day(
    train_df,
    val_df,
    test_df,
    output_dir,
    model_name="lgbm",
    cv_scoring="f1_weighted",      # "f1_weighted", "f1_notaccept", "macro_f1"
    threshold_metric="f1_weighted" # "f1_weighted", "f1_notaccept", "macro_f1"
):
    """
    Train LightGBM classifier for each day and save detailed results.
    Fixed parameters:
      - boosting_type="gbdt"
      - threshold_mode="per_day"
      - weight_mode="both"
      - use_scaling=False
    """
    set_seed()

    # Fixed parameters
    boosting_type = "gbdt"
    threshold_mode = "per_day"
    weight_mode = "both"
    use_scaling = False

    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    unique_days = sorted(np.unique(train_df.DY))

    print(f"\n{'='*60}")
    print(f"Training Metabolite Classifier ({model_name.upper()})")
    print(f"Boosting type       : {boosting_type}")
    print(f"Threshold mode      : {threshold_mode}")
    print(f"Weight mode         : {weight_mode}")
    print(f"Use scaling         : {use_scaling}")
    print(f"CV scoring          : {cv_scoring}")
    print(f"Threshold metric    : {threshold_metric}")
    print(f"{'='*60}\n")

    # Base columns to drop (day-specific logic is handled in prepare_data_for_day)
    cols_to_drop_base = [
        "DY", 'batch', 'img_path', 'mask_path',
        'MalateGlo_initial_concentration',
        'GlucoseGlo_initial_concentration',
        'GlutamateGlo_initial_concentration',
        'LactateGlo_initial_concentration',
        'PyruvateGlo_initial_concentration',
        'day'
    ]

    # ----- PHASE 1: Hyperparameter tuning + VAL predictions (for thresholds/calibration) -----

    per_day_info = {}  # store per-day best_params, weights, etc.

    for days in unique_days:
        day_train = train_df[train_df['DY'] == days].copy()
        day_val = val_df[val_df['DY'] == days].copy()

        if len(day_train) == 0:
            print(f"{days}: no training data, skipping.")
            continue

        day_num = int(re.search(r'\d+', days).group())
        print(f"\n{'-'*60}")
        print(f"[PHASE 1] Day {days} (day_num={day_num})")
        print(f"  Train: {len(day_train)}, Val: {len(day_val)}")

        X_train, y_train, groups_train = prepare_data_for_day(day_train, day_num, cols_to_drop_base)
        X_val, y_val, _ = prepare_data_for_day(day_val, day_num, cols_to_drop_base)

        # Clean + scale (or just clean) based on train, transform val
        if use_scaling:
            X_train_scaled, X_val_scaled, _, _ = clean_and_scale_data(X_train, X_val=X_val)
        else:
            X_train_scaled, X_val_scaled, _ = clean_data_no_scaling(X_train, X_val=X_val)

        if X_train_scaled.shape[1] == 0:
            print(f"  No features left after cleaning; skipping {days}.")
            continue

        # Class weights & scale_pos_weight
        classes = np.unique(y_train)
        if len(classes) < 2:
            # Degenerate case: only one class in training
            class_weight_dict_balanced = None
            ratio_balanced = 1.0
        else:
            class_weights_balanced = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict_balanced = {cls: float(w) for cls, w in zip(classes, class_weights_balanced)}

            pos_label = "Acceptable"
            y_arr = pd.Series(y_train).to_numpy()
            pos = (y_arr == pos_label).sum()
            neg = (y_arr != pos_label).sum()
            ratio_balanced = (neg / pos) if pos > 0 else 1.0

        # weight_mode="both"
        final_class_weight = class_weight_dict_balanced
        final_scale_pos_weight = ratio_balanced

        model = LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            scale_pos_weight=final_scale_pos_weight,
            class_weight=final_class_weight,
            boosting_type=boosting_type
        )

        param_grid = {
            'max_depth': [3, 6],
            'num_leaves': [31, 63],
            'min_child_samples': [10, 20],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [200, 500],
        }

        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        
        if cv_scoring == "f1_weighted":
            scoring_obj = "f1_weighted"
        elif cv_scoring == "balanced_accuracy":
            scoring_obj = "balanced_accuracy"
        elif cv_scoring == "f1_notaccept":
            scoring_obj = make_scorer(f1_score, pos_label="Not Acceptable")
        elif cv_scoring == "macro_f1":
            scoring_obj = "f1_macro"
        else:
            raise ValueError(f"Unknown cv_scoring: {cv_scoring}")

        grid = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring_obj,
            n_jobs=4,
            verbose=0
        )
        grid.fit(X_train_scaled, y_train, groups=groups_train)
        best_params = grid.best_params_
        print(f"  Best Params: {best_params}")

        per_day_info[days] = {
            "day_num": day_num,
            "best_params": best_params,
            "class_weight_dict": final_class_weight,
            "scale_pos_weight": final_scale_pos_weight,
        }

    # ----- Choose thresholds based on VALIDATION predictions -----

    thresholds_per_day = {}
    default_threshold = 0.5

    def tune_threshold(scores, labels, metric="f1_weighted"):
        """
        Tune threshold based on validation scores and labels.
        labels: 1=Acceptable, 0=Not Acceptable
        metric: "f1_weighted", "f1_notaccept", "macro_f1"
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels) # 1=Acceptable, 0=Not Acceptable
        thresholds = np.linspace(0.1, 0.9, 17)
        best_t, best_score = 0.5, -1.0
        
        for t in thresholds:
            # Predict Acceptable (1) if score >= t
            y_pred_bin = (scores >= t).astype(int)
            
            if metric == "f1_weighted":
                score = f1_score(labels, y_pred_bin, average='weighted', zero_division=0)
            elif metric == "f1_notaccept":
                # labels are 1=Acceptable, 0=Not Acceptable
                # We want F1 for Not Acceptable (class 0)
                score = f1_score(labels, y_pred_bin, pos_label=0, zero_division=0)
            elif metric == "macro_f1":
                score = f1_score(labels, y_pred_bin, average='macro', zero_division=0)
            else:
                # Fallback, should not happen with valid metric choices
                score = f1_score(labels, y_pred_bin, zero_division=0) 

            if score > best_score:
                best_score = score
                best_t = t
        return best_t, best_score

    print(f"\n[THRESHOLDS] Mode = per_day, Metric = {threshold_metric}")
    per_day_val_scores = {}
    per_day_val_labels = {}

    for days in unique_days:
        if days not in per_day_info:
            continue
        day_num = per_day_info[days]["day_num"]
        day_train = train_df[train_df['DY'] == days].copy()
        day_val = val_df[val_df['DY'] == days].copy()

        if len(day_val) == 0:
            continue

        X_train, y_train, groups_train = prepare_data_for_day(day_train, day_num, cols_to_drop_base)
        X_val, y_val, _ = prepare_data_for_day(day_val, day_num, cols_to_drop_base)
        
        if use_scaling:
            X_train_scaled, X_val_scaled, _, _ = clean_and_scale_data(X_train, X_val=X_val)
        else:
            X_train_scaled, X_val_scaled, _ = clean_data_no_scaling(X_train, X_val=X_val)

        info = per_day_info[days]
        model = LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            scale_pos_weight=info["scale_pos_weight"],
            class_weight=info["class_weight_dict"],
            boosting_type=boosting_type,
            **info["best_params"]
        )
        model.fit(X_train_scaled, y_train)
        val_proba = model.predict_proba(X_val_scaled)
        classes_order = list(model.classes_)
        if "Acceptable" in classes_order:
            acc_idx = classes_order.index("Acceptable")
            scores = val_proba[:, acc_idx]
        else:
            scores = val_proba[:, 1]
        labels_bin = (pd.Series(y_val) == "Acceptable").astype(int).to_numpy()

        if len(np.unique(labels_bin)) > 1:
            per_day_val_scores[days] = scores
            per_day_val_labels[days] = labels_bin
            t, f = tune_threshold(scores, labels_bin, metric=threshold_metric)
            thresholds_per_day[days] = t
            print(f"  {days}: threshold={t:.3f}, {threshold_metric}={f:.3f}")
        else:
            thresholds_per_day[days] = default_threshold
            print(f"  {days}: one-class VAL, using default threshold={default_threshold:.3f}")

    # For calibration diagnostic, use pooled per-day scores/labels
    all_scores_for_calib = []
    all_labels_for_calib = []
    for d in per_day_val_scores:
        all_scores_for_calib.extend(per_day_val_scores[d].tolist())
        all_labels_for_calib.extend(per_day_val_labels[d].tolist())

    # Calibration diagnostic (based on VAL)
    save_calibration_diagnostic(model_dir, all_scores_for_calib, all_labels_for_calib)

    # ----- PHASE 2: Refit on TRAIN+VAL, evaluate on TEST -----

    for days in unique_days:
        if days not in per_day_info:
            continue

        info = per_day_info[days]
        day_num = info["day_num"]
        day_train = train_df[train_df['DY'] == days].copy()
        day_val = val_df[val_df['DY'] == days].copy()
        day_test = test_df[test_df['DY'] == days].copy()

        if len(day_test) == 0:
            print(f"{days}: no TEST data, skipping.")
            continue

        print(f"\n{'-'*60}")
        print(f"[PHASE 2] Final training + test for {days}")
        print(f"  Train: {len(day_train)}, Val: {len(day_val)}, Test: {len(day_test)}")

        day_combined = pd.concat([day_train, day_val], ignore_index=True)
        X_combined, y_combined, groups_combined = prepare_data_for_day(day_combined, day_num, cols_to_drop_base)
        X_test, y_test, _ = prepare_data_for_day(day_test, day_num, cols_to_drop_base)

        if use_scaling:
            X_combined_scaled, _, X_test_scaled, _ = clean_and_scale_data(X_combined, X_test=X_test)
        else:
            X_combined_scaled, _, X_test_scaled = clean_data_no_scaling(X_combined, X_test=X_test)

        if X_combined_scaled.shape[1] == 0:
            print(f"  No features left after cleaning combined data; skipping {days}.")
            continue

        # Recompute weights on combined
        classes_comb = np.unique(y_combined)
        if len(classes_comb) < 2:
            weight_dict_comb = None
            ratio_comb = 1.0
        else:
            weights_comb = compute_class_weight('balanced', classes=classes_comb, y=y_combined)
            weight_dict_comb = {cls: float(w) for cls, w in zip(classes_comb, weights_comb)}

            pos_label = "Acceptable"
            y_arr_comb = pd.Series(y_combined).to_numpy()
            pos_comb = (y_arr_comb == pos_label).sum()
            neg_comb = (y_arr_comb != pos_label).sum()
            ratio_comb = (neg_comb / pos_comb) if pos_comb > 0 else 1.0

        # weight_mode="both"
        final_class_weight_comb = weight_dict_comb
        final_scale_pos_weight_comb = ratio_comb

        final_model = LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            scale_pos_weight=final_scale_pos_weight_comb,
            class_weight=final_class_weight_comb,
            boosting_type=boosting_type,
            **info["best_params"]
        )
        final_model.fit(X_combined_scaled, y_combined)

        # Extract feature importances
        feature_importance = final_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X_combined_scaled.columns,
            'importance': feature_importance,
            'importance_normalized': feature_importance / feature_importance.sum() if feature_importance.sum() > 0 else 0
        }).sort_values('importance', ascending=False)
        
        # Save feature importance to day directory
        day_dir = model_dir / days
        day_dir.mkdir(parents=True, exist_ok=True)
        
        importance_path = day_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"  Saved feature importance to {importance_path}")
        
        # Visualize top 20 features
        top_n = min(20, len(importance_df))
        plt.figure(figsize=(10, 8))
        top_importance = importance_df.head(top_n)
        
        plt.barh(range(top_n), top_importance['importance'].values, color='steelblue')
        plt.yticks(range(top_n), top_importance['feature'].values)
        plt.xlabel('Importance (split count)', fontweight='bold')
        plt.ylabel('Feature', fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances - {days}', fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        top20_path = day_dir / 'feature_importance_top20.png'
        plt.savefig(top20_path, dpi=150)
        plt.close()
        print(f"  Saved feature importance plot to {top20_path}")
        
        # Store for summary (will be used later)
        info["feature_importance_df"] = importance_df
        info["feature_names"] = list(X_combined_scaled.columns)

        proba_test = final_model.predict_proba(X_test_scaled)
        classes_order = list(final_model.classes_)
        if "Acceptable" in classes_order:
            acc_idx = classes_order.index("Acceptable")
            y_score_test = proba_test[:, acc_idx]
        else:
            y_score_test = proba_test[:, 1]

        threshold_used = thresholds_per_day.get(days, default_threshold)
        y_pred_test = np.where(y_score_test >= threshold_used, "Acceptable", "Not Acceptable")

        # Metrics
        if len(np.unique(y_test)) > 1:
            y_true_bin = (pd.Series(y_test) == "Acceptable").astype(int).to_numpy()
            roc_auc = roc_auc_score(y_true_bin, y_score_test)
        else:
            roc_auc = None

        accuracy = accuracy_score(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)

        precision_accept = report.get('Acceptable', {}).get('precision', 0)
        precision_notaccept = report.get('Not Acceptable', {}).get('precision', 0)
        recall_accept = report.get('Acceptable', {}).get('recall', 0)
        recall_notaccept = report.get('Not Acceptable', {}).get('recall', 0)
        f1_accept = report.get('Acceptable', {}).get('f1-score', 0)
        f1_notaccept = report.get('Not Acceptable', {}).get('f1-score', 0)

        cm = confusion_matrix(y_test, y_pred_test, labels=final_model.classes_)

        if roc_auc is not None:
            print(f"  Test ROC AUC: {roc_auc:.3f}")
        else:
            print("  Test ROC AUC: N/A")
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  Test F1 (Acceptable): {f1_accept:.3f}")
        print(f"  Test F1 (Not Acceptable): {f1_notaccept:.3f}")
        print(f"  Threshold used: {threshold_used:.3f}")

        # Misclassified organoids
        different_rows = day_test[day_test['label'].values != y_pred_test]
        if len(different_rows) > 0:
            print(f"  Misclassified organoids: {list(different_rows['ID'].values)}")

        # Day directory
        day_dir = model_dir / days
        day_dir.mkdir(parents=True, exist_ok=True)

        # Save per-organoid predictions
        save_organoid_predictions(
            day_test.reset_index(drop=True),
            y_test,
            y_pred_test,
            y_score_test,
            day_dir / 'organoid_predictions.csv'
        )

        # Confusion matrix counts
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            classes_cm = final_model.classes_
            # Ensure we know which index is positive
            if "Acceptable" in classes_cm:
                pos_idx = list(classes_cm).index("Acceptable")
                neg_idx = 1 - pos_idx

                # cm[true, pred]
                tp = cm[pos_idx, pos_idx]
                fn = cm[pos_idx, neg_idx]
                fp = cm[neg_idx, pos_idx]
                tn = cm[neg_idx, neg_idx]
            else:
                # Fallback if weird classes
                tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            'day': days,
            'day_no': day_num,
            'test_accuracy': float(accuracy),
            'test_f1_acceptable': float(f1_accept),
            'test_f1_notacceptable': float(f1_notaccept),
            'test_recall_acceptable': float(recall_accept),
            'test_recall_notacceptable': float(recall_notaccept),
            'test_precision_acceptable': float(precision_accept),
            'test_precision_notacceptable': float(precision_notaccept),
            'test_specificity': float(specificity),
            'test_roc_auc': float(roc_auc) if roc_auc is not None else None,
            'best_params': info["best_params"],
            'threshold_used': float(threshold_used),
            'confusion_matrix': {
                'TP': int(tp),
                'FP': int(fp),
                'TN': int(tn),
                'FN': int(fn)
            }
        }

        with open(day_dir / 'metrics_test.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved metrics to {day_dir / 'metrics_test.json'}")

        # Confusion matrix plot
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f"Confusion Matrix - {days}")
        plt.colorbar()
        tick_marks = np.arange(len(final_model.classes_))
        plt.xticks(tick_marks, final_model.classes_, rotation=45)
        plt.yticks(tick_marks, final_model.classes_)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(day_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        print(f"  Saved confusion matrix to {day_dir / 'confusion_matrix.png'}")

        # Flatten results for summary CSV
        row = {
            'Day': days,
            'Day_No': day_num,
            'Test_Accuracy': accuracy,
            'Test_F1_Acceptable': f1_accept,
            'Test_F1_NotAcceptable': f1_notaccept,
            'Test_Recall_Acceptable': recall_accept,
            'Test_Recall_NotAcceptable': recall_notaccept,
            'Test_Precision_Acceptable': precision_accept,
            'Test_Precision_NotAcceptable': precision_notaccept,
            'Test_Specificity': specificity,
            'Test_ROC_AUC': roc_auc if roc_auc is not None else None,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'Threshold_Used': threshold_used,
        }
        # Flatten best_params
        for k, v in info["best_params"].items():
            row[f"BestParam_{k}"] = v
            
        results_summary.append(row)

    if not results_summary:
        print("\nWarning: No results to summarize")
        return

    summary_df = pd.DataFrame(results_summary).sort_values('Day_No')

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(model_dir / 'results_summary.csv', index=False)
    print(f"\nSaved results summary to {model_dir / 'results_summary.csv'}")

    # Metrics-by-day plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Test F1 (Not Acceptable)
    axes[0, 0].plot(summary_df['Day_No'], summary_df['Test_F1_NotAcceptable'], 'o-', color='orange')
    axes[0, 0].set_title('Test F1 Score (Not Acceptable)')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # 2. Test F1 (Acceptable)
    axes[0, 1].plot(summary_df['Day_No'], summary_df['Test_F1_Acceptable'], 'o-', color='blue')
    axes[0, 1].set_title('Test F1 Score (Acceptable)')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # 3. Test Specificity
    axes[1, 0].plot(summary_df['Day_No'], summary_df['Test_Specificity'], 'o-', color='purple')
    axes[1, 0].set_title('Test Specificity (TNR)')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Specificity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # 4. Test ROC-AUC
    auc_data = summary_df.dropna(subset=['Test_ROC_AUC'])
    if len(auc_data) > 0:
        axes[1, 1].plot(auc_data['Day_No'], auc_data['Test_ROC_AUC'], 'o-', color='green')
        axes[1, 1].set_title('Test ROC-AUC')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('ROC-AUC')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(model_dir / 'metrics_by_day.png', dpi=150)
    plt.close()

    print(f"Saved metrics plot to {model_dir / 'metrics_by_day.png'}")
    
    # Create feature importance summary across all days
    print(f"\n{'='*60}")
    print("Creating Feature Importance Summary...")
    print(f"{'='*60}")
    
    # Collect all feature importances across days
    all_features = set()
    for days in unique_days:
        if days in per_day_info and "feature_importance_df" in per_day_info[days]:
            feat_df = per_day_info[days]["feature_importance_df"]
            all_features.update(feat_df['feature'].values)
    
    # Create summary dataframe
    feature_summary_data = []
    for feature in all_features:
        row = {'feature': feature}
        importance_values = []
        
        for days in unique_days:
            if days not in per_day_info or "feature_importance_df" not in per_day_info[days]:
                continue
            
            day_num = per_day_info[days]["day_num"]
            feat_df = per_day_info[days]["feature_importance_df"]
            feature_row = feat_df[feat_df['feature'] == feature]
            
            if len(feature_row) > 0:
                importance = feature_row['importance'].values[0]
                row[f'Day_{day_num}_importance'] = importance
                importance_values.append(importance)
            else:
                row[f'Day_{day_num}_importance'] = 0.0
        
        # Add average and total importance
        if importance_values:
            row['avg_importance'] = np.mean(importance_values)
            row['total_importance'] = np.sum(importance_values)
            row['num_days_used'] = len(importance_values)
        else:
            row['avg_importance'] = 0.0
            row['total_importance'] = 0.0
            row['num_days_used'] = 0
        
        feature_summary_data.append(row)
    
    if feature_summary_data:
        feature_summary_df = pd.DataFrame(feature_summary_data)
        feature_summary_df = feature_summary_df.sort_values('avg_importance', ascending=False)
        
        # Save summary CSV
        feature_summary_path = model_dir / 'feature_importance_summary.csv'
        feature_summary_df.to_csv(feature_summary_path, index=False)
        print(f"Saved feature importance summary to {feature_summary_path}")
        
        # Print top 10 features
        print("\nTop 10 Most Important Features (averaged across all days):")
        top_10 = feature_summary_df.head(10)[['feature', 'avg_importance', 'num_days_used']]
        print(top_10.to_string(index=False))
        
        # Create visualization of top features across days
        top_features = feature_summary_df.head(15)
        day_cols = [c for c in feature_summary_df.columns if c.startswith('Day_') and c.endswith('_importance')]
        
        if len(day_cols) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            x = np.arange(len(top_features))
            width = 0.8 / len(day_cols)
            
            for i, day_col in enumerate(day_cols):
                day_num = day_col.replace('Day_', '').replace('_importance', '')
                offset = (i - len(day_cols)/2) * width + width/2
                ax.bar(x + offset, top_features[day_col].values, width, 
                       label=f'Day {day_num}')
            
            ax.set_xlabel('Feature', fontweight='bold', fontsize=12)
            ax.set_ylabel('Importance', fontweight='bold', fontsize=12)
            ax.set_title('Top 15 Features: Importance Across Days', fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(top_features['feature'].values, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            comparison_plot_path = model_dir / 'feature_importance_comparison.png'
            plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved feature importance comparison plot to {comparison_plot_path}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Results saved to {model_dir}")
    print(f"{'='*60}\n")


def main():
    """Main training function with CLI."""
    parser = argparse.ArgumentParser(description="Train Metabolite Classifiers (Simplified)")
    parser.add_argument("--cv_scoring", choices=["f1_weighted", "f1_notaccept", "macro_f1"],
                        default="f1_weighted",
                        help="Metric to optimize during GridSearch CV.")
    parser.add_argument("--threshold_metric", choices=["f1_weighted", "f1_notaccept", "macro_f1"],
                        default="f1_weighted",
                        help="Metric to optimize during threshold tuning on validation set.")

    args = parser.parse_args()

    train_data_path = 'data_splits/both_train_base.json'
    val_data_path = 'data_splits/both_val_base.json'
    test_data_path = 'data_splits/both_test_base.json'
    output_dir = 'analysis/metabolites/classifier/outputs_metabolites'

    print(f"\n{'='*60}")
    print("Loading data splits...")
    print(f"{'='*60}")

    with open(train_data_path, 'r') as f:
        train_data_json = json.load(f)
    with open(val_data_path, 'r') as f:
        val_data_json = json.load(f)
    with open(test_data_path, 'r') as f:
        test_data_json = json.load(f)

    train_df = json_to_df(train_data_json)
    val_df = json_to_df(val_data_json)
    test_df = json_to_df(test_data_json)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\nComputing growth features...")
    train_df = compute_growth_features(train_df)
    val_df = compute_growth_features(val_df)
    test_df = compute_growth_features(test_df)

    # Construct model name based on args
    model_name = f"lgbm_per_day_noscale_cv_{args.cv_scoring}_thresh_{args.threshold_metric}"

    train_metabolite_classifier_per_day(
        train_df,
        val_df,
        test_df,
        output_dir,
        model_name=model_name,
        cv_scoring=args.cv_scoring,
        threshold_metric=args.threshold_metric
    )


if __name__ == '__main__':
    main()
