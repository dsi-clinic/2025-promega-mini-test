#!/usr/bin/env python3
"""
Metabolite Organoid Quality Classification (CPU Version)
Trains per-day classifiers using LightGBM with GridSearchCV.

This is the CPU version: uses simpler GridSearchCV (not nested) for faster training.
For GPU version with nested CV + RandomizedSearchCV, see train_metabolites_gpu.py.

Usage:
    python train_metabolites_cpu.py                          # Default: f1_notaccept, class_weight, 5-fold
    python train_metabolites_cpu.py --scoring f1_weighted    # Override scoring metric
    python train_metabolites_cpu.py --imbalance smote        # Use SMOTE for class imbalance
    python train_metabolites_cpu.py --n_folds 3              # Use 3-fold CV
    python train_metabolites_cpu.py --use_second_order_growth # Add acceleration features
    python train_metabolites_cpu.py --day_filter Dy03        # Smoke test single day

Flags:
    --scoring:      f1_notaccept (default), f1_weighted, recall_notaccept, macro_f1
    --imbalance:    class_weight (default), scale_pos_weight, smote
    --n_folds:      3 (default), 5
    --day_filter:   Run only on specified day (e.g., Dy03) for smoke testing
    --use_second_order_growth: Add second-order growth features (acceleration)
"""

import json
import re
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score,
    recall_score,
    make_scorer,
)
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

# Optional imports for SMOTE
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

SEED = 42

# Thresholds for warnings
MIN_DAY_SAMPLES = 20
MIN_MINORITY_SAMPLES = 8
MIN_FOLD_MINORITY = 2


class Logger:
    """Minimal logging utility for targeted terminal output (CPU version)."""
    
    def __init__(self, verbose=False, quiet=True, strict=False):
        self.verbose = verbose
        self.quiet = quiet
        self.strict = strict
        self._warnings_issued = set()
    
    def info(self, msg):
        """Print info message."""
        print(msg)
    
    def debug(self, msg):
        """Print debug message (only if verbose)."""
        if self.verbose:
            print(f"  [DEBUG] {msg}")
    
    def warn(self, code, msg, once=True):
        """Print warning with code."""
        key = f"{code}:{msg}" if once else None
        if once and key in self._warnings_issued:
            return
        print(f"  [W{code}] {msg}")
        if once:
            self._warnings_issued.add(key)
    
    def day_line(self, day, n_samples, n_minority, n_folds, cv_score, threshold, 
                 recall_na, fpr, saved=True):
        """Print one-line summary for a day."""
        status = "saved" if saved else "SKIPPED"
        fpr_str = f"{fpr:.2f}" if fpr is not None else "NA"
        print(f"[{day}] n={n_samples} (NA={n_minority}) | folds={n_folds} | "
              f"cv={cv_score:.2f} | thr={threshold:.2f} | recNA={recall_na:.2f} | "
              f"FPR={fpr_str} | {status}")
    
    def startup_block(self, config):
        """Print startup configuration block once (CPU version)."""
        print(f"\n{'=' * 65}")
        print(f"  Mode: CPU / GridSearchCV")
        print(f"  Scoring: {config['scoring']} | Imbalance: {config['imbalance']}")
        print(f"  Folds: {config['n_folds']}")
        print(f"  LightGBM: device=cpu, n_jobs=1 | GridSearchCV: n_jobs=4")
        print(f"  Second-order: {config['use_second_order']}")
        if config.get('day_filter'):
            print(f"  Day filter: {config['day_filter']}")
        print(f"  Output: {config['output_dir']}")
        # Check for CPU oversubscription
        grid_n_jobs = config.get('grid_n_jobs', 4)
        lgbm_n_jobs = config.get('lgbm_n_jobs', 1)
        if grid_n_jobs != 1 and lgbm_n_jobs != 1:
            self.warn("008", f"Oversubscription risk: GridSearchCV n_jobs={grid_n_jobs}, LightGBM n_jobs={lgbm_n_jobs}")
        print(f"{'=' * 65}\n")
    
    def check_minority(self, y, day):
        """Check minority class counts, return whether to skip."""
        minority_count = (y == "Not Acceptable").sum() if hasattr(y, 'sum') else sum(1 for x in y if x == "Not Acceptable")
        n_total = len(y)
        
        skip = False
        if n_total < MIN_DAY_SAMPLES:
            self.warn("001", f"{day}: only {n_total} samples (< {MIN_DAY_SAMPLES})")
            if self.strict:
                skip = True
        
        if minority_count < MIN_MINORITY_SAMPLES:
            self.warn("002", f"{day}: only {minority_count} 'Not Acceptable' samples (< {MIN_MINORITY_SAMPLES})")
            if self.strict:
                skip = True
        
        return skip, minority_count
    
    def check_data_integrity(self, X, day, stage="preprocessing"):
        """Check for NaNs and constant features."""
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            self.warn("005", f"{day}: {nan_count} NaNs after {stage}")
        
        const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if const_cols:
            self.warn("006", f"{day}: {len(const_cols)} constant features")


# Global logger instance
logger = Logger()


def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
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


def compute_growth_features(df, use_second_order=False):
    """Add growth features (difference between consecutive timepoints).
    
    Args:
        df: DataFrame with metabolite concentration columns.
        use_second_order: If True, also compute acceleration (second derivative).
    
    Returns:
        DataFrame with growth feature columns added.
    """
    df = df.copy()
    df["day"] = df["DY"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["ID", "day"])

    # First-order growth (velocity)
    metabolites = [
        ("GlucoseGlo_concentration_uM", "glucose_growth"),
        ("GlutamateGlo_concentration_uM", "glutamate_growth"),
        ("LactateGlo_concentration_uM", "LactateGlo_growth"),
        ("PyruvateGlo_concentration_uM", "PyruvateGlo_growth"),
        ("MalateGlo_concentration_uM", "MalateGlo_growth"),
    ]
    
    for conc_col, growth_col in metabolites:
        if conc_col in df.columns:
            df[growth_col] = df.groupby("ID")[conc_col].diff()

    # Second-order growth (acceleration) - optional
    if use_second_order:
        growth_cols = [gc for _, gc in metabolites if gc in df.columns]
        for gc in growth_cols:
            df[f"{gc}_accel"] = df.groupby("ID")[gc].diff()

    return df


def save_organoid_predictions(selected_test_df, y_test, y_pred, y_score, output_path):
    """Save per-organoid predictions to CSV."""
    label_map = {"Acceptable": 1, "Not Acceptable": 0}

    organoid_results = []
    for idx in range(len(selected_test_df)):
        org_id = selected_test_df.iloc[idx]["ID"]
        true_label_str = selected_test_df.iloc[idx]["label"]
        true_label = label_map.get(true_label_str, 0)
        pred_label_str = y_pred[idx]
        pred_label = label_map.get(pred_label_str, 0)
        pred_prob = float(y_score[idx])
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


def prepare_data_for_day(df, day_num, cols_to_drop_base, use_second_order=False):
    """Prepare features, labels, and groups for a specific day."""
    df_day = df.copy()
    cols_to_drop = cols_to_drop_base.copy()

    # For days <= 10, drop Malate concentration
    if day_num <= 10 and "MalateGlo_concentration_uM" in df_day.columns:
        cols_to_drop.append("MalateGlo_concentration_uM")

    # Drop growth features for day 3 (no previous timepoint)
    growth_features = [
        "glucose_growth", "glutamate_growth", "LactateGlo_growth",
        "PyruvateGlo_growth", "MalateGlo_growth",
    ]
    accel_features = [f"{g}_accel" for g in growth_features]
    
    if day_num == 3:
        cols_to_drop.extend([g for g in growth_features if g in df_day.columns])
        cols_to_drop.extend([a for a in accel_features if a in df_day.columns])
    elif day_num == 6 and use_second_order:
        # Day 6 has first growth but no acceleration
        cols_to_drop.extend([a for a in accel_features if a in df_day.columns])
    elif day_num == 13 and "MalateGlo_growth" in df_day.columns:
        cols_to_drop.append("MalateGlo_growth")
        if use_second_order and "MalateGlo_growth_accel" in df_day.columns:
            cols_to_drop.append("MalateGlo_growth_accel")

    df_day = df_day.drop(columns=[c for c in cols_to_drop if c in df_day.columns])

    X = df_day.drop(columns=["label", "ID"])
    y = df_day["label"]
    groups = df_day["ID"]

    return X, y, groups


def clean_data(X_train, X_test=None):
    """Clean NaNs/constants without scaling (LightGBM doesn't need scaling)."""
    # Drop all-NaN columns
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        X_train = X_train.drop(columns=all_nan_cols)
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in all_nan_cols if c in X_test.columns])

    # Drop constant columns
    constant_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=True) <= 1]
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        if X_test is not None:
            X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

    # Fill NaNs
    if X_train.isna().any().any():
        X_train = X_train.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)

    return X_train, X_test


def clean_and_scale_data(X_train, X_test=None):
    """Clean and scale data (needed for SMOTE)."""
    X_train, X_test = clean_data(X_train, X_test)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
    
    return X_train_scaled, X_test_scaled, scaler


# NOTE: focal_loss removed from CPU script - doesn't work with GridSearchCV scoring


def get_scoring_function(scoring):
    """Get sklearn scorer for a given scoring metric."""
    if scoring == "f1_notaccept":
        return make_scorer(f1_score, pos_label="Not Acceptable")
    elif scoring == "f1_weighted":
        return "f1_weighted"
    elif scoring == "recall_notaccept":
        return make_scorer(recall_score, pos_label="Not Acceptable")
    elif scoring == "macro_f1":
        return "f1_macro"
    else:
        raise ValueError(f"Unknown scoring: {scoring}")


def get_param_grid():
    """Get hyperparameter grid for GridSearchCV (CPU version)."""
    return {
        'max_depth': [3, 6],            # shallow vs deeper
        'num_leaves': [31, 63],         # moderate vs larger tree
        'min_child_samples': [10, 20],  # regularization
        'subsample': [0.8],             # slight row subsampling
        'colsample_bytree': [0.8],      # slight feature subsampling
        'learning_rate': [0.05, 0.1],
        'n_estimators': [200, 500],
    }


def build_model(imbalance, y_train):
    """Build LightGBM model with specified imbalance handling (CPU version).
    
    Args:
        imbalance: One of 'class_weight', 'scale_pos_weight', 'smote'
        y_train: Training labels (for computing weights)
    
    Returns:
        Model or pipeline, and whether it's a pipeline
    """
    device = "cpu"
    n_jobs = 1  # Single thread to avoid oversubscription (GridSearchCV handles parallelism)
    
    classes = np.unique(y_train)
    
    if imbalance == "class_weight":
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}
        model = LGBMClassifier(
            random_state=SEED, verbose=-1, n_jobs=n_jobs, device=device,
            class_weight=class_weight_dict, boosting_type="gbdt"
        )
        return model, False
    
    elif imbalance == "scale_pos_weight":
        # Weight minority class (Not Acceptable) not majority
        minority_label = "Not Acceptable"
        y_arr = np.asarray(y_train)
        pos = (y_arr == minority_label).sum()  # minority count
        neg = (y_arr != minority_label).sum()  # majority count
        ratio = neg / pos if pos > 0 else 1.0
        model = LGBMClassifier(
            random_state=SEED, verbose=-1, n_jobs=n_jobs, device=device,
            scale_pos_weight=ratio, boosting_type="gbdt"
        )
        return model, False
    
    # NOTE: focal_loss removed - use GPU script for focal_loss
    
    elif imbalance == "smote":
        if not SMOTE_AVAILABLE:
            raise ImportError("SMOTE requires imblearn. Install with: pip install imbalanced-learn")
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
            ("model", LGBMClassifier(
                random_state=SEED, verbose=-1, n_jobs=n_jobs, device=device,
                boosting_type="gbdt"
            ))
        ])
        return pipe, True
    
    else:
        raise ValueError(f"Unknown imbalance method: {imbalance}")


def train_metabolite_classifier_per_day(
    train_df, val_df, test_df, output_dir, model_name,
    scoring, imbalance, n_folds, use_second_order, day_filter=None
):
    """Train LightGBM classifier for each day with GridSearchCV (CPU version).
    
    Args:
        train_df, val_df, test_df: DataFrames with training/val/test data
        output_dir: Output directory path
        model_name: Name for this model run
        scoring: Scoring metric for CV
        imbalance: Class imbalance handling method
        n_folds: Number of CV folds
        use_second_order: Whether to use second-order growth features
        day_filter: If specified, run only on this day (e.g., 'Dy03')
    """
    set_seed()

    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    unique_days = sorted(np.unique(train_df.DY))
    
    # Apply day filter if specified
    if day_filter is not None:
        if day_filter in unique_days:
            unique_days = [day_filter]
            print(f"Day filter applied: running only on {day_filter}")
        else:
            print(f"WARNING: day_filter '{day_filter}' not found in data. Available: {unique_days}")

    # Get git hash for reproducibility (optional)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = None

    # Save run config for experiment tracking
    param_grid = get_param_grid()
    run_config = {
        "scoring": scoring,
        "imbalance": imbalance,
        "n_folds": n_folds,
        "use_gpu": False,  # Always CPU in this version
        "use_second_order": use_second_order,
        "seed": SEED,
        "search_method": "GridSearchCV",
        "param_grid": param_grid,
        "git_hash": git_hash,
        "day_filter": day_filter,
    }
    with open(model_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    cols_to_drop_base = [
        "DY", "batch", "img_path", "mask_path",
        "MalateGlo_initial_concentration", "GlucoseGlo_initial_concentration",
        "GlutamateGlo_initial_concentration", "LactateGlo_initial_concentration",
        "PyruvateGlo_initial_concentration", "day",
    ]

    # Process each day
    for days in unique_days:
        day_train = train_df[train_df["DY"] == days].copy()
        day_val = val_df[val_df["DY"] == days].copy()
        day_test = test_df[test_df["DY"] == days].copy()

        if len(day_train) == 0 or len(day_test) == 0:
            logger.warn("010", f"{days}: insufficient data, skipping")
            continue

        day_num = int(re.search(r"\d+", days).group())
        
        # Combine train + val for training
        day_trainval = pd.concat([day_train, day_val], ignore_index=True)
        
        # Check minority counts
        skip_day, minority_count = logger.check_minority(day_trainval["label"], days)
        if skip_day:
            continue
        
        logger.debug(f"{days}: n={len(day_trainval)} (NA={minority_count})")
        
        X_trainval, y_trainval, groups_trainval = prepare_data_for_day(
            day_trainval, day_num, cols_to_drop_base, use_second_order
        )
        X_test, y_test, _ = prepare_data_for_day(
            day_test, day_num, cols_to_drop_base, use_second_order
        )

        # Clean data
        if imbalance == "smote":
            X_trainval_clean, X_test_clean, _ = clean_and_scale_data(X_trainval, X_test)
        else:
            X_trainval_clean, X_test_clean = clean_data(X_trainval, X_test)

        if X_trainval_clean.shape[1] == 0:
            print(f"  No features left after cleaning; skipping {days}.")
            continue

        # Build model
        model, is_pipeline = build_model(imbalance, y_trainval)
        
        # Get param grid
        param_grid = get_param_grid()
        if is_pipeline:
            param_grid = {f"model__{k}": v for k, v in param_grid.items()}
            param_grid["smote__k_neighbors"] = [3, 5]

        # GridSearchCV (simpler than nested CV in GPU version)
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        scoring_fn = get_scoring_function(scoring)
        
        logger.debug(f"{days}: running GridSearchCV...")
        
        grid = GridSearchCV(
            model, param_grid,
            scoring=scoring_fn, cv=cv, n_jobs=4,
            verbose=0  # Reduced from 1 to minimize output
        )
        
        try:
            # Suppress warnings for focal_loss (custom objective breaks sklearn scorers)
            if is_pipeline:
                grid.fit(X_trainval_clean, y_trainval)
            else:
                grid.fit(X_trainval_clean, y_trainval, groups=groups_trainval)
        except Exception as e:
            logger.warn("009", f"{days}: GridSearchCV failed ({e}), using defaults")
            model.fit(X_trainval_clean, y_trainval)
            grid = type('obj', (object,), {'best_estimator_': model, 'best_score_': 0, 'best_params_': {}})()

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        logger.debug(f"{days}: CV score={best_cv_score:.3f}")

        # Predict on test set
        if hasattr(best_model, 'predict_proba'):
            test_proba = best_model.predict_proba(X_test_clean)
            actual_model = best_model.named_steps["model"] if is_pipeline else best_model
            classes_order = list(actual_model.classes_)
            if "Acceptable" in classes_order:
                acc_idx = classes_order.index("Acceptable")
                y_score_test = test_proba[:, acc_idx]
            else:
                y_score_test = test_proba[:, 1]
        else:
            y_score_test = np.ones(len(y_test)) * 0.5

        # Apply default threshold (0.5)
        y_pred_test = np.where(y_score_test >= 0.5, "Acceptable", "Not Acceptable")

        # Calculate metrics
        y_true_bin = (y_test == "Acceptable").astype(int).to_numpy()
        
        try:
            pr_auc = average_precision_score(y_true_bin, y_score_test)
        except ValueError:
            pr_auc = None

        accuracy = accuracy_score(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)

        f1_accept = report.get("Acceptable", {}).get("f1-score", 0)
        f1_notaccept = report.get("Not Acceptable", {}).get("f1-score", 0)
        recall_accept = report.get("Acceptable", {}).get("recall", 0)
        recall_notaccept = report.get("Not Acceptable", {}).get("recall", 0)
        precision_accept = report.get("Acceptable", {}).get("precision", 0)
        precision_notaccept = report.get("Not Acceptable", {}).get("precision", 0)

        actual_model = best_model.named_steps["model"] if is_pipeline else best_model
        cm = confusion_matrix(y_test, y_pred_test, labels=actual_model.classes_)

        # Extract confusion matrix values
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            classes_cm = actual_model.classes_
            if "Acceptable" in classes_cm:
                pos_idx = list(classes_cm).index("Acceptable")
                neg_idx = 1 - pos_idx
                tp = cm[pos_idx, pos_idx]
                fn = cm[pos_idx, neg_idx]
                fp = cm[neg_idx, pos_idx]
                tn = cm[neg_idx, neg_idx]
            else:
                tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        
        # One-line day summary
        logger.day_line(
            day=days,
            n_samples=len(day_trainval),
            n_minority=minority_count,
            n_folds=n_folds,
            cv_score=best_cv_score,
            threshold=0.5,  # CPU uses default threshold
            recall_na=recall_notaccept,
            fpr=fpr,
            saved=True
        )

        # Save day results
        day_dir = model_dir / days
        day_dir.mkdir(parents=True, exist_ok=True)

        save_organoid_predictions(
            day_test.reset_index(drop=True), y_test, y_pred_test, y_score_test,
            day_dir / "organoid_predictions.csv"
        )

        # Save feature importance
        if not is_pipeline:
            feature_importance = best_model.feature_importances_
            importance_df = pd.DataFrame({
                "feature": X_trainval_clean.columns,
                "importance": feature_importance,
            }).sort_values("importance", ascending=False)
            importance_df.to_csv(day_dir / "feature_importance.csv", index=False)

        # Save metrics
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
            "test_pr_auc": float(pr_auc) if pr_auc is not None else None,
            "best_params": best_params,
            "cv_score": float(best_cv_score),
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        }

        with open(day_dir / "metrics_test.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save confusion matrix plot
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix - {days}")
        plt.colorbar()
        tick_marks = np.arange(len(actual_model.classes_))
        plt.xticks(tick_marks, actual_model.classes_, rotation=45)
        plt.yticks(tick_marks, actual_model.classes_)
        thresh_cm = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                         color="white" if cm[i, j] > thresh_cm else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(day_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # Add to summary
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
            "Test_PR_AUC": pr_auc,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "CV_Score": best_cv_score,
        })

    if not results_summary:
        logger.warn("012", "No results to summarize")
        return

    # Save summary
    summary_df = pd.DataFrame(results_summary).sort_values("Day_No")
    summary_df.to_csv(model_dir / "results_summary.csv", index=False)

    # Create metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(summary_df["Day_No"], summary_df["Test_Accuracy"], "o-", color="blue")
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    axes[0, 1].plot(summary_df["Day_No"], summary_df["Test_F1_Acceptable"], "o-", color="green", label="Acceptable")
    axes[0, 1].plot(summary_df["Day_No"], summary_df["Test_F1_NotAcceptable"], "o--", color="red", label="Not Acceptable")
    axes[0, 1].set_title("Test F1 Score")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    axes[1, 0].plot(summary_df["Day_No"], summary_df["Test_Specificity"], "o-", color="purple")
    axes[1, 0].set_title("Test Specificity (TNR)")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Specificity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    pr_auc_data = summary_df.dropna(subset=["Test_PR_AUC"])
    if len(pr_auc_data) > 0:
        axes[1, 1].plot(pr_auc_data["Day_No"], pr_auc_data["Test_PR_AUC"], "o-", color="orange")
        axes[1, 1].set_title("Test PR-AUC")
        axes[1, 1].set_xlabel("Day")
        axes[1, 1].set_ylabel("PR-AUC")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(model_dir / "metrics_by_day.png", dpi=150)
    plt.close()

    # Single completion line
    print(f"\nDone. Results: {model_dir}")


def main():
    """Main training function with CLI."""
    parser = argparse.ArgumentParser(
        description="Train Metabolite Classifiers with GridSearchCV (CPU version)"
    )
    parser.add_argument(
        "--scoring",
        choices=["f1_notaccept", "f1_weighted", "recall_notaccept", "macro_f1"],
        default="f1_notaccept",
        help="Scoring metric for CV (default: f1_notaccept)",
    )
    parser.add_argument(
        "--imbalance",
        choices=["class_weight", "scale_pos_weight", "smote"],
        default="class_weight",
        help="Class imbalance handling method (default: class_weight)",
    )
    parser.add_argument(
        "--n_folds",
        type=int, choices=[3, 5], default=3,
        help="Number of CV folds (default: 3)",
    )
    parser.add_argument(
        "--use_second_order_growth",
        action="store_true",
        help="Add second-order growth features (acceleration)",
    )
    parser.add_argument(
        "--day_filter",
        type=str, default=None,
        help="Run only on specified day (e.g., 'Dy03') for smoke testing",
    )
    # Verbosity/strictness flags
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-fold details and debug info",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=True,
        help="Minimal output: startup block + one line per day + warnings (default)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Skip days with insufficient samples, abort on errors",
    )

    args = parser.parse_args()

    # Configure global logger
    global logger
    logger = Logger(verbose=args.verbose, quiet=args.quiet, strict=args.strict)

    train_data_path = "data_splits/both_train_base.json"
    val_data_path = "data_splits/both_val_base.json"
    test_data_path = "data_splits/both_test_base.json"
    output_dir = "analysis/metabolites/classifier/outputs_metabolites"

    # Startup config block
    logger.startup_block({
        'scoring': args.scoring,
        'imbalance': args.imbalance,
        'n_folds': args.n_folds,
        'use_second_order': args.use_second_order_growth,
        'day_filter': args.day_filter,
        'output_dir': output_dir,
        'grid_n_jobs': 4,
        'lgbm_n_jobs': 1,
    })

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

    print("\nComputing growth features...")
    train_df = compute_growth_features(train_df, use_second_order=args.use_second_order_growth)
    val_df = compute_growth_features(val_df, use_second_order=args.use_second_order_growth)
    test_df = compute_growth_features(test_df, use_second_order=args.use_second_order_growth)

    # Construct model name (always CPU)
    model_name = f"lgbm_{args.scoring}_{args.imbalance}_cpu"
    if args.use_second_order_growth:
        model_name += "_accel"

    train_metabolite_classifier_per_day(
        train_df, val_df, test_df, output_dir, model_name,
        scoring=args.scoring,
        imbalance=args.imbalance,
        n_folds=args.n_folds,
        use_second_order=args.use_second_order_growth,
        day_filter=args.day_filter,
    )


if __name__ == "__main__":
    main()
