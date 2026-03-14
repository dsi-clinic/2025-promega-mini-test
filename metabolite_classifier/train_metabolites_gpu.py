#!/usr/bin/env python3
"""
Metabolite Organoid Quality Classification (GPU Version)
Trains per-day classifiers using LightGBM with nested CV + RandomizedSearchCV.

This is the GPU version: uses device="gpu" and deep hyperparameter search (120 configs).
For CPU version with simpler GridSearchCV, see train_metabolites_cpu.py.

Usage:
    python train_metabolites_gpu.py                          # Default: f1_notaccept, class_weight, 5-fold
    python train_metabolites_gpu.py --scoring f1_weighted    # Override scoring metric
    python train_metabolites_gpu.py --imbalance smote        # Use SMOTE for class imbalance
    python train_metabolites_gpu.py --n_folds 3              # Use 3-fold CV
    python train_metabolites_gpu.py --use_second_order_growth # Add acceleration features
    python train_metabolites_gpu.py --day_filter Dy03        # Smoke test single day

Flags:
    --scoring:      f1_notaccept (default), f1_weighted, recall_notaccept, macro_f1
    --imbalance:    class_weight (default), scale_pos_weight, focal_loss, smote
    --n_folds:      3 (default), 5 - outer CV folds (inner always 3)
    --n_iter:       Override RandomizedSearchCV n_iter (default: 30)
    --search_n_jobs: n_jobs for RandomizedSearchCV (default: 1 for stability)
    --day_filter:   Run only on specified day (e.g., Dy03) for smoke testing
    --use_second_order_growth: Add second-order growth features (acceleration)
"""

import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
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
INNER_FOLDS = 3  # Always fixed at 3 for inner CV
SEARCH_ITERATIONS = 30  # Default, use --n_iter to increase

# Thresholds for warnings
MIN_DAY_SAMPLES = 20
MIN_MINORITY_SAMPLES = 8
MIN_FOLD_MINORITY = 2


class Logger:
    """Minimal logging utility for targeted terminal output."""

    def __init__(self, verbose=False, quiet=True, strict=False):
        self.verbose = verbose
        self.quiet = quiet
        self.strict = strict
        self._warnings_issued = set()

    def info(self, msg):
        """Print info message (always shown unless quiet and not startup)."""
        print(msg)

    def debug(self, msg):
        """Print debug message (only if verbose)."""
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def warn(self, code, msg, once=True):
        """Print warning with code (only if not already issued or once=False)."""
        key = f"{code}:{msg}" if once else None
        if once and key in self._warnings_issued:
            return
        print(f"  [W{code}] {msg}")
        if once:
            self._warnings_issued.add(key)

    def day_line(
        self,
        day,
        n_samples,
        n_minority,
        n_folds,
        cv_score,
        threshold,
        recall_na,
        fpr,
        saved=True,
    ):
        """Print one-line summary for a day."""
        status = "saved" if saved else "SKIPPED"
        fpr_str = f"{fpr:.2f}" if fpr is not None else "NA"
        print(
            f"[{day}] n={n_samples} (NA={n_minority}) | folds={n_folds} | "
            f"cv={cv_score:.2f} | thr={threshold:.2f} | recNA={recall_na:.2f} | "
            f"FPR={fpr_str} | {status}"
        )

    def startup_block(self, config):
        """Print startup configuration block once."""
        print(f"\n{'=' * 65}")
        print("  Mode: GPU / RandomizedSearchCV (nested CV)")
        print(f"  Scoring: {config['scoring']} | Imbalance: {config['imbalance']}")
        print(f"  Outer folds: {config['n_folds']} | Inner folds: {INNER_FOLDS}")
        print(
            f"  Search iters: {config['n_iter']} | search_n_jobs: {config['search_n_jobs']}"
        )
        print("  LightGBM: device=gpu, n_jobs=1")
        print(f"  Second-order: {config['use_second_order']}")
        if config.get("day_filter"):
            print(f"  Day filter: {config['day_filter']}")
        print(f"  Output: {config['output_dir']}")
        est_fits = config["n_folds"] * config["n_iter"] * INNER_FOLDS
        print(f"  Est. fits/day: ~{est_fits}")
        print(f"{'=' * 65}\n")

    def check_data_integrity(self, X, day, stage="preprocessing"):
        """Check for NaNs and constant features, issue warnings."""
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            self.warn("005", f"{day}: {nan_count} NaNs after {stage}")

        const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if const_cols:
            self.warn("006", f"{day}: {len(const_cols)} constant features")

    def check_minority(self, y, day, folds=None):
        """Check minority class counts, return whether to skip."""
        minority_count = (
            (y == "Not Acceptable").sum()
            if hasattr(y, "sum")
            else sum(1 for x in y if x == "Not Acceptable")
        )
        n_total = len(y)

        skip = False
        if n_total < MIN_DAY_SAMPLES:
            self.warn("001", f"{day}: only {n_total} samples (< {MIN_DAY_SAMPLES})")
            if self.strict:
                skip = True

        if minority_count < MIN_MINORITY_SAMPLES:
            self.warn(
                "002",
                f"{day}: only {minority_count} 'Not Acceptable' samples (< {MIN_MINORITY_SAMPLES})",
            )
            if self.strict:
                skip = True

        return skip, minority_count

    def check_fold_minority(self, y, groups, n_folds, day):
        """Check per-fold minority counts."""
        from sklearn.model_selection import StratifiedGroupKFold

        try:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
            min_fold_minority = float("inf")
            for _, va_idx in cv.split(np.zeros(len(y)), y, groups=groups):
                fold_minority = (y.iloc[va_idx] == "Not Acceptable").sum()
                min_fold_minority = min(min_fold_minority, fold_minority)

            if min_fold_minority < MIN_FOLD_MINORITY:
                self.warn(
                    "003",
                    f"{day}: fold has only {min_fold_minority} minority samples (< {MIN_FOLD_MINORITY})",
                )
                return self.strict  # skip if strict
        except Exception as e:
            logger.warn("004", f"Day {day}: fold check failed: {e}")
        return False

    def check_smote_safety(self, minority_count, k_neighbors, day):
        """Check if SMOTE is safe to use."""
        if minority_count < k_neighbors + 1:
            self.warn(
                "007",
                f"{day}: minority ({minority_count}) < k_neighbors+1 ({k_neighbors + 1}) for SMOTE",
            )
            return False
        return True


# Global logger instance (will be configured in main)
logger = Logger()


def check_gpu_available():
    """Fail fast if GPU is not available for LightGBM."""
    try:
        import lightgbm as lgb

        # Create a minimal model with GPU params
        test_model = lgb.LGBMClassifier(
            device="gpu", n_estimators=1, num_leaves=2, verbose=-1, n_jobs=1
        )
        # Try to fit on minimal data to verify GPU works
        import numpy as np

        X_test = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        y_test = np.array([0, 1, 1, 0])
        test_model.fit(X_test, y_test)
        print("GPU availability check: PASSED")
    except Exception as e:
        raise RuntimeError(
            f"GPU not available for LightGBM. Ensure CUDA/OpenCL is configured.\n"
            f"Error: {e}\n"
            f"For CPU training, use train_metabolites_cpu.py instead."
        )


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


def prepare_data_for_day(df, day_num, cols_to_drop_base, use_second_order=False):
    """Prepare features, labels, and groups for a specific day."""
    df_day = df.copy()
    cols_to_drop = cols_to_drop_base.copy()

    # For days <= 10, drop Malate concentration
    if day_num <= 10 and "MalateGlo_concentration_uM" in df_day.columns:
        cols_to_drop.append("MalateGlo_concentration_uM")

    # Drop growth features for day 3 (no previous timepoint)
    growth_features = [
        "glucose_growth",
        "glutamate_growth",
        "LactateGlo_growth",
        "PyruvateGlo_growth",
        "MalateGlo_growth",
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


def clean_data(X_train, X_val=None, X_test=None):
    """Clean NaNs/constants without scaling (LightGBM doesn't need scaling)."""
    # Drop all-NaN columns
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
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
        X_train = X_train.drop(columns=constant_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(
                columns=[c for c in constant_cols if c in X_test.columns]
            )

    # Fill NaNs
    if X_train.isna().any().any():
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)

    return X_train, X_val, X_test


def clean_and_scale_data(X_train, X_val=None, X_test=None):
    """Clean and scale data (needed for SMOTE)."""
    X_train, X_val, X_test = clean_data(X_train, X_val, X_test)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


class FocalLossLGBMClassifier(LGBMClassifier):
    """
    Wrapper for LGBMClassifier with Focal Loss that behaves like a standard
    sklearn classifier (predict_proba returns [1-p, p] from sigmoid).
    """

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        self.set_params(objective=focal_loss_lgbm(gamma=self.gamma, alpha=self.alpha))
        super().fit(X, y, **kwargs)
        return self

    def predict(self, X, raw_score=False, **kwargs):
        # Return 0/1 predictions based on probability > 0.5
        probas = self.predict_proba(X, **kwargs)
        # Select class 1 probability
        p = probas[:, 1]
        indices = (p >= 0.5).astype(int)

        # Decode to original labels if available
        if hasattr(self, "classes_"):
            return self.classes_[indices]
        return indices

    def predict_proba(self, X, **kwargs):
        # Use the booster directly to avoid recursion
        # LightGBM's sklearn wrapper can loop infinitely if we call super().predict(raw_score=True)
        # because predict() logic calls predict_proba() for custom objectives.
        if self.booster_ is None:
            raise RuntimeError("Estimator not fitted, call fit first")

        # Get raw scores (logits) from the booster
        raw_scores = self.booster_.predict(X, raw_score=True, **kwargs)

        # Sigmoid transform to get probability p
        p = 1.0 / (1.0 + np.exp(-raw_scores))

        # Return [1-p, p] shape (n_samples, 2)
        return np.vstack([1.0 - p, p]).T


def focal_loss_lgbm(gamma=2.0, alpha=0.25):
    """Create focal loss objective for LightGBM."""

    def focal_loss_objective(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        grad = (
            p
            - y_true
            + alpha
            * gamma
            * (1 - p) ** gamma
            * p
            * (gamma * p * np.log(np.maximum(p, 1e-15)) / (1 - p + 1e-15) - 1)
        )
        hess = (
            p * (1 - p) * (1 + alpha * gamma * (1 - p) ** (gamma - 1) * (gamma * p - 1))
        )
        return grad, np.maximum(np.abs(hess), 1e-8)

    return focal_loss_objective


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


def tune_threshold(y_true_bin, y_prob, scoring):
    """Tune classification threshold on binary labels.

    Args:
        y_true_bin: Binary labels (1=Acceptable, 0=Not Acceptable)
        y_prob: Predicted probabilities for Acceptable class
        scoring: Scoring metric name

    Returns:
        Tuple of (best_threshold, best_score)
    """
    # Safety check: Ensure y_prob is a 1D array
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 0:
        # Scalar case (single sample or error), return default
        return 0.5, 0.0
    if y_prob.ndim > 1:
        y_prob = y_prob.ravel()

    thresholds = np.linspace(0.1, 0.9, 17)
    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred_bin = (y_prob >= t).astype(int)

        try:
            if scoring == "f1_notaccept":
                score = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
            elif scoring == "recall_notaccept":
                score = recall_score(
                    y_true_bin, y_pred_bin, pos_label=0, zero_division=0
                )
            elif scoring == "f1_weighted":
                score = f1_score(
                    y_true_bin, y_pred_bin, average="weighted", zero_division=0
                )
            elif scoring == "macro_f1":
                score = f1_score(
                    y_true_bin, y_pred_bin, average="macro", zero_division=0
                )
            else:
                score = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        except Exception as e:
            logger.warn("011", f"Threshold search f1_score failed: {e}, using 0.0")
            score = 0.0

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


def get_param_distributions():
    """Get hyperparameter distributions for RandomizedSearchCV.

    Tightened for small-N, day-wise training with minority class often single-digit per fold.
    Avoids extremes (num_leaves=127, max_depth=-1, n_estimators=1000) that cause overfitting.
    """
    return {
        "num_leaves": [7, 15, 31, 63],
        "max_depth": [3, 5, 7],
        "min_child_samples": [10, 20, 40, 60],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300, 500],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0],
        "min_split_gain": [0, 0.01, 0.05, 0.1],
    }


def build_model(imbalance, y_train):
    """Build LightGBM model with specified imbalance handling (GPU version).

    Args:
        imbalance: One of 'class_weight', 'scale_pos_weight', 'focal_loss', 'smote'
        y_train: Training labels (for computing weights)

    Returns:
        Model or pipeline, and whether it's a pipeline
    """
    device = "gpu"  # Always GPU in this version
    n_jobs = 1  # Single thread (GPU handles parallelism)

    classes = np.unique(y_train)

    if imbalance == "class_weight":
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}
        model = LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=n_jobs,
            device=device,
            class_weight=class_weight_dict,
            boosting_type="gbdt",
        )
        return model, False

    elif imbalance == "scale_pos_weight":
        # Fix: Weight minority class (Not Acceptable) not majority
        # scale_pos_weight weights the positive class; we want to upweight the rare class
        minority_label = "Not Acceptable"
        y_arr = np.asarray(y_train)
        pos = (y_arr == minority_label).sum()  # minority count
        neg = (y_arr != minority_label).sum()  # majority count
        ratio = neg / pos if pos > 0 else 1.0
        model = LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=n_jobs,
            device=device,
            scale_pos_weight=ratio,
            boosting_type="gbdt",
        )
        return model, False

    elif imbalance == "focal_loss":
        # Use wrapper class for scikit-learn compatibility
        model = FocalLossLGBMClassifier(
            alpha=0.25,
            gamma=2.0,
            random_state=SEED,
            verbose=-1,
            n_jobs=n_jobs,
            device=device,
            boosting_type="gbdt",
        )
        return model, False

    elif imbalance == "smote":
        if not SMOTE_AVAILABLE:
            raise ImportError(
                "SMOTE requires imblearn. Install with: pip install imbalanced-learn"
            )
        pipe = ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
                (
                    "model",
                    LGBMClassifier(
                        random_state=SEED,
                        verbose=-1,
                        n_jobs=n_jobs,
                        device=device,
                        boosting_type="gbdt",
                    ),
                ),
            ]
        )
        return pipe, True

    else:
        raise ValueError(f"Unknown imbalance method: {imbalance}")


def train_metabolite_classifier_per_day(
    train_df,
    val_df,
    test_df,
    output_dir,
    model_name,
    scoring,
    imbalance,
    n_folds,
    use_second_order,
    n_iter=None,
    search_n_jobs=1,
    day_filter=None,
):
    """Train LightGBM classifier for each day with proper nested CV (GPU version).

    Args:
        train_df, val_df, test_df: DataFrames with training/val/test data
        output_dir: Output directory path
        model_name: Name for this model run
        scoring: Scoring metric for CV and threshold tuning
        imbalance: Class imbalance handling method
        n_folds: Number of CV folds
        use_second_order: Whether to use second-order growth features
        n_iter: Override for RandomizedSearchCV n_iter (default: 120)
        search_n_jobs: n_jobs for RandomizedSearchCV (default: 1 for stability)
        day_filter: If specified, run only on this day (e.g., 'Dy03')
    """
    set_seed()

    # Determine search iterations (GPU default: 120)
    if n_iter is not None:
        search_iterations = n_iter
    else:
        search_iterations = SEARCH_ITERATIONS

    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    unique_days = sorted(np.unique(train_df.DY))

    # Apply day filter if specified (I)
    if day_filter is not None:
        if day_filter in unique_days:
            unique_days = [day_filter]
            print(f"Day filter applied: running only on {day_filter}")
        else:
            print(
                f"WARNING: day_filter '{day_filter}' not found in data. Available: {unique_days}"
            )

    # Git hash omitted per DSI clinic standards (subprocess banned)
    git_hash = None

    # Save run config for experiment tracking (G)
    param_dist = get_param_distributions()
    run_config = {
        "scoring": scoring,
        "imbalance": imbalance,
        "n_folds": n_folds,
        "use_gpu": True,  # Always GPU in this version
        "use_second_order": use_second_order,
        "seed": SEED,
        "search_iterations": search_iterations,
        "search_n_jobs": search_n_jobs,
        "inner_cv_splits": 3,
        "param_distributions": param_dist,
        "git_hash": git_hash,
        "day_filter": day_filter,
    }
    with open(model_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    logger.debug(f"Saved run config to {model_dir / 'run_config.json'}")

    cols_to_drop_base = [
        "DY",
        "batch",
        "img_path",
        "mask_path",
        "MalateGlo_initial_concentration",
        "GlucoseGlo_initial_concentration",
        "GlutamateGlo_initial_concentration",
        "LactateGlo_initial_concentration",
        "PyruvateGlo_initial_concentration",
        "day",
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

        # Combine train + val for CV
        day_trainval = pd.concat([day_train, day_val], ignore_index=True)

        # Check minority counts (may skip if strict mode)
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
            X_trainval_clean, _, X_test_clean, _ = clean_and_scale_data(
                X_trainval, X_test=X_test
            )
            # Check SMOTE safety
            logger.check_smote_safety(minority_count, 5, days)
        else:
            X_trainval_clean, _, X_test_clean = clean_data(X_trainval, X_test=X_test)

        if X_trainval_clean.shape[1] == 0:
            logger.warn("006", f"{days}: no features left after cleaning, skipping")
            continue

        # Check data integrity
        logger.check_data_integrity(X_trainval_clean, days)

        # ===== PHASE 1: Cross-Validation with Threshold Tuning =====
        logger.debug(f"{days}: starting {n_folds}-fold CV...")

        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        fold_thresholds = []
        fold_scores = []
        fold_search_best_scores = []  # Track inner CV best scores for logging (E)
        best_params = None
        best_cv_score = -1.0

        for fold_idx, (tr_idx, va_idx) in enumerate(
            cv.split(X_trainval_clean, y_trainval, groups=groups_trainval)
        ):
            X_tr = X_trainval_clean.iloc[tr_idx]
            X_va = X_trainval_clean.iloc[va_idx]
            y_tr = y_trainval.iloc[tr_idx]
            y_va = y_trainval.iloc[va_idx]
            groups_tr = groups_trainval.iloc[tr_idx]

            # Build model
            model, is_pipeline = build_model(imbalance, y_tr)

            # Get param distributions
            param_dist = get_param_distributions()
            if is_pipeline:
                param_dist = {f"model__{k}": v for k, v in param_dist.items()}
                param_dist["smote__k_neighbors"] = [3, 5, 7]

            # Inner CV for hyperparameter search
            inner_cv = StratifiedGroupKFold(
                n_splits=INNER_FOLDS, shuffle=True, random_state=SEED
            )
            scoring_fn = get_scoring_function(scoring)

            # RandomizedSearchCV for hyperparameter search
            search = RandomizedSearchCV(
                model,
                param_dist,
                n_iter=search_iterations,
                scoring=scoring_fn,
                cv=inner_cv,
                n_jobs=search_n_jobs,
                random_state=SEED,
                verbose=0,
            )

            try:
                if is_pipeline:
                    search.fit(X_tr, y_tr)
                else:
                    search.fit(X_tr, y_tr, groups=groups_tr)
            except Exception as e:
                print(f"    Fold {fold_idx}: Search failed ({e}), using defaults")
                model.fit(X_tr, y_tr)
                search = type(
                    "obj",
                    (object,),
                    {"best_estimator_": model, "best_score_": 0, "best_params_": {}},
                )()

            best_model = search.best_estimator_
            fold_search_best_scores.append(float(search.best_score_))

            # ====== FIX THRESHOLD LEAKAGE ======
            # Tune threshold on INNER-CV OOF predictions (training fold only)
            # Not on outer holdout - that would be leakage!
            try:
                # Get OOF predictions on training fold using inner CV
                oof_proba = cross_val_predict(
                    best_model,
                    X_tr,
                    y_tr,
                    cv=inner_cv,
                    method="predict_proba",
                    n_jobs=search_n_jobs,
                )
                # Get index for "Acceptable" class
                actual_model_inner = (
                    best_model.named_steps["model"] if is_pipeline else best_model
                )
                classes_order_inner = list(actual_model_inner.classes_)
                if "Acceptable" in classes_order_inner:
                    acc_idx_inner = classes_order_inner.index("Acceptable")
                    oof_prob_acceptable = oof_proba[:, acc_idx_inner]
                else:
                    oof_prob_acceptable = oof_proba[:, 1]

                # Tune threshold on inner OOF predictions (no leakage)
                y_tr_bin = (y_tr == "Acceptable").astype(int).to_numpy()

                # Check shapes match
                if len(oof_prob_acceptable) != len(y_tr_bin):
                    # Fallback if cross_val_predict returns different size (rare but possible w/ groups)
                    threshold = 0.5
                else:
                    threshold, _ = tune_threshold(
                        y_tr_bin, oof_prob_acceptable, scoring
                    )
            except Exception:
                # print(f"    Fold {fold_idx}: OOF threshold tuning failed ({e}), using 0.5")  # Reduce clutter
                threshold = 0.5

            fold_thresholds.append(threshold)

            # ====== EVALUATE ON OUTER HOLDOUT (with fixed threshold) ======
            # Apply the threshold to outer holdout WITHOUT re-tuning
            if hasattr(best_model, "predict_proba"):
                va_proba = best_model.predict_proba(X_va)
                classes_order = (
                    list(best_model.classes_)
                    if hasattr(best_model, "classes_")
                    else (
                        list(best_model.named_steps["model"].classes_)
                        if is_pipeline
                        else ["Not Acceptable", "Acceptable"]
                    )
                )
                if "Acceptable" in classes_order:
                    acc_idx = classes_order.index("Acceptable")
                    y_prob_va = va_proba[:, acc_idx]
                else:
                    y_prob_va = va_proba[:, 1]
            else:
                y_prob_va = np.ones(len(y_va)) * 0.5

            # Score using the threshold tuned on inner OOF (no leakage)
            y_va_bin = (y_va == "Acceptable").astype(int).to_numpy()
            y_pred_va = (y_prob_va >= threshold).astype(int)

            if scoring == "f1_notaccept":
                fold_score = f1_score(y_va_bin, y_pred_va, pos_label=0, zero_division=0)
            elif scoring == "recall_notaccept":
                fold_score = recall_score(
                    y_va_bin, y_pred_va, pos_label=0, zero_division=0
                )
            elif scoring == "f1_weighted":
                fold_score = f1_score(
                    y_va_bin, y_pred_va, average="weighted", zero_division=0
                )
            else:  # macro_f1
                fold_score = f1_score(
                    y_va_bin, y_pred_va, average="macro", zero_division=0
                )

            fold_scores.append(fold_score)

            if search.best_score_ > best_cv_score:
                best_cv_score = search.best_score_
                if is_pipeline:
                    best_params = {
                        k.replace("model__", ""): v
                        for k, v in search.best_params_.items()
                        if k.startswith("model__")
                    }
                else:
                    best_params = search.best_params_

            logger.debug(
                f"{days} Fold {fold_idx}: thr={threshold:.3f}, score={fold_score:.3f}"
            )

        # Use MEDIAN threshold (fixed for final model)
        final_threshold = float(np.median(fold_thresholds))
        mean_cv_score = float(np.mean(fold_scores))

        # Stability diagnostics
        std_threshold = float(np.std(fold_thresholds))
        std_score = float(np.std(fold_scores))
        logger.debug(
            f"{days}: CV mean={mean_cv_score:.3f}, median_thr={final_threshold:.3f}"
        )
        if std_threshold > 0.15:
            logger.warn(
                "011", f"{days}: threshold unstable (std={std_threshold:.3f} > 0.15)"
            )

        # ===== PHASE 2: Final Retrain on TRAIN+VAL =====
        logger.debug(f"{days}: retraining on full train+val...")

        final_model, is_pipeline = build_model(imbalance, y_trainval)

        if best_params:
            if is_pipeline:
                for k, v in best_params.items():
                    final_model.named_steps["model"].set_params(**{k: v})
            else:
                final_model.set_params(**best_params)

        final_model.fit(X_trainval_clean, y_trainval)

        # ===== PHASE 3: Evaluate on TEST =====
        logger.debug(f"{days}: evaluating on test set...")

        if hasattr(final_model, "predict_proba"):
            test_proba = final_model.predict_proba(X_test_clean)
            actual_model = (
                final_model.named_steps["model"] if is_pipeline else final_model
            )
            classes_order = list(actual_model.classes_)
            if "Acceptable" in classes_order:
                acc_idx = classes_order.index("Acceptable")
                y_score_test = test_proba[:, acc_idx]
            else:
                y_score_test = test_proba[:, 1]
        else:
            y_score_test = np.ones(len(y_test)) * 0.5

        # Apply fixed threshold
        y_pred_test = np.where(
            y_score_test >= final_threshold, "Acceptable", "Not Acceptable"
        )

        # Calculate metrics
        y_true_bin = (y_test == "Acceptable").astype(int).to_numpy()

        try:
            pr_auc = average_precision_score(y_true_bin, y_score_test)
        except ValueError:
            pr_auc = None

        accuracy = accuracy_score(y_test, y_pred_test)
        report = classification_report(
            y_test, y_pred_test, output_dict=True, zero_division=0
        )

        f1_accept = report.get("Acceptable", {}).get("f1-score", 0)
        f1_notaccept = report.get("Not Acceptable", {}).get("f1-score", 0)
        recall_accept = report.get("Acceptable", {}).get("recall", 0)
        recall_notaccept = report.get("Not Acceptable", {}).get("recall", 0)
        precision_accept = report.get("Acceptable", {}).get("precision", 0)
        precision_notaccept = report.get("Not Acceptable", {}).get("precision", 0)

        actual_model = final_model.named_steps["model"] if is_pipeline else final_model
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
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None  # For day_line output

        # Issue warning if PR-AUC undefined
        if pr_auc is None:
            logger.warn("004", f"{days}: PR-AUC undefined (only one class in test set)")

        # One-line day summary (main output for quiet mode)
        logger.day_line(
            day=days,
            n_samples=len(day_trainval),
            n_minority=minority_count,
            n_folds=n_folds,
            cv_score=mean_cv_score,
            threshold=final_threshold,
            recall_na=recall_notaccept,
            fpr=fpr,
            saved=True,
        )

        # Save day results
        day_dir = model_dir / days
        day_dir.mkdir(parents=True, exist_ok=True)

        save_organoid_predictions(
            day_test.reset_index(drop=True),
            y_test,
            y_pred_test,
            y_score_test,
            day_dir / "organoid_predictions.csv",
        )

        # Save feature importance
        if not is_pipeline:
            feature_importance = final_model.feature_importances_
            importance_df = pd.DataFrame(
                {
                    "feature": X_trainval_clean.columns,
                    "importance": feature_importance,
                }
            ).sort_values("importance", ascending=False)
            importance_df.to_csv(day_dir / "feature_importance.csv", index=False)

        # Save metrics (E) Enhanced with fold data and stability diagnostics
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
            "threshold_used": float(final_threshold),
            "cv_score": float(mean_cv_score),
            "confusion_matrix": {
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            },
            # (E) Enhanced logging for reproducibility
            "n_iter": search_iterations,
            "search_n_jobs": search_n_jobs,
            "inner_cv_splits": 3,
            "outer_cv_splits": n_folds,
            "fold_thresholds": fold_thresholds,
            "fold_scores": fold_scores,
            "fold_search_best_scores": fold_search_best_scores,
            # (F) Stability diagnostics
            "std_threshold": std_threshold,
            "std_score": std_score,
            "threshold_unstable": std_threshold > 0.15,
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
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh_cm else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(day_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # Add to summary
        results_summary.append(
            {
                "Day": days,
                "Day_No": day_num,
                "Test_Accuracy": accuracy,
                "Test_F1_Acceptable": f1_accept,
                "Test_F1_NotAcceptable": f1_notaccept,
                "Test_Recall_Acceptable": recall_accept,
                "Test_Recall_NotAcceptable": recall_notaccept,
                "Test_Precision_Acceptable": precision_accept,
                "Test_Precision_NotAcceptable": precision_notaccept,
                "Test_Specificity": specificity,
                "Test_PR_AUC": pr_auc,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Threshold_Used": final_threshold,
                "CV_Score": mean_cv_score,
            }
        )

    if not results_summary:
        logger.warn("012", "No results produced")
        return

    # Save summary
    summary_df = pd.DataFrame(results_summary).sort_values("Day_No")
    summary_df.to_csv(model_dir / "results_summary.csv", index=False)

    # Create metrics-by-day plot (4 panels: F1 Not Acceptable, F1 Acceptable, TNR, PR-AUC)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(
        summary_df["Day_No"], summary_df["Test_F1_NotAcceptable"], "o-", color="orange"
    )
    axes[0, 0].set_title("Test F1 Score (Not Acceptable)")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    axes[0, 1].plot(
        summary_df["Day_No"], summary_df["Test_F1_Acceptable"], "o-", color="blue"
    )
    axes[0, 1].set_title("Test F1 Score (Acceptable)")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    axes[1, 0].plot(
        summary_df["Day_No"], summary_df["Test_Specificity"], "o-", color="purple"
    )
    axes[1, 0].set_title("Test Specificity (TNR)")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Specificity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    auc_data = summary_df.dropna(subset=["Test_PR_AUC"])
    if len(auc_data) > 0:
        axes[1, 1].plot(
            auc_data["Day_No"], auc_data["Test_PR_AUC"], "o-", color="green"
        )
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
        description="Train Metabolite Classifiers with proper nested CV"
    )
    parser.add_argument(
        "--scoring",
        choices=["f1_notaccept", "f1_weighted", "recall_notaccept", "macro_f1"],
        default="f1_notaccept",
        help="Scoring metric for CV and threshold tuning (default: f1_notaccept)",
    )
    parser.add_argument(
        "--imbalance",
        choices=["class_weight", "scale_pos_weight", "focal_loss", "smote"],
        default="class_weight",
        help="Class imbalance handling method (default: class_weight)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        choices=[3, 5],
        default=3,
        help="Number of outer CV folds (default: 3)",
    )
    parser.add_argument(
        "--use_second_order_growth",
        action="store_true",
        help="Add second-order growth features (acceleration)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=None,
        help="Override RandomizedSearchCV n_iter (default: 30)",
    )
    parser.add_argument(
        "--search_n_jobs",
        type=int,
        default=None,
        help="n_jobs for RandomizedSearchCV (default: 1 for stability)",
    )
    parser.add_argument(
        "--day_filter",
        type=str,
        default=None,
        help="Run only on specified day (e.g., 'Dy03') for smoke testing",
    )
    # Verbosity/strictness flags
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-fold details and debug info",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=True,
        help="Minimal output: startup block + one line per day + warnings (default)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Skip days with insufficient samples, abort on errors",
    )

    args = parser.parse_args()

    # Configure global logger
    global logger
    logger = Logger(verbose=args.verbose, quiet=args.quiet, strict=args.strict)

    # Set defaults for search_n_jobs (default: 1 for GPU for stability)
    if args.search_n_jobs is None:
        args.search_n_jobs = 1

    # Set defaults for n_iter
    if args.n_iter is None:
        args.n_iter = SEARCH_ITERATIONS

    # GPU availability check - fail fast if GPU not available
    check_gpu_available()

    train_data_path = "data_splits/both_train_base.json"
    val_data_path = "data_splits/both_val_base.json"
    test_data_path = "data_splits/both_test_base.json"
    output_dir = "analysis/metabolites/classifier/outputs_metabolites"

    # Startup config block (replaces multiple print statements)
    logger.startup_block(
        {
            "scoring": args.scoring,
            "imbalance": args.imbalance,
            "n_folds": args.n_folds,
            "n_iter": args.n_iter,
            "search_n_jobs": args.search_n_jobs,
            "use_second_order": args.use_second_order_growth,
            "day_filter": args.day_filter,
            "output_dir": output_dir,
        }
    )

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
    train_df = compute_growth_features(
        train_df, use_second_order=args.use_second_order_growth
    )
    val_df = compute_growth_features(
        val_df, use_second_order=args.use_second_order_growth
    )
    test_df = compute_growth_features(
        test_df, use_second_order=args.use_second_order_growth
    )

    # Construct model name (always GPU)
    model_name = f"lgbm_{args.scoring}_{args.imbalance}_{args.n_folds}fold_gpu"
    if args.use_second_order_growth:
        model_name += "_accel"

    train_metabolite_classifier_per_day(
        train_df,
        val_df,
        test_df,
        output_dir,
        model_name,
        scoring=args.scoring,
        imbalance=args.imbalance,
        n_folds=args.n_folds,
        use_second_order=args.use_second_order_growth,
        n_iter=args.n_iter,
        search_n_jobs=args.search_n_jobs,
        day_filter=args.day_filter,
    )


if __name__ == "__main__":
    main()
