#!/usr/bin/env python3
"""Reproduce metabolite-only model results: LightGBM and Logistic Regression per day.

Pipeline (same shape for both models):
  1. Build features via OrganoidDataset.get_metabolite_features(
         split, day, include_growth=True, include_initial=True)
     — single source of truth for concentration / initial / growth columns.
  2. GridSearchCV with StratifiedGroupKFold(3) on train.
  3. Threshold tuning on validation (model-specific grid + scoring).
  4. Refit on train+val, evaluate on test.

Differences vs LightGBM are encoded as a small DSL in MODEL_SPECS, not as
forked train_lgbm_day / train_logreg_day functions. Outputs match the legacy
results.json schema so feature_importance.py + three_model_plot.py keep
working.

Outputs:
  - $ANALYSIS_OUTPUT_DIR/metabolites/results.json
  - $ANALYSIS_OUTPUT_DIR/figures/LightGBM_vs_Logistic_Regression.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.metabolites_train"
    make run ARGS="-m analysis.paper_2026_04.metabolites_train --days Dy30"
"""

import argparse
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    OrganoidDataset,
)
from pipeline.splits import Splits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .common_renamed import (
    compute_classification_metrics,
    plot_balanced_accuracy_by_day,
)

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "metabolites"


@dataclass
class ModelSpec:
    """All the bits that distinguish LightGBM-day from LogReg-day."""

    name: str  # 'lgbm' / 'logreg'
    display: str  # 'LightGBM' / 'Logistic Regression'
    factory: Callable  # () → unfit estimator with class-balanced base config
    param_grid: dict
    cv_scoring: str  # 'f1' (lgbm: minority class) or 'f1_weighted' (lr)
    threshold_grid: np.ndarray
    threshold_scoring: Callable[[np.ndarray, np.ndarray], float]
    use_scaler: bool
    captures_feature_importance: bool
    include_growth: bool = True  # LightGBM uses differences per paper
    include_initial: bool = True


def _f1_minority(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def _f1_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _lgbm_factory():
    """LightGBM with class_weight='balanced' per paper: 'class weighting to
    address label imbalance: the Not Acceptable class receiving additional
    emphasis during fitting.'"""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=SEED,
        verbosity=-1,
        n_jobs=1,
    )


def _logreg_factory():
    """LogReg matching old `train_metabolites_logreg_nogrowth.py`. Paper does
    not specify solver explicitly.
    max_iter left at sklearn default (100)."""
    return LogisticRegression(
        class_weight="balanced",
        random_state=SEED,
        solver="liblinear",
    )


MODEL_SPECS = {
    "lgbm": ModelSpec(
        name="lgbm",
        display="LightGBM",
        factory=_lgbm_factory,
        param_grid={
            "max_depth": [3, 6],
            "num_leaves": [31, 47, 63],
            "min_child_samples": [10, 20],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
        },
        cv_scoring="f1",
        threshold_grid=np.linspace(0.3, 0.7, 9),
        threshold_scoring=_f1_minority,
        use_scaler=False,
        captures_feature_importance=True,
    ),
    "logreg": ModelSpec(
        name="logreg",
        display="Logistic Regression",
        factory=_logreg_factory,
        param_grid={
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
        },
        cv_scoring="f1_weighted",
        threshold_grid=np.linspace(0.1, 0.9, 17),
        threshold_scoring=_f1_weighted,
        use_scaler=False,  # paper: deliberately simple baseline, no scaling
        captures_feature_importance=False,
        include_growth=False,  # paper: absolute concentrations only
        include_initial=False,
    ),
}


def _features_for_day(ds: OrganoidDataset, day: str, spec: "ModelSpec"):
    """Pull (X, y, names, ids) for each split on one day.

    Feature set (include_growth, include_initial) comes from the ModelSpec.
    """
    out = {}
    for split in ("train", "val", "test"):
        X, y, names, ids = ds.get_metabolite_features(
            split,
            day,
            include_growth=spec.include_growth,
            include_initial=spec.include_initial,
        )
        out[split] = (X, y, names, ids)
    return out


def _train_one(
    spec: ModelSpec, day: str, day_features: dict, *, verbose: bool
) -> dict | None:
    X_train, y_train, feat_names, ids_train = day_features["train"]
    X_val, y_val, _, _ = day_features["val"]
    X_test, y_test, _, _ = day_features["test"]

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    # Optional standardization (LR uses it; LGBM doesn't).
    if spec.use_scaler:
        scaler = StandardScaler()
        X_train_p = scaler.fit_transform(X_train)
        X_val_p = (
            scaler.transform(X_val)
            if len(X_val) > 0
            else np.empty((0, X_train.shape[1]))
        )
        X_test_p = scaler.transform(X_test)
    else:
        X_train_p, X_val_p, X_test_p = X_train, X_val, X_test

    # Phase 1: GridSearchCV on train (group-aware).
    base = spec.factory()
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        base, spec.param_grid, cv=cv, scoring=spec.cv_scoring, n_jobs=-1, refit=True
    )
    grid.fit(X_train_p, y_train, groups=np.asarray(ids_train))
    best_params = grid.best_params_
    if verbose:
        print(f"  Best params: {best_params}")

    # Phase 2: threshold tuning on validation.
    best_threshold = 0.5
    if len(X_val_p) > 0 and len(np.unique(y_val)) > 1:
        val_probs = grid.predict_proba(X_val_p)[:, 1]
        best_score = -np.inf
        for t in spec.threshold_grid:
            score = spec.threshold_scoring(y_val, (val_probs >= t).astype(int))
            if score > best_score:
                best_score = score
                best_threshold = t
    if verbose:
        print(f"  Best threshold: {best_threshold:.2f}")

    # Phase 3: refit on train+val, evaluate on test.
    if len(X_val_p) > 0:
        X_tv = np.vstack([X_train_p, X_val_p])
        y_tv = np.concatenate([y_train, y_val])
    else:
        X_tv, y_tv = X_train_p, y_train

    final = spec.factory()
    final.set_params(**best_params)
    final.fit(X_tv, y_tv)

    test_probs = final.predict_proba(X_test_p)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    metrics["threshold"] = float(best_threshold)
    metrics["best_params"] = best_params
    metrics["feature_names"] = feat_names

    if spec.captures_feature_importance and hasattr(final, "feature_importances_"):
        ranked = sorted(
            zip(feat_names, final.feature_importances_, strict=False),
            key=lambda kv: kv[1],
            reverse=True,
        )
        metrics["feature_importance"] = [
            {"feature": f, "importance": int(i)} for f, i in ranked
        ]

    return metrics


def _print_aggregate(results: dict) -> None:
    print(f"\n{'=' * 60}\nAGGREGATE COMPARISON (Table 3)\n{'=' * 60}")
    for spec in MODEL_SPECS.values():
        per_day = results.get(spec.name, {})
        if not per_day:
            continue
        accs, bal_accs, recall_nas = [], [], []
        zero_recall_days = 0
        best_bal_acc = 0.0
        for day in DAY_ORDER:
            m = per_day.get(day)
            if m is None:
                continue
            accs.append(m["accuracy"])
            bal_accs.append(m["balanced_accuracy"])
            r_na = m["recall_not_acceptable"]
            recall_nas.append(r_na)
            zero_recall_days += int(r_na == 0.0)
            best_bal_acc = max(best_bal_acc, m["balanced_accuracy"])
        n_days = len(accs)
        print(f"\n{spec.display}:")
        print(f"  Avg Accuracy:       {np.mean(accs):.1%}")
        print(f"  Avg Balanced Acc:   {np.mean(bal_accs):.1%}")
        print(f"  Avg Recall (N.A.):  {np.mean(recall_nas):.1%}")
        print(f"  Days Recall_NA = 0: {zero_recall_days}/{n_days}")
        print(f"  Best Balanced Acc:  {best_bal_acc:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days",
        nargs="+",
        default=None,
        help="Specific days to train (e.g. Dy30 Dy24)",
    )
    parser.add_argument(
        "--skip-lr", action="store_true", help="Skip logistic regression"
    )
    parser.add_argument("--skip-lgbm", action="store_true", help="Skip LightGBM")
    args = parser.parse_args()

    enabled = []
    if not args.skip_lgbm:
        enabled.append(MODEL_SPECS["lgbm"])
    if not args.skip_lr:
        enabled.append(MODEL_SPECS["logreg"])

    ds = OrganoidDataset(ALL_DATA_PATH, splits=Splits.canonical())
    print(ds.summary())

    days_to_train = args.days or DAY_ORDER
    results: dict = {spec.name: {} for spec in enabled}

    for day in days_to_train:
        if day not in ds.days:
            print(f"\nSkipping {day} (no data)")
            continue
        # Cache features by (include_growth, include_initial) so LightGBM and
        # LogReg can share the call when their feature sets match.
        feature_cache = {}
        for spec in enabled:
            cache_key = (spec.include_growth, spec.include_initial)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = _features_for_day(ds, day, spec)
            day_features = feature_cache[cache_key]
            print(f"\n{'=' * 50}\n{spec.display} - {day}\n{'=' * 50}")
            m = _train_one(spec, day, day_features, verbose=True)
            if m is None:
                continue
            results[spec.name][day] = m
            print(f"  Accuracy:     {m['accuracy']:.4f}")
            print(f"  Balanced Acc: {m['balanced_accuracy']:.4f}")
            print(f"  Recall (NA):  {m['recall_not_acceptable']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    _print_aggregate(results)

    if results.get("lgbm") and results.get("logreg"):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_balanced_accuracy_by_day(
            {"LightGBM": results["lgbm"], "Logistic Regression": results["logreg"]},
            day_order=DAY_ORDER,
            output_path=FIGURE_DIR / "LightGBM_vs_Logistic_Regression.png",
            title="LightGBM vs Logistic Regression: Balanced Accuracy by Day",
            style_overrides={
                "LightGBM": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
                "Logistic Regression": {
                    "color": "#ff7f0e",
                    "marker": "s",
                    "linestyle": "--",
                },
            },
            late_stage_shade_from_day=24,
        )


if __name__ == "__main__":
    main()
