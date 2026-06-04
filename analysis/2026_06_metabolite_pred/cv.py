#!/usr/bin/env python3
"""Stratified cross-validation harness for metabolite prediction.

Ports the modeling config from ``analysis.paper_2026_04.metabolites_train`` (the
``MODEL_SPECS`` DSL: estimator factory, hyperparameter grid, CV scoring,
threshold grid + scoring, scaler flag) but replaces its single held-out
train/val/test split with **nested cross-validation**, which is more stable for
our small cohorts (198 / 248 organoids with a small minority class).

For one (cohort, day, model) ``run_cv_for_day`` does, per outer fold:
  1. scale on the train fold only (logreg);
  2. inner GridSearchCV (group-aware) to pick hyperparameters;
  3. tune the decision threshold on the outer-train via inner cross_val_predict;
  4. refit and predict the held-out outer-test, storing out-of-fold predictions.

It returns pooled out-of-fold metrics (every organoid predicted exactly once)
plus per-fold mean/std of balanced accuracy and recall(Not Acceptable).
"""

import logging

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

from analysis.paper_2026_04.common import compute_classification_metrics

logger = logging.getLogger(__name__)


def _minority(y) -> int:
    return int(min((y == 0).sum(), (y == 1).sum()))


def _scale_pos_weight(y) -> float:
    """LightGBM class weight, matching the source: n_neg / max(n_pos, 1)."""
    return float((y == 0).sum()) / max(int((y == 1).sum()), 1)


def _make_base(spec, y_train):
    """Fresh unfit estimator for this train fold (lgbm gets a fold-local spw)."""
    if spec.name == "lgbm":
        return spec.factory(scale_pos_weight=_scale_pos_weight(y_train))
    return spec.factory()


def _tune_threshold(spec, estimator, X_train, y_train, ids_train, inner_splits, seed):
    """Pick the threshold maximizing spec.threshold_scoring on inner OOF probs.

    Uses a clone of the grid-selected estimator. For lgbm the clone keeps the
    outer-train ``scale_pos_weight`` (a documented coarse approximation: the
    source likewise does not re-derive it per inner fold); for logreg the inner
    folds reuse the outer-train-fit scaling, also an accepted approximation.
    """
    if inner_splits < 2 or len(np.unique(y_train)) < 2:
        return 0.5
    cvp = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    probs = cross_val_predict(
        clone(estimator), X_train, y_train,
        groups=ids_train, cv=cvp, method="predict_proba",
    )[:, 1]
    best_t, best_score = 0.5, -np.inf
    for t in spec.threshold_grid:
        score = spec.threshold_scoring(y_train, (probs >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


def run_cv_for_day(spec, X, y, ids, *, n_folds: int = 5, seed: int = 42, verbose: bool = False):
    """Nested CV for one model on one day. Returns a metrics dict or None.

    None when the day has no rows or too few minority examples to stratify even
    two outer folds.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    ids = np.asarray(ids)
    if len(X) == 0:
        return None

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    minority = min(n_pos, n_neg)
    eff_folds = min(n_folds, minority)
    if eff_folds < 2:
        logger.info("  skip: minority=%d < 2, cannot stratify", minority)
        return None
    if eff_folds < n_folds:
        logger.info("  reducing outer folds %d -> %d (minority=%d)", n_folds, eff_folds, minority)

    outer = StratifiedGroupKFold(n_splits=eff_folds, shuffle=True, random_state=seed)
    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1, dtype=int)
    fold_bal_acc, fold_recall_na = [], []

    for tr, te in outer.split(X, y, groups=ids):
        X_tr, y_tr, ids_tr = X[tr], y[tr], ids[tr]
        X_te, y_te = X[te], y[te]

        if spec.use_scaler:
            scaler = StandardScaler()
            X_tr_p = scaler.fit_transform(X_tr)
            X_te_p = scaler.transform(X_te)
        else:
            X_tr_p, X_te_p = X_tr, X_te

        base = _make_base(spec, y_tr)

        inner_splits = min(3, _minority(y_tr))
        if inner_splits < 2:
            logger.info("  inner stratification impossible; using base params")
            best = base
        else:
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
            grid = GridSearchCV(base, spec.param_grid, cv=inner,
                                scoring=spec.cv_scoring, n_jobs=-1, refit=True)
            grid.fit(X_tr_p, y_tr, groups=ids_tr)
            best = grid.best_estimator_

        threshold = _tune_threshold(spec, best, X_tr_p, y_tr, ids_tr, inner_splits, seed)

        best.fit(X_tr_p, y_tr)  # ensure refit on the full outer-train
        prob = best.predict_proba(X_te_p)[:, 1]
        pred = (prob >= threshold).astype(int)
        oof_prob[te] = prob
        oof_pred[te] = pred

        fold_m = compute_classification_metrics(y_te, pred, prob)
        fold_bal_acc.append(fold_m["balanced_accuracy"])
        fold_recall_na.append(fold_m["recall_not_acceptable"])

    assert not np.isnan(oof_prob).any(), "some organoids never predicted"
    assert (oof_pred >= 0).all(), "some organoids never predicted"

    metrics = compute_classification_metrics(y, oof_pred, oof_prob)
    metrics.update({
        "balanced_accuracy_cv_mean": float(np.mean(fold_bal_acc)),
        "balanced_accuracy_cv_std": float(np.std(fold_bal_acc)),
        "recall_not_acceptable_cv_mean": float(np.mean(fold_recall_na)),
        "recall_not_acceptable_cv_std": float(np.std(fold_recall_na)),
        "n_folds": int(eff_folds),
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
    })
    return metrics
