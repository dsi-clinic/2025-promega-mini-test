#!/usr/bin/env python3
"""Three-thread combined classifier: metabolite + morphology (+ optional image embeddings).

Extends the per-day LightGBM metabolite baseline by adding:
  - Morphology thread: ``mask_area_um2`` and ``edge_fraction`` per day, plus
    temporal deltas (day-over-day change in area).
  - Image thread (optional, --image-embeddings): pre-extracted EfficientNet-B0
    embeddings (1280-dim) reduced to ``--pca-components`` via PCA.

Three fusion modes (``--fusion``):
  - ``met+morph``   metabolite features + morphology features (default, CPU-only)
  - ``met+img``     metabolite features + PCA image embeddings (needs embedding CSV)
  - ``all``         all three threads combined

Evaluation: same StratifiedGroupKFold(5) CV protocol as the metabolite baseline.
Comparison output includes metabolite-only alongside combined results so the
gain/loss from fusion is immediately visible.

Outputs:
  $ANALYSIS_OUTPUT_DIR/combined/results_<fusion>_<suffix>.json
  $ANALYSIS_OUTPUT_DIR/figures/combined_<fusion>_<suffix>_balanced_accuracy.png

Usage:
  make run ARGS="-m analysis.paper_2026_04.combined_lgbm"
  make run ARGS="-m analysis.paper_2026_04.combined_lgbm --fusion all --image-embeddings /path/to/image_embeddings.csv"
  make run ARGS="-m analysis.paper_2026_04.combined_lgbm --days Dy30"
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    OrganoidDataset,
    filters_for_mode,
    get_edge_fraction,
    get_mask_area_um2,
)
from pipeline.splits import Splits

from .common import compute_classification_metrics, plot_balanced_accuracy_by_day
from .metabolites_train import MODEL_SPECS

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEED = 42
ALL_DATA_PATH = "data/all_data.json"
N_FOLDS = 5

# Default path where image_embeddings.csv is written by the feature_correlation analysis.
_DEFAULT_EMB_CSV = Path(
    "/net/projects2/promega/2026_04_15_data/analysis_output/combined/feature_correlation/image_embeddings.csv"
)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _build_metabolite_frame(ds, day, split="all"):
    """Return (X_met, y, org_ids) for one day via the canonical accessor.

    Uses the same feature set as metabolites_train.py (concentration_uM +
    initial_concentration per metabolite, no growth).
    """
    X, y, _names, org_ids = ds.get_metabolite_features(
        split, day, include_growth=False, include_initial=True
    )
    return X, y, org_ids


def _build_morphology_frame(ds, day, split="all"):
    """Return (X_morph, org_ids) for one day.

    Features:
      - mask_area_um2 at the current day (log-scaled to stabilise range)
      - edge_fraction at the current day
      - log area delta from the previous day (0 for Dy03)
    """
    subset = ds.get_split(split, day=day)
    day_idx = DAY_ORDER.index(day) if day in DAY_ORDER else -1
    prev_day = DAY_ORDER[day_idx - 1] if day_idx > 0 else None

    rows = []
    org_ids = []
    for org_id, info in subset.items():
        rec = info["records"].get(day)
        if rec is None:
            continue
        area = get_mask_area_um2(rec)
        ef = get_edge_fraction(rec)
        if area is None:
            continue  # drop rows with missing segmentation area

        log_area = np.log1p(area)
        ef_val = ef if ef is not None else 0.0

        # Delta area (log scale) from previous day
        delta_log_area = 0.0
        if prev_day is not None:
            prev_rec = info["records"].get(prev_day)
            prev_area = get_mask_area_um2(prev_rec) if prev_rec else None
            if prev_area:
                delta_log_area = log_area - np.log1p(prev_area)

        rows.append([log_area, ef_val, delta_log_area])
        org_ids.append(org_id)

    X = np.array(rows, dtype=float)
    return X, org_ids


def _build_image_frame(emb_df, day):
    """Return (X_img, org_ids) from a pre-extracted embeddings DataFrame.

    emb_df must have columns: org_id, day, and img_emb_0..img_emb_1279.
    """
    day_df = emb_df[emb_df["day"] == day].copy()
    emb_cols = [c for c in day_df.columns if c.startswith("img_emb_")]
    if not emb_cols:
        return None, []
    X = day_df[emb_cols].values.astype(float)
    return X, day_df["org_id"].tolist()


def _align_features(met_ids, X_met, morph_ids=None, X_morph=None,
                    img_ids=None, X_img=None):
    """Intersect organoid IDs across active threads, return aligned arrays.

    Returns (X_combined, aligned_ids).
    """
    # Start with metabolite IDs as the reference
    active_sets = [set(met_ids)]
    if morph_ids is not None:
        active_sets.append(set(morph_ids))
    if img_ids is not None:
        active_sets.append(set(img_ids))

    common = active_sets[0]
    for s in active_sets[1:]:
        common = common & s
    if not common:
        return None, []

    keep = [i for i, oid in enumerate(met_ids) if oid in common]
    aligned_ids = [met_ids[i] for i in keep]
    parts = [X_met[keep]]

    if morph_ids is not None:
        morph_idx = {oid: i for i, oid in enumerate(morph_ids)}
        morph_keep = [morph_idx[oid] for oid in aligned_ids]
        parts.append(X_morph[morph_keep])

    if img_ids is not None:
        img_idx = {oid: i for i, oid in enumerate(img_ids)}
        img_keep = [img_idx[oid] for oid in aligned_ids]
        parts.append(X_img[img_keep])

    return np.hstack(parts), aligned_ids


def _apply_pca(X_img, n_components, scaler=None, pca=None, fit=True):
    """Scale + PCA on image embeddings. Returns (X_reduced, scaler, pca)."""
    if fit:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_img)
        n = min(n_components, X_s.shape[1], X_s.shape[0] - 1)
        pca = PCA(n_components=n, random_state=SEED)
        X_r = pca.fit_transform(X_s)
    else:
        X_s = scaler.transform(X_img)
        X_r = pca.transform(X_s)
    return X_r, scaler, pca


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def _run_cv_for_day(X, y, org_ids, *, n_folds=N_FOLDS, seed=SEED):
    """StratifiedGroupKFold(n_folds) CV over organoids.

    Groups = org_ids so each organoid lands in exactly one fold.
    Returns aggregated metrics dict.
    """
    spec = MODEL_SPECS["lgbm"]
    from sklearn.metrics import balanced_accuracy_score

    kf = StratifiedGroupKFold(n_splits=n_folds)
    groups = np.array(org_ids)
    ba_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        n_neg = (y_tr == 0).sum()
        n_pos = (y_tr == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        # Scaler on this fold's training data only
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)

        model = spec.factory()
        # For LightGBM set scale_pos_weight dynamically
        try:
            model.set_params(scale_pos_weight=spw)
        except Exception:
            pass
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_va_s)
        ba = balanced_accuracy_score(y_va, y_pred)
        ba_scores.append(ba)

    if not ba_scores:
        return None
    return {
        "balanced_accuracy_mean": float(np.mean(ba_scores)),
        "balanced_accuracy_std": float(np.std(ba_scores)),
        "n_folds": len(ba_scores),
        "n_samples": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Per-cohort run
# ---------------------------------------------------------------------------

def run_combined(ds, days, fusion, emb_df, pca_components):
    """Run combined model for each day. Returns {day: metrics_dict}."""
    results_met_only = {}
    results_combined = {}

    for day in days:
        if day not in ds.days:
            continue

        # ---- metabolite features ------------------------------------------
        X_met, y, met_ids = _build_metabolite_frame(ds, day)
        if X_met.shape[0] < N_FOLDS * 2:
            logger.info("  %s: too few samples (%d), skipping", day, X_met.shape[0])
            continue

        # ---- metabolite-only baseline --------------------------------------
        m_only = _run_cv_for_day(X_met, y, met_ids)
        if m_only:
            results_met_only[day] = m_only
            logger.info("  [met-only]   %s  n=%d  BalAcc=%.4f ± %.4f",
                        day, m_only["n_samples"],
                        m_only["balanced_accuracy_mean"], m_only["balanced_accuracy_std"])

        # ---- morphology features ------------------------------------------
        morph_ids = img_ids = None
        X_morph = X_img = None

        if fusion in ("met+morph", "all"):
            X_morph, morph_ids = _build_morphology_frame(ds, day)
            if X_morph.shape[0] == 0:
                morph_ids = None
                X_morph = None

        # ---- image features -----------------------------------------------
        if emb_df is not None and fusion in ("met+img", "all"):
            X_img_raw, img_ids_raw = _build_image_frame(emb_df, day)
            if X_img_raw is not None and len(img_ids_raw) > 0:
                # Align to met_ids first to get training set for PCA fit
                common_train = set(met_ids) & set(img_ids_raw)
                if len(common_train) > pca_components + 1:
                    idx_in_raw = [i for i, oid in enumerate(img_ids_raw) if oid in common_train]
                    X_img_fit = X_img_raw[idx_in_raw]
                    X_img_reduced, _scaler, _pca = _apply_pca(X_img_fit, pca_components, fit=True)
                    # Map to all rows in img_ids_raw
                    X_img_all, _, _ = _apply_pca(X_img_raw, pca_components,
                                                 scaler=_scaler, pca=_pca, fit=False)
                    X_img = X_img_all
                    img_ids = img_ids_raw
                else:
                    img_ids = None

        # ---- align and combine --------------------------------------------
        X_combined, aligned_ids = _align_features(
            met_ids, X_met,
            morph_ids=morph_ids, X_morph=X_morph,
            img_ids=img_ids, X_img=X_img,
        )
        if X_combined is None or len(aligned_ids) < N_FOLDS * 2:
            logger.info("  [combined]   %s  insufficient aligned samples, skipping", day)
            continue

        # Rebuild y for aligned IDs
        label_map = {oid: int(info["label"] == "Not Acceptable")
                     for oid, info in ds.iter_organoids()}
        y_aligned = np.array([label_map[oid] for oid in aligned_ids], dtype=int)

        m_comb = _run_cv_for_day(X_combined, y_aligned, aligned_ids)
        if m_comb:
            results_combined[day] = m_comb
            delta = (m_comb["balanced_accuracy_mean"]
                     - (results_met_only.get(day, {}).get("balanced_accuracy_mean", float("nan"))))
            delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "n/a"
            logger.info("  [combined]   %s  n=%d  BalAcc=%.4f ± %.4f  (Δ vs met-only: %s)",
                        day, m_comb["n_samples"],
                        m_comb["balanced_accuracy_mean"], m_comb["balanced_accuracy_std"],
                        delta_s)

    return results_met_only, results_combined


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_results(fusion, suffix, met_only, combined):
    out_dir = ANALYSIS_OUTPUT_DIR / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"met_only": met_only, "combined": combined, "fusion": fusion}
    path = out_dir / f"results_{fusion}_{suffix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved results to %s", path)
    return path


def _plot_comparison(fusion, suffix, met_only, combined, days):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / f"combined_{fusion}_{suffix}_balanced_accuracy.png"

    # Reformat for plot_balanced_accuracy_by_day: {model_name: {day: metrics_dict}}
    # metrics_dict needs 'balanced_accuracy' key
    met_for_plot = {d: {"balanced_accuracy": v["balanced_accuracy_mean"]}
                    for d, v in met_only.items()}
    comb_for_plot = {d: {"balanced_accuracy": v["balanced_accuracy_mean"]}
                     for d, v in combined.items()}

    thread_label = {
        "met+morph": "Metabolite + Morphology",
        "met+img": "Metabolite + Image (PCA)",
        "all": "All Three Threads",
    }.get(fusion, fusion)

    plot_balanced_accuracy_by_day(
        {"Metabolite only (LightGBM)": met_for_plot,
         thread_label: comb_for_plot},
        day_order=days,
        output_path=out_path,
        title=f"Combined ({fusion}) vs Metabolite-Only: Balanced Accuracy by Day",
        style_overrides={
            "Metabolite only (LightGBM)": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
            thread_label: {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
        },
    )
    logger.info("Saved figure to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", choices=["met+morph", "met+img", "all"],
                        default="met+morph",
                        help="Which threads to combine (default: met+morph)")
    parser.add_argument("--image-embeddings", default=str(_DEFAULT_EMB_CSV),
                        help="Path to image_embeddings.csv (needed for met+img / all)")
    parser.add_argument("--pca-components", type=int, default=30,
                        help="PCA components for image embeddings (default: 30)")
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    days = args.days or DAY_ORDER

    # Load image embeddings if needed
    emb_df = None
    if args.fusion in ("met+img", "all"):
        emb_path = Path(args.image_embeddings)
        if not emb_path.exists():
            logger.warning("Image embeddings CSV not found at %s; image thread disabled", emb_path)
        else:
            logger.info("Loading image embeddings from %s", emb_path)
            emb_df = pd.read_csv(emb_path)
            logger.info("  %d rows, %d columns", *emb_df.shape)

    # Load dataset — all organoids in one "all" split for CV
    ds = OrganoidDataset(
        ALL_DATA_PATH,
        filters=filters_for_mode("base"),
    )
    ds.apply_splits(
        Splits.from_dict({oid: "all" for oid in ds.organoid_ids},
                         name="cv_all_combined",
                         provenance="single-split CV harness for combined model"),
        strict=True,
    )
    logger.info("Dataset: %d organoids, days: %s", len(ds.organoid_ids), ds.days)

    suffix = f"pca{args.pca_components}" if emb_df is not None else "nopca"
    logger.info("\n=== Running combined model: fusion=%s ===\n", args.fusion)
    met_only, combined = run_combined(ds, days, args.fusion, emb_df, args.pca_components)

    if not combined:
        logger.warning("No combined results produced.")
        return

    _write_results(args.fusion, suffix, met_only, combined)
    _plot_comparison(args.fusion, suffix, met_only, combined, days)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — fusion={args.fusion}")
    print(f"{'='*60}")
    print(f"{'Day':<10} {'Met-only BalAcc':>16} {'Combined BalAcc':>16} {'Δ':>8}")
    for day in days:
        m = met_only.get(day, {}).get("balanced_accuracy_mean")
        c = combined.get(day, {}).get("balanced_accuracy_mean")
        if m is None and c is None:
            continue
        m_s = f"{m:.4f}" if m is not None else "  n/a "
        c_s = f"{c:.4f}" if c is not None else "  n/a "
        delta = (c - m) if (m is not None and c is not None) else None
        d_s = f"{delta:+.4f}" if delta is not None else "  n/a "
        print(f"{day:<10} {m_s:>16} {c_s:>16} {d_s:>8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
