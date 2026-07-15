#!/usr/bin/env python3
"""Metabolite & feature-group ablation for the per-day good/bad classifier.

Quantifies how much each metabolite (and each feature-engineering group)
contributes to the per-day LightGBM (+ Logistic Regression) classifier by
*ablation*: take the headline ``nominal_delta`` feature matrix as the baseline,
drop one thing at a time, re-run the same nested cross-validation
(``cv.run_cv_for_day``) and measure the change in balanced accuracy vs the
full-feature baseline — per day, for both cohorts (strong-consensus=198,
full=248).

Baseline config (headline ``nominal_delta`` from ``run.py``):
    include_growth=True, include_initial=True, normalize_by_size=False,
    winsorize=False.

Ablations
---------
Leave-one-metabolite-out (6): for each of GlucoseGlo, GlutamateGlo, LactateGlo,
  PyruvateGlo, BCAAGlo, MalateGlo, drop *all* of that metabolite's feature
  columns (concentration, initial, growth) and re-evaluate. This is a pure
  column drop — the organoid/row set is unchanged (asserted; rule 11).

Leave-one-feature-group-out (3):
  - ``group:growth``  — drop the ``*_growth`` (delta) columns.
  - ``group:initial`` — drop the ``*_initial_concentration`` columns.
    Both are pure column drops (row set unchanged; asserted).
  - ``group:size_norm`` — this dial is *off* in the baseline, so it cannot be a
    column drop. Instead we rebuild features with ``normalize_by_size=True`` and
    compare, on the organoids common to both matrices, size-normalized vs
    nominal. Size-normalization needs a segmentation area, so it can legitimately
    drop organoids; the dropped count is logged (rule 15) and the *baseline* is
    re-scored on the same common set so the comparison is apples-to-apples.

Metric conventions (per cohort/day/model):
  - ``balanced_accuracy`` — pooled out-of-fold balanced accuracy for that matrix.
  - ``balacc_drop`` (leave-out ablations) = baseline − ablated. Positive drop =
    removing the thing *hurt* = the feature is contributing.
  - ``balacc_gain_from_size_norm`` (size_norm) = size-normalized − nominal on the
    common set. Positive = turning size-normalization on helped.

Outputs (both gitignored):
  - $ANALYSIS_OUTPUT_DIR/metabolite_pred/ablation_results.json
  - $FIGURE_DIR/metabolite_ablation_<model>_drop_heatmap.png  (per-day
    balanced-accuracy DROP for each ablated metabolite; one panel per cohort).

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/ablation.py"
    make run ARGS="analysis/2026_06_metabolite_pred/ablation.py --cohort full --days Dy24 Dy30"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/ablation.py
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence

import numpy as np

# Sibling modules imported as top-level (package name starts with a digit; the
# script is run by path, which puts this directory on sys.path). Mirrors run.py.
from cohorts import ALL_DATA_PATH, build_cohort
from cv import run_cv_for_day

# Importing metabolites_train also installs its warnings.filterwarnings("ignore").
from analysis.paper_2026_04.metabolites_train import MODEL_SPECS
from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    REQUIRED_METABOLITES,
)
from pipeline.splits import Splits

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COHORTS: tuple[str, ...] = ("strong-consensus", "full")
_COHORT_ALIASES = {
    "strong": "strong-consensus",
    "strong-consensus": "strong-consensus",
    "full": "full",
}

# Headline baseline = nominal amounts + deltas (nominal_delta in run.py).
BASELINE: dict[str, bool] = {
    "include_growth": True,
    "include_initial": True,
    "normalize_by_size": False,
    "winsorize": False,
}

# Feature-group column-drop ablations: name -> predicate that is True for the
# columns to KEEP (i.e. False for the columns the ablation removes).
_GROUP_KEEP: dict[str, Callable[[str], bool]] = {
    "group:growth": lambda name: "_growth" not in name,
    "group:initial": lambda name: "_initial_concentration" not in name,
}


def _features(ds, day: str, **overrides: bool):
    """(X, y, names, ids) for one day under the baseline config + overrides."""
    cfg = {**BASELINE, **overrides}
    return ds.get_metabolite_features("all", day, **cfg)


def _drop_columns(
    X: np.ndarray, names: Sequence[str], keep: Callable[[str], bool]
) -> tuple[np.ndarray, list[str]]:
    """Return (X, names) with only the columns for which ``keep(name)`` is True.

    Rule 11: dropping columns must never drop organoids — the row count is
    asserted unchanged. Fails loudly if the predicate removes every column.
    """
    X = np.asarray(X)
    keep_idx = [i for i, n in enumerate(names) if keep(n)]
    assert keep_idx, "ablation removed every feature column"
    X_sub = X[:, keep_idx]
    kept_names = [names[i] for i in keep_idx]
    assert X_sub.shape[0] == X.shape[0], (
        f"row count changed on column drop: {X.shape[0]} -> {X_sub.shape[0]}"
    )
    return X_sub, kept_names


def _restrict_rows(
    X: np.ndarray, ids: Sequence[str], keep_ids: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict (X, ids) to ``keep_ids`` in the current row order. Returns (X, ids)."""
    X = np.asarray(X)
    ids = np.asarray(ids)
    mask = np.array([i in keep_ids for i in ids])
    return X[mask], ids[mask]


def metabolite_ablations() -> dict[str, Callable[[str], bool]]:
    """name -> keep-predicate for the 6 leave-one-metabolite-out ablations.

    Metabolite names are mutually non-overlapping substrings of the feature
    column names, so ``met not in name`` cleanly removes exactly that
    metabolite's columns (concentration + initial + growth). Sorted for
    deterministic order (rule 17).
    """
    return {
        f"metabolite:{met}": (lambda name, m=met: m not in name)
        for met in sorted(REQUIRED_METABOLITES)
    }


def _labels_for(ds, ids: Sequence[str]) -> np.ndarray:
    """Binary labels (1=Not Acceptable) aligned to ``ids`` order, from all_data.

    Uses the dataset's organoid_labels() map (rule 3/16: derived from
    all_data.json, not recomputed).
    """
    label_map = ds.organoid_labels()
    return np.array([1 if label_map[i] == "Not Acceptable" else 0 for i in ids])


def _run_size_norm_ablation(ds, day: str, spec, *, n_folds: int, seed: int) -> dict | None:
    """Rebuild with size-normalization and compare vs nominal on common ids.

    Returns a metrics dict, or None if either matrix cannot be cross-validated.
    """
    X_base, _, _, ids_base = _features(ds, day)
    X_sz, _, _, ids_sz = _features(ds, day, normalize_by_size=True)
    common = set(ids_base) & set(ids_sz)
    dropped = len(set(ids_base)) - len(common)
    if dropped:
        logger.warning(
            "  [%s] group:size_norm dropped %d/%d organoids lacking mask area",
            day, dropped, len(set(ids_base)),
        )
    if not common:
        return None

    # y is recovered per matrix by re-fetching aligned to the restricted ids.
    Xb, ids_b = _restrict_rows(X_base, ids_base, common)
    Xs, ids_s = _restrict_rows(X_sz, ids_sz, common)
    yb = _labels_for(ds, ids_b)
    ys = _labels_for(ds, ids_s)

    m_base = run_cv_for_day(spec, Xb, yb, ids_b, n_folds=n_folds, seed=seed)
    m_sz = run_cv_for_day(spec, Xs, ys, ids_s, n_folds=n_folds, seed=seed)
    if m_base is None or m_sz is None:
        return None
    return {
        "balanced_accuracy": m_sz["balanced_accuracy"],
        "baseline_common_balanced_accuracy": m_base["balanced_accuracy"],
        "balacc_gain_from_size_norm": (
            m_sz["balanced_accuracy"] - m_base["balanced_accuracy"]
        ),
        "n": m_sz["n"],
        "n_dropped_vs_baseline": dropped,
        "recall_not_acceptable": m_sz["recall_not_acceptable"],
    }


def run_day(ds, day: str, specs, *, n_folds: int, seed: int) -> dict | None:
    """All ablations for one day. Returns {model_display: {...}} or None.

    None if the baseline itself cannot be cross-validated on this day.
    """
    X, y, names, ids = _features(ds, day)
    n0 = X.shape[0]
    assert len(y) == n0 == len(ids), "baseline X/y/ids row-count mismatch"

    met_keep = metabolite_ablations()
    per_model: dict[str, dict] = {}
    for spec in specs:
        base = run_cv_for_day(spec, X, y, ids, n_folds=n_folds, seed=seed)
        if base is None:
            continue
        base_ba = base["balanced_accuracy"]
        entry: dict = {
            "baseline": {
                "balanced_accuracy": base_ba,
                "recall_not_acceptable": base["recall_not_acceptable"],
                "n": base["n"],
                "n_features": len(names),
                "feature_names": list(names),
            },
            "ablations": {},
        }

        # Column-drop ablations (metabolites + growth/initial groups): row set
        # is conserved (rule 11), so we score on the same y / ids.
        col_keep = {**met_keep, **_GROUP_KEEP}
        for ab_name in sorted(col_keep):
            keep = col_keep[ab_name]
            n_dropped = sum(1 for n in names if not keep(n))
            if n_dropped == 0:
                # Nothing to ablate on this day (e.g. group:growth at Dy03, where
                # growth features do not exist). Skip rather than log a no-op result.
                logger.info("    skip %s: no matching columns on %s", ab_name, day)
                continue
            X_sub, kept = _drop_columns(X, names, keep)
            assert X_sub.shape[0] == n0, "column drop changed organoid count"
            m = run_cv_for_day(spec, X_sub, y, ids, n_folds=n_folds, seed=seed)
            if m is None:
                continue
            entry["ablations"][ab_name] = {
                "balanced_accuracy": m["balanced_accuracy"],
                "balacc_drop": base_ba - m["balanced_accuracy"],
                "recall_not_acceptable": m["recall_not_acceptable"],
                "n": m["n"],
                "n_features_dropped": len(names) - len(kept),
            }

        # Size-normalization is a rebuild (row set may shrink), handled apart.
        sz = _run_size_norm_ablation(ds, day, spec, n_folds=n_folds, seed=seed)
        if sz is not None:
            entry["ablations"]["group:size_norm"] = sz

        per_model[spec.display] = entry
        logger.info(
            "  %-19s %-7s baseline BalAcc=%.4f  (%d ablations)",
            spec.display, day, base_ba, len(entry["ablations"]),
        )
    return per_model or None


def run_cohort(cohort: str, specs, *, days: Sequence[str], n_folds: int, seed: int) -> dict:
    """Run all ablations across all days for one cohort. Returns results dict."""
    ds, counts = build_cohort(cohort, ALL_DATA_PATH)
    n_org = len(ds.organoid_ids)
    logger.info("\n%s\nCohort %s: %d organoids  %s\n%s", "=" * 60, cohort, n_org, counts, "=" * 60)

    ds.apply_splits(
        Splits.from_dict(
            {oid: "all" for oid in ds.organoid_ids},
            name=f"ablation_all_{cohort}",
            provenance=f"single-split CV harness (ablation), cohort={cohort}",
        ),
        strict=True,
    )

    results: dict = {
        "cohort": cohort,
        "n_organoids": n_org,
        "label_counts": counts,
        "baseline_config": BASELINE,
        "n_folds": n_folds,
        "seed": seed,
        "days": {},
    }
    for day in days:
        if day not in ds.days:
            continue
        logger.info("\n----- %s / %s -----", cohort, day)
        day_res = run_day(ds, day, specs, n_folds=n_folds, seed=seed)
        if day_res is not None:
            results["days"][day] = day_res
    return results


def _write_json(all_results: dict) -> None:
    out_dir = ANALYSIS_OUTPUT_DIR / "metabolite_pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_results.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved results to %s", path)


def _drop_matrix(
    cohort_res: dict, model_display: str, days: Sequence[str], metabolites: Sequence[str]
) -> np.ndarray:
    """metabolites x days matrix of balacc_drop for one model (NaN if missing)."""
    mat = np.full((len(metabolites), len(days)), np.nan)
    day_map = cohort_res["days"]
    for j, day in enumerate(days):
        model = day_map.get(day, {}).get(model_display)
        if not model:
            continue
        for i, met in enumerate(metabolites):
            ab = model["ablations"].get(f"metabolite:{met}")
            if ab is not None:
                mat[i, j] = ab["balacc_drop"]
    return mat


def plot_metabolite_drop_heatmap(
    all_results: dict, model_display: str, *, output_path
) -> None:
    """Per-day balanced-accuracy DROP for each ablated metabolite; panel/cohort.

    Diverging colormap centered at 0 (dataviz: polarity => two hues + neutral
    midpoint): red = removing the metabolite HURT (positive drop = it was
    contributing), blue = removing it helped, near-white = no effect.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    cohorts = [c for c in COHORTS if c in all_results]
    metabolites = sorted(REQUIRED_METABOLITES)
    days_present = [d for d in DAY_ORDER if any(d in all_results[c]["days"] for c in cohorts)]
    if not days_present:
        logger.warning("no days to plot; skipping heatmap")
        return

    mats = [_drop_matrix(all_results[c], model_display, days_present, metabolites) for c in cohorts]
    finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mats]) if mats else np.array([0.0])
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 0.05
    vmax = max(vmax, 1e-3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        len(cohorts), 1, figsize=(1.1 * len(days_present) + 3, 2.4 * len(cohorts) + 1),
        squeeze=False,
    )
    for row, (cohort, mat) in enumerate(zip(cohorts, mats)):
        ax = axes[row][0]
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(days_present)))
        ax.set_xticklabels(days_present, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(metabolites)))
        ax.set_yticklabels(metabolites, fontsize=8)
        ax.set_title(f"{cohort}  (n={all_results[cohort]['n_organoids']})", fontsize=10)
        for i in range(len(metabolites)):
            for j in range(len(days_present)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                            color="black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("BalAcc drop (baseline − ablated)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Leave-one-metabolite-out: balanced-accuracy drop by day ({model_display})\n"
        "red = metabolite contributes (removing it hurts); blue = removing it helps",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["strong", "strong-consensus", "full", "all"],
                        default="all")
    parser.add_argument("--days", nargs="+", default=None,
                        help="Specific days (e.g. Dy30 Dy24); default all")
    parser.add_argument("--skip-lgbm", action="store_true")
    parser.add_argument("--skip-lr", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    specs = []
    if not args.skip_lgbm:
        specs.append(MODEL_SPECS["lgbm"])
    if not args.skip_lr:
        specs.append(MODEL_SPECS["logreg"])
    if not specs:
        parser.error("nothing to run: both models skipped")

    cohorts = COHORTS if args.cohort == "all" else (_COHORT_ALIASES[args.cohort],)
    days = args.days if args.days else DAY_ORDER

    all_results: dict = {}
    for cohort in cohorts:
        all_results[cohort] = run_cohort(cohort, specs, days=days, n_folds=args.folds, seed=args.seed)

    _write_json(all_results)

    # Primary model heatmap (LightGBM if present, else the first model run).
    primary = "LightGBM" if not args.skip_lgbm else specs[0].display
    for spec in specs:
        png = FIGURE_DIR / f"metabolite_ablation_{spec.name}_drop_heatmap.png"
        plot_metabolite_drop_heatmap(all_results, spec.display, output_path=png)
    logger.info("Primary model for headline: %s", primary)


if __name__ == "__main__":
    main()
