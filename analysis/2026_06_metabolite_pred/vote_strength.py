#!/usr/bin/env python3
"""Vote-strength stratification of the full (248) metabolite cohort.

Treats the Dy30 5-rater *regular* survey as a certainty measure. Each organoid's
5 regular votes split into a majority and minority; the split defines a stratum:

  - ``5-0`` — unanimous, the most certain calls.
  - ``4-1`` — one dissenter, less certain.
  - ``3-2`` — near-even, the least certain / most ambiguous calls.

The 5-0 and 4-1 strata together are exactly the ``strong-consensus`` cohort
(>= 4/5 agreement); the 3-2 stratum is the 50 organoids the ``full`` cohort adds
by simple majority. On the current all_data.json the split is 119 / 79 / 50
(= 248), and 119 + 79 = 198 = the strong-consensus size.

Two deliverables (AGENTS.md rule 3/16: everything reads through
``pipeline.data_loader`` accessors off all_data.json — no raw json.load):

1. **Vote-split distribution** across the 248 cohort, broken down by resolved
   label (Acceptable / Not Acceptable): a CSV table + a stacked-bar figure.
2. **Performance by vote strength**: per-day balanced accuracy of the metabolite
   good/bad classifier (nominal+delta headline config, the same per-day nested
   CV as ``run.py``) computed *within* each stratum, from pooled out-of-fold
   predictions. Surfaces whether the model is more accurate on the "easy" 5-0
   organoids than on the "ambiguous" 3-2 ones. Writes a JSON + a per-model
   balanced-accuracy-by-day figure.

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/vote_strength.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/vote_strength.py
"""

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import ALL_DATA_PATH, build_cohort
from cv import run_cv_for_day

from analysis.paper_2026_04.metabolites_train import MODEL_SPECS
from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    LABEL_DAY,
    OrganoidDataset,
    get_survey_vote_counts,
)
from pipeline.splits import Splits

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Ordinal, most-certain -> least-certain. Order is load-bearing (rule 17): the
# table, figures and JSON all iterate this so two runs are byte-identical.
STRATA = ("5-0", "4-1", "3-2")
LABELS = ("Acceptable", "Not Acceptable")

# The strong-consensus cohort = the 5-0 and 4-1 strata (>= 4/5 agreement).
STRONG_CONSENSUS_STRATA = ("5-0", "4-1")

# Headline feature config (nominal + delta), matching run.py / shap_importance.py.
HEADLINE_CONFIG = {
    "key": "nominal_delta",
    "label": "Nominal + delta",
    "normalize_by_size": False,
    "include_growth": True,
    "winsorize": False,
}

# Okabe-Ito colorblind-safe hues, one per stratum, plus a redundant marker /
# linestyle so identity never rests on color alone (dataviz accessibility pass).
_STRATUM_STYLE = {
    "5-0": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "4-1": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "3-2": {"color": "#009E73", "marker": "^", "linestyle": ":"},
}
# Acceptable = good (blue), Not Acceptable = flagged (orange); same safe pair.
_LABEL_COLOR = {"Acceptable": "#0072B2", "Not Acceptable": "#E69F00"}


def vote_strength_stratum(record: dict) -> str:
    """Return the vote-strength stratum ('5-0' / '4-1' / '3-2') for a Dy30 record.

    Reads the 5 regular-image votes via ``get_survey_vote_counts`` and buckets by
    the majority count (max of Acceptable vs Not-Acceptable votes). Raises
    ValueError if the regular votes do not total 5 or the majority is not in
    {3, 4, 5} — the cohort is defined so every organoid has exactly 5, so a
    surprise here is a data-drift bug we want to fail loudly on (rule 15).
    """
    acc, total = get_survey_vote_counts(record)
    if total != 5:
        raise ValueError(f"expected 5 regular votes, got {total}")
    majority = max(acc, total - acc)
    mapping = {5: "5-0", 4: "4-1", 3: "3-2"}
    if majority not in mapping:
        raise ValueError(f"unexpected majority count {majority} (acc={acc}, total={total})")
    return mapping[majority]


def stratum_by_organoid(ds: OrganoidDataset) -> dict[str, str]:
    """Map every organoid in the dataset to its vote-strength stratum.

    Deterministic: iterates organoids in sorted id order (rule 17).
    """
    out: dict[str, str] = {}
    for oid in sorted(ds.organoid_ids):
        rec = ds.get_record(oid, LABEL_DAY)
        if rec is None:
            raise ValueError(f"{oid}: no {LABEL_DAY} record to read votes from")
        out[oid] = vote_strength_stratum(rec)
    return out


def build_distribution(
    ds: OrganoidDataset, strata: dict[str, str]
) -> dict[str, dict[str, int]]:
    """Cross-tabulate stratum x resolved label into counts.

    Returns ``{stratum: {label: count}}`` covering every (stratum, label) pair
    (zero-filled), for every organoid in ``ds``. Asserts the total count equals
    the cohort size — no organoid may be dropped or double-counted (rule 11).
    """
    labels = ds.organoid_labels()
    table = {s: {lab: 0 for lab in LABELS} for s in STRATA}
    for oid in sorted(ds.organoid_ids):
        s = strata[oid]
        lab = labels[oid]
        if lab not in table[s]:
            raise ValueError(f"{oid}: unexpected label {lab!r}")
        table[s][lab] += 1

    total = sum(table[s][lab] for s in STRATA for lab in LABELS)
    assert total == len(ds.organoid_ids), (
        f"distribution total {total} != cohort size {len(ds.organoid_ids)}"
    )
    return table


def write_distribution_csv(table: dict[str, dict[str, int]], path) -> None:
    """Write the stratum x label distribution as a tidy CSV (one row per pair)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vote_strength", "label", "count", "stratum_total"])
        for s in STRATA:
            stratum_total = sum(table[s].values())
            for lab in LABELS:
                w.writerow([s, lab, table[s][lab], stratum_total])
        # A trailing all-cohort total row keeps the file self-checking.
        grand = sum(table[s][lab] for s in STRATA for lab in LABELS)
        w.writerow(["ALL", "ALL", grand, grand])
    logger.info("Saved distribution table to %s", path)


def plot_distribution(table: dict[str, dict[str, int]], path) -> None:
    """Stacked-bar figure: one bar per stratum, segmented by resolved label."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(STRATA))
    fig, ax = plt.subplots(figsize=(8, 5.5))

    bottoms = np.zeros(len(STRATA))
    for lab in LABELS:
        heights = np.array([table[s][lab] for s in STRATA], dtype=float)
        bars = ax.bar(
            x, heights, bottom=bottoms, width=0.62,
            label=lab, color=_LABEL_COLOR[lab], edgecolor="white", linewidth=2,
        )
        for rect, h in zip(bars, heights):
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + h / 2, f"{int(h)}",
                        ha="center", va="center", color="white", fontsize=10,
                        fontweight="bold")
        bottoms += heights

    totals = [sum(table[s].values()) for s in STRATA]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + 2, f"n={tot}", ha="center", va="bottom",
                fontsize=10, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({_stratum_desc(s)})" for s in STRATA])
    ax.set_ylabel("Organoids")
    ax.set_xlabel("Vote strength (majority-minority of 5 regular Dy30 votes)")
    ax.set_title("Vote-split distribution across the full cohort (248)")
    ax.set_ylim(0, max(totals) * 1.15)
    ax.legend(title="Resolved label", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved distribution figure to %s", path)


def _stratum_desc(s: str) -> str:
    return {"5-0": "certain", "4-1": "less certain", "3-2": "ambiguous"}[s]


def _features(ds: OrganoidDataset, day: str, cfg: dict):
    return ds.get_metabolite_features(
        "all", day,
        include_growth=cfg["include_growth"], include_initial=True,
        normalize_by_size=cfg["normalize_by_size"], winsorize=cfg["winsorize"],
    )


def stratified_accuracy(
    ds: OrganoidDataset, strata: dict[str, str], *, specs, days, n_folds, seed
) -> dict:
    """Per-day balanced accuracy within each vote-strength stratum, per model.

    Runs the same per-day nested CV as run.py (headline config) with
    ``return_oof=True`` to pool the exact out-of-fold predictions, then buckets
    the held-out organoids by stratum and scores each bucket. Balanced accuracy
    needs both classes, so a stratum/day with only one class present is recorded
    as ``null`` with the reason logged (rule 15) rather than a misleading number.

    Returns ``{model_display: {day: {stratum: {balanced_accuracy, n, n_pos,
    n_neg}}}}``.
    """
    results = {spec.display: {} for spec in specs}
    for day in days:
        if day not in ds.days:
            continue
        X, y, _names, ids = _features(ds, day, HEADLINE_CONFIG)
        ids = list(ids)
        n0 = len(ids)
        for spec in specs:
            m = run_cv_for_day(spec, X, y, ids, n_folds=n_folds, seed=seed,
                               return_oof=True)
            if m is None:
                continue
            oof_pred = np.asarray(m["oof_pred"])
            oof_ids = list(m["oof_ids"])
            assert oof_ids == ids, f"{day}/{spec.display}: OOF id order drifted"
            assert len(oof_pred) == n0, (
                f"{day}/{spec.display}: {len(oof_pred)} preds for {n0} organoids"
            )
            y_arr = np.asarray(y)

            per_stratum = {}
            covered = 0
            for s in STRATA:
                mask = np.array([strata[oid] == s for oid in ids], dtype=bool)
                covered += int(mask.sum())
                yt, yp = y_arr[mask], oof_pred[mask]
                n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
                entry = {"n": int(mask.sum()), "n_pos": n_pos, "n_neg": n_neg,
                         "balanced_accuracy": None}
                if n_pos > 0 and n_neg > 0:
                    entry["balanced_accuracy"] = round(
                        float(balanced_accuracy_score(yt, yp)), 4
                    )
                else:
                    logger.info(
                        "  [%s] %s %s: single-class bucket (n_pos=%d, n_neg=%d) -> null",
                        spec.display, day, s, n_pos, n_neg,
                    )
                per_stratum[s] = entry
            # Rule 11: every held-out organoid lands in exactly one stratum.
            assert covered == n0, f"{day}/{spec.display}: {covered} bucketed of {n0}"
            results[spec.display][day] = per_stratum
            _log_day(spec.display, day, per_stratum)
    return results


def _log_day(display: str, day: str, per_stratum: dict) -> None:
    parts = []
    for s in STRATA:
        ba = per_stratum[s]["balanced_accuracy"]
        parts.append(f"{s}={ba:.3f}" if ba is not None else f"{s}=--")
    logger.info("  [%s] %-7s  %s", display, day, "  ".join(parts))


def plot_stratified_accuracy(results: dict, path_for) -> None:
    """One balanced-accuracy-by-day figure per model, a line per stratum."""
    import matplotlib.pyplot as plt

    for display, per_day in results.items():
        days = [d for d in DAY_ORDER if d in per_day]
        if not days:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for s in STRATA:
            ys = [per_day[d][s]["balanced_accuracy"] for d in days]
            xs = [i for i, v in enumerate(ys) if v is not None]
            vals = [v for v in ys if v is not None]
            if not xs:
                continue
            st = _STRATUM_STYLE[s]
            ax.plot(xs, vals, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], linewidth=2,
                    label=f"{s} ({_stratum_desc(s)})")
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days, rotation=45)
        ax.set_xlabel("Day")
        ax.set_ylabel("Balanced accuracy (within stratum)")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{display}: balanced accuracy by day, by vote strength "
                     f"(full cohort, {HEADLINE_CONFIG['label']})")
        ax.legend(title="Vote strength", frameon=False)
        fig.tight_layout()
        p = path_for(display)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        logger.info("Saved stratified-accuracy figure to %s", p)


def summarize_hypothesis(results: dict) -> dict:
    """Mean balanced accuracy per stratum per model (days where both are scored).

    Compares 5-0 vs 3-2 to surface the "easy vs ambiguous" hypothesis. Only days
    where *both* the 5-0 and 3-2 buckets have a defined balanced accuracy are
    averaged, so the comparison is like-for-like.
    """
    summary = {}
    for display, per_day in results.items():
        paired = [
            d for d in DAY_ORDER
            if d in per_day
            and per_day[d]["5-0"]["balanced_accuracy"] is not None
            and per_day[d]["3-2"]["balanced_accuracy"] is not None
        ]
        means = {}
        for s in STRATA:
            vals = [per_day[d][s]["balanced_accuracy"] for d in paired]
            means[s] = round(float(np.mean(vals)), 4) if vals else None
        summary[display] = {
            "paired_days": paired,
            "mean_balanced_accuracy": means,
            "five_zero_beats_three_two": (
                means["5-0"] is not None and means["3-2"] is not None
                and means["5-0"] > means["3-2"]
            ),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=None,
                    help="Specific days (default: all in DAY_ORDER)")
    ap.add_argument("--skip-lgbm", action="store_true")
    ap.add_argument("--skip-lr", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    specs = []
    if not args.skip_lgbm:
        specs.append(MODEL_SPECS["lgbm"])
    if not args.skip_lr:
        specs.append(MODEL_SPECS["logreg"])
    if not specs:
        ap.error("nothing to run: both models skipped")

    ds, counts = build_cohort("full", ALL_DATA_PATH)
    n0 = len(ds.organoid_ids)
    logger.info("Full cohort: %d organoids  %s", n0, counts)
    ds.apply_splits(
        Splits.from_dict(
            {oid: "all" for oid in ds.organoid_ids},
            name="vote_strength_all",
            provenance="single-split CV harness, vote_strength.py",
        ),
        strict=True,
    )

    strata = stratum_by_organoid(ds)
    assert len(strata) == n0, f"stratified {len(strata)} of {n0} organoids"

    # -- Deliverable 1: distribution ----------------------------------------
    table = build_distribution(ds, strata)
    logger.info("\nVote-split distribution (stratum -> label -> count):")
    for s in STRATA:
        logger.info("  %-4s (%-12s) total=%3d  %s", s, _stratum_desc(s),
                    sum(table[s].values()), dict(table[s]))
    strong = sum(sum(table[s].values()) for s in STRONG_CONSENSUS_STRATA)
    logger.info("  5-0 + 4-1 = %d (== strong-consensus cohort size)", strong)

    out_dir = ANALYSIS_OUTPUT_DIR / "metabolite_pred"
    write_distribution_csv(table, out_dir / "vote_strength_distribution.csv")
    plot_distribution(table, FIGURE_DIR / "vote_strength_distribution.png")

    # -- Deliverable 2: stratified per-day accuracy -------------------------
    days = args.days if args.days else DAY_ORDER
    acc = stratified_accuracy(ds, strata, specs=specs, days=days,
                              n_folds=args.folds, seed=args.seed)
    hyp = summarize_hypothesis(acc)

    payload = {
        "cohort": "full",
        "config": HEADLINE_CONFIG["key"],
        "seed": args.seed,
        "n_folds": args.folds,
        "distribution": table,
        "per_day_by_model": acc,
        "hypothesis_5_0_vs_3_2": hyp,
    }
    acc_path = out_dir / "vote_strength_accuracy.json"
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(acc_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved stratified accuracy to %s", acc_path)

    plot_stratified_accuracy(
        acc, lambda d: FIGURE_DIR / f"vote_strength_accuracy_{_slug(d)}.png"
    )

    logger.info("\nHypothesis (5-0 'easy' vs 3-2 'ambiguous'):")
    for display, s in hyp.items():
        m = s["mean_balanced_accuracy"]
        logger.info(
            "  %-19s over %d paired days: 5-0=%s  4-1=%s  3-2=%s  -> 5-0>3-2: %s",
            display, len(s["paired_days"]),
            _fmt(m["5-0"]), _fmt(m["4-1"]), _fmt(m["3-2"]),
            s["five_zero_beats_three_two"],
        )


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "--"


def _slug(display: str) -> str:
    return display.lower().replace(" ", "_")


if __name__ == "__main__":
    main()
