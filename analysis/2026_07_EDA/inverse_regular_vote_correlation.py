#!/usr/bin/env python3
"""Regular vs inverted-image survey-vote agreement.

Every re-shown organoid is evaluated twice at Dy30: once on the regular image
and once on a horizontally/vertically inverted ("INV") copy of the same image.
The merge step keeps these as separate buckets (``regular_votes`` /
``inverted_votes``) and the consensus label is decided by the regular bucket
alone (see ``surveys_mapper.compute_survey_majority``). This script measures how
much the two passes actually agree — i.e. does inverting the image change how
reviewers vote?

For every organoid carrying BOTH buckets it pairs the per-bucket "Acceptable"
counts (each out of 5) and reports:
  - Pearson and Spearman correlation of regular-acc vs inverted-acc
  - mean absolute vote difference
  - consensus agreement (do both buckets reach the same >=MIN_VOTES label?)

Outputs:
  - Console: correlation + agreement summary
  - $ANALYSIS_OUTPUT_DIR/figures/inverse_regular_vote_pairs.csv
  - $ANALYSIS_OUTPUT_DIR/figures/inverse_regular_vote_correlation.png

Usage (package name starts with a digit, so run by path, not ``-m``):
    make run ARGS="analysis/2026_07_EDA/inverse_regular_vote_correlation.py"
    # or:
    PYTHONPATH=. python analysis/2026_07_EDA/inverse_regular_vote_correlation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from pipeline.data_loader import (
    FIGURE_DIR,
    LABEL_DAY,
    MIN_VOTES,
    iter_organoid_records,
)

ALL_DATA_PATH = Path("data/all_data.json")
OUTPUT_DIR = FIGURE_DIR


def _bucket_consensus(acc: int, total: int) -> str:
    """Canonical >=MIN_VOTES consensus for a single bucket."""
    nacc = total - acc
    if acc >= MIN_VOTES:
        return "Acceptable"
    if nacc >= MIN_VOTES:
        return "Not Acceptable"
    return "no consensus"


def _vote_pairs(all_data_path: Path) -> pd.DataFrame:
    """One row per organoid that carries BOTH regular and inverted Dy30 votes."""
    rows = []
    for oid, recs, batch in iter_organoid_records(all_data_path):
        rec = recs.get(LABEL_DAY)
        label = (rec or {}).get("label") or {}
        reg = label.get("regular_votes") or {}
        inv = label.get("inverted_votes") or {}
        if not (reg and inv):
            continue
        reg_acc = int(reg.get("Acceptable", 0))
        reg_total = sum(int(v) for v in reg.values())
        inv_acc = int(inv.get("Acceptable", 0))
        inv_total = sum(int(v) for v in inv.values())
        rows.append({
            "organoid_id": oid,
            "batch": batch,
            "reg_acc": reg_acc,
            "reg_total": reg_total,
            "inv_acc": inv_acc,
            "inv_total": inv_total,
            "abs_diff": abs(reg_acc - inv_acc),
            "reg_consensus": _bucket_consensus(reg_acc, reg_total),
            "inv_consensus": _bucket_consensus(inv_acc, inv_total),
        })
    df = pd.DataFrame(rows).sort_values(["batch", "organoid_id"]).reset_index(drop=True)
    df["consensus_agree"] = df["reg_consensus"] == df["inv_consensus"]
    return df


def _scatter(df: pd.DataFrame, pearson_r: float, out_path: Path) -> None:
    """Scatter of regular vs inverted Acceptable counts (jittered, y=x reference)."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    # Count coincident points so identical pairs are visible as larger markers.
    sizes = df.groupby(["reg_acc", "inv_acc"]).size().reset_index(name="n")
    ax.scatter(sizes["reg_acc"], sizes["inv_acc"], s=sizes["n"] * 80,
               alpha=0.6, edgecolor="black", linewidth=0.5, color="#4C72B0")
    for _, r in sizes.iterrows():
        ax.annotate(str(r["n"]), (r["reg_acc"], r["inv_acc"]),
                    ha="center", va="center", fontsize=8, color="white")
    lim = max(df["reg_total"].max(), df["inv_total"].max())
    ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(-0.5, lim + 0.5)
    ax.set_ylim(-0.5, lim + 0.5)
    ax.set_xlabel("Regular-image 'Acceptable' votes (of 5)")
    ax.set_ylabel("Inverted-image 'Acceptable' votes (of 5)")
    ax.set_title(f"Regular vs inverted vote agreement\n"
                 f"N={len(df)} organoids, Pearson r={pearson_r:.3f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = _vote_pairs(ALL_DATA_PATH)
    n = len(df)

    pearson_r, pearson_p = pearsonr(df["reg_acc"], df["inv_acc"])
    spearman_r, spearman_p = spearmanr(df["reg_acc"], df["inv_acc"])
    mean_abs_diff = df["abs_diff"].mean()
    consensus_agree = int(df["consensus_agree"].sum())

    print("=== Regular vs inverted-image vote agreement ===\n")
    print(f"Organoids with both buckets : {n}")
    print(f"  by batch                  : "
          f"{df['batch'].value_counts().sort_index().to_dict()}")
    print(f"Pearson  r (reg_acc, inv_acc): {pearson_r:.3f}  (p={pearson_p:.2g})")
    print(f"Spearman r (reg_acc, inv_acc): {spearman_r:.3f}  (p={spearman_p:.2g})")
    print(f"Mean |reg_acc - inv_acc|     : {mean_abs_diff:.2f} votes")
    print(f"Same >=MIN_VOTES consensus   : {consensus_agree}/{n} "
          f"({100 * consensus_agree / n:.1f}%)")

    print("\nPer-organoid pairs:")
    print(df[["organoid_id", "batch", "reg_acc", "inv_acc",
              "abs_diff", "reg_consensus", "inv_consensus"]].to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "inverse_regular_vote_pairs.csv"
    png_path = OUTPUT_DIR / "inverse_regular_vote_correlation.png"
    df.to_csv(csv_path, index=False)
    _scatter(df, pearson_r, png_path)
    print(f"\nSaved to {csv_path}\n        {png_path}")


if __name__ == "__main__":
    main()
