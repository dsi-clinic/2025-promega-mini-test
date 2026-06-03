#!/usr/bin/env python3
"""
analyze_misses.py — find organoids that are CONSISTENTLY misclassified across
model variants.

Reads the FP/FN lists already saved in each model's results JSON and counts how
often each test-set organoid is misclassified across the full variant sweep:

    base_effnet            → one variant per `target_day` (~11 variants)
    temporal_ablation_attn → one variant per `max_day`    (~10 variants)
    temporal_ablation_lstm → one variant per `max_day`    (~10 variants)

No model loading, no inference, no retraining. Pure aggregation over the
results JSONs.

Outputs to --output-dir:
    misses_<label>.csv   – one row per test organoid (sorted by miss count desc)
    misses_<label>.png   – bar chart of the top-N most-missed organoids
    stdout summary       – top-N table

Usage
-----
    python analysis/images/cnn_lstm/analyze_misses.py \\
        --run-dir /net/projects2/promega/project_data/model_tests/lstm_runs \\
        --label   expanded \\
        --cohorts-dir data/cohorts \\
        --output-dir /net/projects2/promega/project_data/amanda_test/model_plots

Or compare consistently-missed organoids across cohorts:

    python analysis/images/cnn_lstm/analyze_misses.py \\
        --run-dir /net/projects2/promega/project_data/model_tests/lstm_runs \\
        --compare idor idor_minvotes3 expanded expanded_minvotes3 \\
        --cohorts-dir data/cohorts

CSV columns
-----------
    organoid_id, true_label, n_votes_good, n_votes_total, vote_fraction,
    base_misses, temporal_attn_misses, temporal_lstm_misses,
    total_misses, total_runs, miss_rate

`miss_rate = total_misses / total_runs`. Assumes each organoid in the cohort
test split is evaluated in every variant of every model. For base_effnet,
an organoid without a particular day's image is skipped silently — that
slightly underestimates its miss_rate. Flagged here, not corrected.

Caveat
------
Don't read too much into miss_rate alone. A high-miss organoid + a *borderline*
vote_fraction (close to 0.5) often just means the label is genuinely ambiguous,
not that the model is broken. Combine miss_rate with vote_fraction to triage:
  - high miss_rate + unanimous (vote_fraction 0 or 1)  → real model failure
  - high miss_rate + borderline                        → likely label-noise
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Model registry — mirrors make_run_montage.py for consistency
# ============================================================

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "label":        "base_effnet",
        "subdir":       "base_effnet",
        "results_file": "baseline_results.json",
        "variant_key":  "target_day",
    },
    {
        "label":        "temporal_ablation_attn",
        "subdir":       "temporal_ablation_attn",
        "results_file": "temporal_ablation_results.json",
        "variant_key":  "max_day",
    },
    {
        "label":        "temporal_ablation_lstm",
        "subdir":       "temporal_ablation_lstm",
        "results_file": "temporal_ablation_results_lstm.json",
        "variant_key":  "max_day",
    },
]


# ============================================================
# Loaders
# ============================================================

def load_test_split(cohorts_dir: Path, label: str) -> dict | None:
    """Return {organoid_id: {label, n_votes_good, n_votes_total}} from series test."""
    p = cohorts_dir / label / "series" / "test.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_results_list(run_dir: Path, label: str, spec: dict) -> list | None:
    """Return list-of-variant-results, or None if missing. All 3 models are list-shaped."""
    p = run_dir / label / spec["subdir"] / spec["results_file"]
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] failed to load {p}: {e}")
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]  # single-result model, wrap for uniformity
    return None


# ============================================================
# Aggregation
# ============================================================

def aggregate_misses(run_dir: Path, label: str, cohorts_dir: Path) -> list[dict] | None:
    """
    Build a per-organoid miss summary. Returns a list of rows sorted by
    total_misses descending. Returns None if the cohort test split is missing.
    """
    split = load_test_split(cohorts_dir, label)
    if split is None:
        print(f"  [error] test split not found for cohort '{label}' "
              f"(looked in {cohorts_dir / label / 'series' / 'test.json'})")
        return None

    # Initialize counters for every organoid in the test split.
    per_org_misses: dict[str, dict] = {
        oid: {
            "true_label":   split[oid].get("label", "?"),
            "n_votes_good": split[oid].get("n_votes_good"),
            "n_votes_total": split[oid].get("n_votes_total"),
            "model_misses": defaultdict(int),
            "model_runs":   defaultdict(int),
        }
        for oid in split
    }

    # Walk every variant of every model.
    for spec in MODEL_SPECS:
        results = load_results_list(run_dir, label, spec)
        if results is None:
            print(f"  [skip] {spec['label']}: no results JSON")
            continue
        n_variants = len(results)
        print(f"  [{spec['label']}] {n_variants} variants")

        for variant in results:
            fp_set = set(variant.get("test_false_positives") or [])
            fn_set = set(variant.get("test_false_negatives") or [])
            missed = fp_set | fn_set
            for oid, info in per_org_misses.items():
                info["model_runs"][spec["label"]] += 1
                if oid in missed:
                    info["model_misses"][spec["label"]] += 1

    # Build flat rows.
    rows: list[dict] = []
    for oid, info in per_org_misses.items():
        base_m   = info["model_misses"].get("base_effnet", 0)
        attn_m   = info["model_misses"].get("temporal_ablation_attn", 0)
        lstm_m   = info["model_misses"].get("temporal_ablation_lstm", 0)
        total_m  = base_m + attn_m + lstm_m
        total_r  = sum(info["model_runs"].values())
        n_good   = info["n_votes_good"]
        n_total  = info["n_votes_total"]
        frac     = (n_good / n_total) if (n_good is not None and n_total) else None
        rows.append({
            "organoid_id":         oid,
            "true_label":          info["true_label"],
            "n_votes_good":        n_good,
            "n_votes_total":       n_total,
            "vote_fraction":       frac,
            "base_misses":         base_m,
            "temporal_attn_misses": attn_m,
            "temporal_lstm_misses": lstm_m,
            "total_misses":        total_m,
            "total_runs":          total_r,
            "miss_rate":           (total_m / total_r) if total_r else 0.0,
        })

    rows.sort(key=lambda r: (-r["total_misses"], r["organoid_id"]))
    return rows


# ============================================================
# Output
# ============================================================

def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "organoid_id", "true_label", "n_votes_good", "n_votes_total",
        "vote_fraction",
        "base_misses", "temporal_attn_misses", "temporal_lstm_misses",
        "total_misses", "total_runs", "miss_rate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r = dict(r)
            if r["vote_fraction"] is not None:
                r["vote_fraction"] = round(r["vote_fraction"], 3)
            r["miss_rate"] = round(r["miss_rate"], 3)
            w.writerow(r)


def render_png(rows: list[dict], path: Path, label: str, top_n: int = 20) -> None:
    """Top-N most-missed organoids as a horizontal bar chart, colored by true label."""
    top = [r for r in rows if r["total_misses"] > 0][:top_n]
    if not top:
        print("  [png] no misses to plot")
        return

    fig_h = max(3.0, 0.35 * len(top) + 1.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y = np.arange(len(top))[::-1]

    colors = ["tab:green" if r["true_label"] == "Acceptable" else "tab:red"
              for r in top]
    ax.barh(y, [r["miss_rate"] for r in top], color=colors, edgecolor="black")

    # y-axis labels: short id + (true_label, votes)
    yticks = []
    for r in top:
        tl = "Acc" if r["true_label"] == "Acceptable" else "Bad"
        vf = f"{r['n_votes_good']}/{r['n_votes_total']}" if r["n_votes_total"] else "?"
        yticks.append(f"{r['organoid_id']}  [{tl}, {vf}]")
    ax.set_yticks(y)
    ax.set_yticklabels(yticks, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Miss rate (fraction of all model × variant runs that misclassified)")
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="tab:green", edgecolor="black", label="Acceptable (true label)"),
            Patch(facecolor="tab:red",   edgecolor="black", label="Not Acceptable (true label)"),
        ],
        loc="lower right", fontsize=9,
    )
    ax.set_title(f"Consistently misclassified organoids — {label} (top {len(top)})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict], label: str, top_n: int = 15) -> None:
    print(f"\n=== consistently misclassified — cohort '{label}' (top {top_n}) ===")
    print(f"{'organoid_id':35} {'true':>5} {'votes':>6} {'base':>5} "
          f"{'attn':>5} {'lstm':>5} {'tot':>5} {'/of':>4}  {'rate':>5}")
    for r in rows[:top_n]:
        if r["total_misses"] == 0:
            break
        tl = "Acc" if r["true_label"] == "Acceptable" else "Bad"
        vf = f"{r['n_votes_good']}/{r['n_votes_total']}" if r["n_votes_total"] else "?"
        print(f"{r['organoid_id']:35} {tl:>5} {vf:>6} "
              f"{r['base_misses']:>5} {r['temporal_attn_misses']:>5} "
              f"{r['temporal_lstm_misses']:>5} {r['total_misses']:>5} "
              f"{r['total_runs']:>4}  {r['miss_rate']:>5.2f}")
    n_perfect = sum(1 for r in rows if r["total_misses"] == 0)
    print(f"\n({n_perfect}/{len(rows)} organoids never misclassified.)")


# ============================================================
# Comparison mode (across cohorts)
# ============================================================

def render_comparison(run_dir: Path, labels: list[str], cohorts_dir: Path,
                      output_dir: Path, top_n: int = 15) -> Path:
    """
    For each cohort, find the top-N most-missed organoids; report which
    organoid_ids appear in multiple cohorts' top-N lists.
    """
    per_cohort_tops: dict[str, list[dict]] = {}
    for lab in labels:
        rows = aggregate_misses(run_dir, lab, cohorts_dir) or []
        per_cohort_tops[lab] = [r for r in rows if r["total_misses"] > 0][:top_n]

    # Cross-tabulate: which organoid_ids show up in multiple cohorts' top-N
    cross_count: dict[str, set[str]] = defaultdict(set)
    for lab, top in per_cohort_tops.items():
        for r in top:
            cross_count[r["organoid_id"]].add(lab)

    repeat_offenders = sorted(
        [(oid, cohorts) for oid, cohorts in cross_count.items() if len(cohorts) > 1],
        key=lambda x: -len(x[1]),
    )

    print("\n=== cross-cohort repeat offenders ===")
    if not repeat_offenders:
        print("  (no organoid in the top-N of more than one cohort)")
    else:
        print(f"{'organoid_id':35}  {'cohorts':<40}")
        for oid, cohorts in repeat_offenders:
            print(f"{oid:35}  {sorted(cohorts)}")

    # Save a CSV listing the union of top-Ns with cohort flags
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"misses_compare_{'_vs_'.join(labels)}.csv"
    all_ids = sorted(cross_count.keys())
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["organoid_id", "n_cohorts_in_top_n", *labels])
        for oid in all_ids:
            cohorts = cross_count[oid]
            w.writerow([oid, len(cohorts), *[("X" if lab in cohorts else "") for lab in labels]])
    print(f"\n[wrote] {out_path}")
    return out_path


# ============================================================
# Entrypoint
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Top-level lstm_runs directory.")
    p.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"),
                   help="Root containing data/cohorts/<label>/series/test.json")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/net/projects2/promega/project_data/amanda_test/model_plots"),
                   help="Where misses_<label>.csv and .png are written.")
    p.add_argument("--top-n", type=int, default=20,
                   help="How many top-missed organoids to plot/print (default 20).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--label", type=str, help="Cohort label to analyze.")
    g.add_argument("--compare", nargs="+",
                   help="Two or more cohort labels for repeat-offender analysis.")
    args = p.parse_args()

    if args.label:
        print(f"[analyze_misses] cohort = {args.label}")
        rows = aggregate_misses(args.run_dir, args.label, args.cohorts_dir)
        if rows is None:
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / f"misses_{args.label}.csv"
        png_path = args.output_dir / f"misses_{args.label}.png"
        write_csv(rows, csv_path)
        render_png(rows, png_path, args.label, top_n=args.top_n)
        print_summary(rows, args.label, top_n=args.top_n)
        print(f"\n[wrote] {csv_path}")
        print(f"[wrote] {png_path}")
    else:
        if len(args.compare) < 2:
            p.error("--compare requires at least 2 labels")
        render_comparison(args.run_dir, args.compare, args.cohorts_dir,
                          args.output_dir, top_n=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
