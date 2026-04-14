"""
Generate Figure 7: Per-Day vs Time-Series balanced accuracy by day.
Reads results from analysis/images/classifier/per_day_study/{per_day,effnet_ts}/day_*/results.json
Outputs: analysis/images/classifier/fig7_perday_vs_timeseries.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = Path(__file__).parent / "per_day_study"
OUT_PATH = Path(__file__).parent / "fig7_perday_vs_timeseries.png"

DAYS = [6, 8, 10, 13, 15, 17, 20.5, 24, 26, 28, 30]


def load_bal_acc(model_type):
    """Load test balanced_acc at threshold=0.5 for each day."""
    results = {}
    model_dir = STUDY_DIR / model_type
    for day in DAYS:
        day_str = str(day).replace(".", "_")
        # directory names like day_6_0, day_20_5
        if day == int(day):
            candidates = [f"day_{int(day)}_0", f"day_{day_str}"]
        else:
            candidates = [f"day_{day_str}", f"day_{day_str}_0"]
        for cand in candidates:
            result_file = model_dir / cand / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                bal_acc = data.get("test_at_0.5", {}).get("balanced_acc")
                if bal_acc is not None:
                    results[day] = bal_acc
                break
    return results


def main():
    per_day = load_bal_acc("per_day")
    effnet_ts = load_bal_acc("effnet_ts")

    days_common = sorted(set(per_day) & set(effnet_ts))
    if not days_common:
        print("ERROR: No results found. Check that both per_day and effnet_ts runs completed.")
        return

    pd_vals = [per_day[d] for d in days_common]
    ts_vals = [effnet_ts[d] for d in days_common]
    diffs = [p - t for p, t in zip(pd_vals, ts_vals)]

    print("Day | Per-Day | TimeSeries | Diff")
    for d, p, t, diff in zip(days_common, pd_vals, ts_vals, diffs):
        print(f"{d:5.1f} | {p:.4f}  | {t:.4f}     | {diff:+.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(days_common, pd_vals, ts_vals, alpha=0.15, color="steelblue")

    ax.plot(days_common, pd_vals, "o-", color="steelblue", linewidth=2,
            markersize=6, label="Per-Day")
    ax.plot(days_common, ts_vals, "s--", color="firebrick", linewidth=2,
            markersize=6, label="Time Series")

    for d, p, t, diff in zip(days_common, pd_vals, ts_vals, diffs):
        ax.annotate(f"{p:.2f}", (d, p), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="steelblue")
        ax.annotate(f"{t:.2f}", (d, t), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="firebrick")
        mid = (p + t) / 2
        ax.annotate(f"{diff:+.2f}", (d, mid), textcoords="offset points",
                    xytext=(8, 0), ha="left", fontsize=7, color="gray")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.50)")
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (threshold = 0.5)", fontsize=12)
    ax.set_title("Per-Day vs. Time Series", fontsize=14, fontweight="bold")
    ax.set_xticks(days_common)
    ax.set_xticklabels([str(d) for d in days_common])
    ax.legend(fontsize=10)
    ax.set_ylim(0.35, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
