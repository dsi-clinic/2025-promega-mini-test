"""
Generate Figure 7: Per-Day vs Time-Series accuracy by day.

Two sources supported:
  - run_per_day_study outputs: per_day_study/{per_day,effnet_ts}/day_*/results.json
    (reports balanced_acc at threshold=0.5)
  - train_model_deep_ensemble outputs: per_day_study/efficientnet_ensemble/final_test_summary.json
    (reports plain test accuracy)

Outputs: analysis/images/classifier/fig7_perday_vs_timeseries.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

STUDY_DIR = Path(__file__).parent / "per_day_study"
OUT_PATH = Path(__file__).parent / "fig7_perday_vs_timeseries.png"

DAYS = [6, 8, 10, 13, 15, 17, 20.5, 24, 26, 28, 30]

DAY_STR_MAP = {
    6: "Dy6", 8: "Dy8", 10: "Dy10", 13: "Dy13", 15: "Dy15",
    17: "Dy17", 20.5: "Dy20_5", 24: "Dy24", 26: "Dy26", 28: "Dy28", 30: "Dy30"
}


def load_bal_acc(model_type):
    """Load test balanced_acc at threshold=0.5 for each day (run_per_day_study format)."""
    results = {}
    model_dir = STUDY_DIR / model_type
    for day in DAYS:
        day_str = str(day).replace(".", "_")
        candidates = [f"day_{int(day)}_0", f"day_{day_str}"] if day == int(day) else [f"day_{day_str}", f"day_{day_str}_0"]
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


def load_ensemble_acc():
    """Load test accuracy from train_model_deep_ensemble final_test_summary.json."""
    summary_file = STUDY_DIR / "efficientnet_ensemble" / "final_test_summary.json"
    if not summary_file.exists():
        return {}
    with open(summary_file) as f:
        summary = json.load(f)
    per_day_acc = summary.get("per_day_test_accuracy", {})
    results = {}
    for day in DAYS:
        key = DAY_STR_MAP.get(day)
        if key and key in per_day_acc:
            results[day] = per_day_acc[key]
    return results


def main():
    # Prefer ensemble (train_model_deep_ensemble) for per-day; fall back to run_per_day_study
    ensemble = load_ensemble_acc()
    per_day_study = load_bal_acc("per_day")
    per_day = ensemble if ensemble else per_day_study
    per_day_label = "Per-Day (accuracy)" if ensemble else "Per-Day (balanced acc)"

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
            markersize=6, label=per_day_label)
    ax.plot(days_common, ts_vals, "s--", color="firebrick", linewidth=2,
            markersize=6, label="Time Series (balanced acc)")

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
    ax.set_ylabel("Accuracy", fontsize=12)
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
