#!/usr/bin/env python3
"""
Feature importance changes over time: line chart across selected days.

Reads saved results from $ANALYSIS_OUTPUT_DIR/metabolites/results.json
(produced by analysis.paper_2026_04.metabolites_train) and plots normalized
feature importance as a polyline for each of the 12 metabolite features
(6 concentrations + 6 growth/delta) across selected days.

Outputs:
  - $ANALYSIS_OUTPUT_DIR/figures/Feature_Importance_Over_Time.png

Usage:
    python -m analysis.paper_2026_04.feature_importance_over_time
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, FIGURE_DIR

RESULTS_PATH = ANALYSIS_OUTPUT_DIR / "metabolites" / "results.json"

SELECTED_DAYS = ["Dy06", "Dy17", "Dy24", "Dy30"]

CONCENTRATION_FEATURES = {
    "GlucoseGlo_concentration_uM": "Glucose conc.",
    "GlutamateGlo_concentration_uM": "Glutamate conc.",
    "LactateGlo_concentration_uM": "Lactate conc.",
    "PyruvateGlo_concentration_uM": "Pyruvate conc.",
    "BCAAGlo_concentration_uM": "BCAA conc.",
    "MalateGlo_concentration_uM": "Malate conc.",
}

GROWTH_FEATURES = {
    "GlucoseGlo_growth": "Glucose Δ",
    "GlutamateGlo_growth": "Glutamate Δ",
    "LactateGlo_growth": "Lactate Δ",
    "PyruvateGlo_growth": "Pyruvate Δ",
    "BCAAGlo_growth": "BCAA Δ",
    "MalateGlo_growth": "Malate Δ",
}


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    lgbm_results = results.get("lgbm", {})

    day_importances = {}
    for day in SELECTED_DAYS:
        day_result = lgbm_results.get(day)
        if day_result is None:
            print(f"Warning: no LightGBM results for {day}")
            continue
        fi = day_result.get("feature_importance", [])
        imp_dict = {item["feature"]: item["importance"] for item in fi}
        max_imp = max(imp_dict.values()) if imp_dict else 1
        day_importances[day] = {k: v / max_imp for k, v in imp_dict.items()}

    days_available = [d for d in SELECTED_DAYS if d in day_importances]
    x = np.arange(len(days_available))
    x_labels = [d.replace("Dy0", "Day ").replace("Dy", "Day ") for d in days_available]

    fig, ax = plt.subplots(figsize=(11, 7))

    conc_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    growth_colors = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]

    for i, (feat, label) in enumerate(CONCENTRATION_FEATURES.items()):
        vals = [day_importances[d].get(feat, 0) for d in days_available]
        ax.plot(x, vals, "o-", label=label, color=conc_colors[i], linewidth=2, markersize=6)

    for i, (feat, label) in enumerate(GROWTH_FEATURES.items()):
        vals = [day_importances[d].get(feat, 0) for d in days_available]
        ax.plot(x, vals, "s--", label=label, color=growth_colors[i], linewidth=1.5, markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Normalized Feature Importance")
    ax.set_xlabel("Day")
    ax.set_title("Feature Importance Changes Over Time (LightGBM)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "Feature_Importance_Over_Time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
