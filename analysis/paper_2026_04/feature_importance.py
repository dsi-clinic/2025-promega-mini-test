#!/usr/bin/env python3
"""
Reproduce Figure 10: Feature importance from LightGBM across selected days.

Reads saved results from $ANALYSIS_OUTPUT_DIR/metabolites/results.json
(produced by analysis.paper_2026_04.metabolites_train) and creates the
feature importance plot.

Outputs:
  - $ANALYSIS_OUTPUT_DIR/figures/Feature Importance Graph.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.feature_importance"
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, FIGURE_DIR

RESULTS_PATH = ANALYSIS_OUTPUT_DIR / "metabolites" / "results.json"

# Days shown in the paper's feature importance figure
SELECTED_DAYS = ["Dy06", "Dy15", "Dy24", "Dy30"]

# Feature display names
CONCENTRATION_FEATURES = {
    "GlucoseGlo_concentration_uM": "Glucose conc.",
    "GlutamateGlo_concentration_uM": "Glutamate conc.",
    "LactateGlo_concentration_uM": "Lactate conc.",
    "PyruvateGlo_concentration_uM": "Pyruvate conc.",
    "MalateGlo_concentration_uM": "Malate conc.",
    "GlucoseGlo_initial_concentration": "Glucose init.",
    "GlutamateGlo_initial_concentration": "Glutamate init.",
    "LactateGlo_initial_concentration": "Lactate init.",
    "PyruvateGlo_initial_concentration": "Pyruvate init.",
    "MalateGlo_initial_concentration": "Malate init.",
}

GROWTH_FEATURES = {
    "GlucoseGlo_growth": "Glucose Δ",
    "GlutamateGlo_growth": "Glutamate Δ",
    "LactateGlo_growth": "Lactate Δ",
    "PyruvateGlo_growth": "Pyruvate Δ",
    "MalateGlo_growth": "Malate Δ",
}


def get_display_name(feat):
    return CONCENTRATION_FEATURES.get(feat) or GROWTH_FEATURES.get(feat) or feat


def is_growth_feature(feat):
    return feat in GROWTH_FEATURES


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    lgbm_results = results.get("lgbm", {})

    # Collect all features across selected days
    all_features = set()
    day_importances = {}

    for day in SELECTED_DAYS:
        day_result = lgbm_results.get(day)
        if day_result is None:
            print(f"Warning: no LightGBM results for {day}")
            continue
        fi = day_result.get("feature_importance", [])
        imp_dict = {item["feature"]: item["importance"] for item in fi}
        day_importances[day] = imp_dict
        all_features.update(imp_dict.keys())

    if not day_importances:
        print("No feature importance data found.")
        return

    # Normalize importances per day (relative to max)
    for day in day_importances:
        imp = day_importances[day]
        max_imp = max(imp.values()) if imp else 1
        if max_imp > 0:
            day_importances[day] = {k: v / max_imp for k, v in imp.items()}

    # Sort features by average importance across days
    feat_avg = {}
    for feat in all_features:
        vals = [day_importances[d].get(feat, 0) for d in SELECTED_DAYS if d in day_importances]
        feat_avg[feat] = np.mean(vals) if vals else 0
    sorted_features = sorted(feat_avg.keys(), key=lambda f: feat_avg[f], reverse=True)

    # Top 15 features
    top_features = sorted_features[:15]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(top_features))
    width = 0.18
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, day in enumerate(SELECTED_DAYS):
        if day not in day_importances:
            continue
        vals = [day_importances[day].get(f, 0) for f in top_features]
        markers = []
        for f in top_features:
            if is_growth_feature(f):
                markers.append("^")  # triangle for growth
            else:
                markers.append("o")  # circle for concentration

        bars = ax.barh(
            x + i * width, vals, width,
            label=day.replace("Dy0", "Day ").replace("Dy", "Day "),
            color=colors[i], alpha=0.85,
        )

    ax.set_yticks(x + width * 1.5)
    ax.set_yticklabels([get_display_name(f) for f in top_features])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Feature Importance")
    ax.set_title("Feature Importance from LightGBM Metabolite Classifier")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "Feature Importance Graph.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
