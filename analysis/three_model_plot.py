#!/usr/bin/env python3
"""
Reproduce Figure 11: Three-model comparison (image, metabolite, combined).

Reads saved results from:
  - analysis/outputs/images/perday_results.json
  - analysis/outputs/metabolites/results.json
  - analysis/outputs/combined/results.json (if available)

Outputs:
  - analysis/outputs/figures/three_model.png

Usage:
    make run ARGS="-m analysis.three_model_plot"
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from analysis.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER, FIGURE_DIR


def load_results(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    img_results = load_results(ANALYSIS_OUTPUT_DIR / "images" / "perday_results.json")
    met_results = load_results(ANALYSIS_OUTPUT_DIR / "metabolites" / "results.json")
    combined_results = load_results(ANALYSIS_OUTPUT_DIR / "combined" / "results.json")

    if not img_results and not met_results:
        print("No results found.")
        return

    days = []
    img_ba = []
    lgbm_ba = []
    combined_ba = []

    lgbm = met_results.get("lgbm", {}) if met_results else {}

    for day in DAY_ORDER:
        has_img = img_results and day in img_results
        has_met = day in lgbm
        if has_img or has_met:
            days.append(day)
            img_ba.append(img_results[day]["balanced_accuracy"] if has_img else None)
            lgbm_ba.append(lgbm[day]["balanced_accuracy"] if has_met else None)
            if combined_results and day in combined_results:
                combined_ba.append(combined_results[day]["balanced_accuracy"])
            else:
                combined_ba.append(None)

    x = range(len(days))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot available series
    if any(v is not None for v in img_ba):
        valid = [(i, v) for i, v in enumerate(img_ba) if v is not None]
        ax.plot([v[0] for v in valid], [v[1] for v in valid],
                "o-", label="Image (EfficientNet)", color="#1f77b4", linewidth=2)

    if any(v is not None for v in lgbm_ba):
        valid = [(i, v) for i, v in enumerate(lgbm_ba) if v is not None]
        ax.plot([v[0] for v in valid], [v[1] for v in valid],
                "s-", label="Metabolite (LightGBM)", color="#ff7f0e", linewidth=2)

    if any(v is not None for v in combined_ba):
        valid = [(i, v) for i, v in enumerate(combined_ba) if v is not None]
        ax.plot([v[0] for v in valid], [v[1] for v in valid],
                "^-", label="Combined", color="#2ca02c", linewidth=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(days, rotation=45)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Day")
    ax.set_title("Three-Model Comparison: Balanced Accuracy by Day")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "three_model.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
