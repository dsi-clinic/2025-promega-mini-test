#!/usr/bin/env python3
"""
Compare balanced accuracy across combined, image-only, and metabolite-only models.

Reads results from:
  - analysis/combined_model/outputs/adaptive_multimodal/results.json
  - analysis_output/images/perday_results.json
  - analysis_output/metabolites/results.json

Outputs:
  analysis/combined_model/outputs/model_comparison/balanced_accuracy_comparison.png
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DAY_ORDER = [
    "Dy03", "Dy06", "Dy08", "Dy10", "Dy13",
    "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30",
]

ROOT = Path(__file__).resolve().parents[2]


def day_to_int(day):
    """Convert a day string like 'Dy03' or 'Dy20_5' to a numeric value."""
    if day == "Dy20_5":
        return 20.5
    return int(day.replace("Dy", ""))


def load_combined_results(path):
    """Load and parse combined model results from results.json.

    Args:
        path: Path to the combined model results.json file.

    Returns:
        DataFrame with columns: day, x, bal_acc, std.
    """
    with open(path) as f:
        combined_json = json.load(f)

    combined = combined_json["aggregated"]
    rows = []

    for day in DAY_ORDER:
        if day in combined:
            rows.append({
                "day": day,
                "x": day_to_int(day),
                "bal_acc": combined[day]["bal_acc_mean"],
                "std": combined[day]["bal_acc_std"],
            })

    return pd.DataFrame(rows)


def load_image_results(path):
    """Load and parse image-only model results.

    Args:
        path: Path to the image model perday_results.json file.

    Returns:
        DataFrame with columns: day, x, bal_acc.
    """
    with open(path) as f:
        image_json = json.load(f)

    rows = []

    for day in DAY_ORDER:
        if day in image_json:
            rows.append({
                "day": day,
                "x": day_to_int(day),
                "bal_acc": image_json[day]["balanced_accuracy"],
            })

    return pd.DataFrame(rows)


def load_metabolite_results(path):
    """Load and parse metabolite-only model results, using LightGBM if available.

    Args:
        path: Path to the metabolite model results.json file.

    Returns:
        DataFrame with columns: day, x, bal_acc.
    """
    with open(path) as f:
        met_json = json.load(f)

    met_results = met_json["lgbm"] if "lgbm" in met_json else met_json
    rows = []

    for day in DAY_ORDER:
        if day in met_results:
            rows.append({
                "day": day,
                "x": day_to_int(day),
                "bal_acc": met_results[day]["balanced_accuracy"],
            })

    return pd.DataFrame(rows)


def plot_comparison(combined_df, image_df, met_df, outpath):
    """Plot balanced accuracy curves for all three models and save to disk.

    Args:
        combined_df: DataFrame with combined model results (includes std column).
        image_df: DataFrame with image-only model results.
        met_df: DataFrame with metabolite-only model results.
        outpath: Path to save the output figure.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    if not combined_df.empty:
        ax.plot(
            combined_df["x"],
            combined_df["bal_acc"],
            marker="o",
            linewidth=2.5,
            label="Combined Model",
        )
        ax.fill_between(
            combined_df["x"],
            combined_df["bal_acc"] - combined_df["std"],
            combined_df["bal_acc"] + combined_df["std"],
            alpha=0.15,
        )

    if not image_df.empty:
        ax.plot(
            image_df["x"],
            image_df["bal_acc"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Image Only",
        )

    if not met_df.empty:
        ax.plot(
            met_df["x"],
            met_df["bal_acc"],
            marker="^",
            linestyle=":",
            linewidth=2,
            label="Metabolite Only / LightGBM",
        )

    ax.axvspan(19, 31, alpha=0.08)
    ax.axhline(0.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Development Day")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy Comparison Across Models")

    ax.set_xticks([day_to_int(d) for d in DAY_ORDER])
    ax.set_xticklabels(DAY_ORDER, rotation=45)

    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)

    print(f"Saved figure to: {outpath}")


def main():
    """Load results from all three models and generate comparison plot."""
    combined_path = ROOT / "analysis/combined_model/outputs/adaptive_multimodal/results.json"
    image_path = ROOT / "analysis_output/images/perday_results.json"
    metabolite_path = ROOT / "analysis_output/metabolites/results.json"

    outdir = ROOT / "analysis/combined_model/outputs/model_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "balanced_accuracy_comparison.png"

    combined_df = load_combined_results(combined_path)
    image_df = load_image_results(image_path)
    met_df = load_metabolite_results(metabolite_path)

    plot_comparison(combined_df, image_df, met_df, outpath)


if __name__ == "__main__":
    main()
