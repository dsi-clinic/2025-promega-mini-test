#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DAY_ORDER = [
    "Dy03", "Dy06", "Dy08", "Dy10", "Dy13",
    "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30",
]


def day_to_int(day):
    if day == "Dy20_5":
        return 20.5
    return int(day.replace("Dy", ""))


ROOT = Path("/home/feng27/2025-promega-mini-test")

combined_path = ROOT / "analysis/combined_model/outputs/adaptive_cv_v2_efficientnet/results.json"
image_path = ROOT / "analysis_output/images/perday_results.json"
metabolite_path = ROOT / "analysis_output/metabolites/results.json"

outdir = ROOT / "analysis/combined_model/outputs/model_comparison"
outdir.mkdir(parents=True, exist_ok=True)

outpath = outdir / "balanced_accuracy_comparison.png"


# ----------------------------
# Combined model
# ----------------------------
with open(combined_path) as f:
    combined_json = json.load(f)

combined = combined_json["aggregated"]

combined_rows = []
for day in DAY_ORDER:
    if day in combined:
        combined_rows.append({
            "day": day,
            "x": day_to_int(day),
            "bal_acc": combined[day]["bal_acc_mean"],
            "std": combined[day]["bal_acc_std"],
        })

combined_df = pd.DataFrame(combined_rows)


# ----------------------------
# Image model
# ----------------------------
with open(image_path) as f:
    image_json = json.load(f)

image_rows = []
for day in DAY_ORDER:
    if day in image_json:
        image_rows.append({
            "day": day,
            "x": day_to_int(day),
            "bal_acc": image_json[day]["balanced_accuracy"],
        })

image_df = pd.DataFrame(image_rows)


# ----------------------------
# Metabolite model
# ----------------------------
with open(metabolite_path) as f:
    met_json = json.load(f)

# Use LightGBM as metabolite model
met_results = met_json["lgbm"] if "lgbm" in met_json else met_json

met_rows = []
for day in DAY_ORDER:
    if day in met_results:
        met_rows.append({
            "day": day,
            "x": day_to_int(day),
            "bal_acc": met_results[day]["balanced_accuracy"],
        })

met_df = pd.DataFrame(met_rows)


# ----------------------------
# Plot
# ----------------------------
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
