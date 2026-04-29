#!/usr/bin/env python3
"""
Reproduce Figure 5: Metabolite concentration boxplots.

Outputs:
  - analysis/outputs/figures/metabolite_concentration_boxplot.png

Usage:
    make run ARGS="-m analysis.metabolite_boxplot"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline.data_loader import CONDITIONAL_METABOLITES, FIGURE_DIR, OrganoidDataset, get_day_int_floor

ALL_DATA_PATH = "data/all_data.json"
SPLITS_CSV = "data/2026_winter_student_splits.csv"
OUTPUT_DIR = FIGURE_DIR

METABOLITE_NAMES = [
    "GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo", "MalateGlo"
]
DISPLAY_NAMES = {
    "GlucoseGlo": "Glucose",
    "GlutamateGlo": "Glutamate",
    "LactateGlo": "Lactate",
    "PyruvateGlo": "Pyruvate",
    "MalateGlo": "Malate",
}


def main():
    ds = OrganoidDataset(ALL_DATA_PATH, splits_csv=SPLITS_CSV)

    # Collect all concentration values across all organoids and days
    rows = []
    for org_id in ds.organoid_ids:
        info = ds._organoids[org_id]
        for day, rec in info["records"].items():
            day_num = get_day_int_floor(day)
            mets = rec.get("metabolite", {})
            for m in METABOLITE_NAMES:
                # Apply conditional metabolite filtering (e.g. MalateGlo only days > 10)
                if m in CONDITIONAL_METABOLITES:
                    if day_num is None or not CONDITIONAL_METABOLITES[m](day_num):
                        continue
                if m in mets:
                    conc = mets[m].get("concentration_uM")
                    if conc is not None:
                        rows.append({
                            "Metabolite": DISPLAY_NAMES[m],
                            "Concentration (μM)": conc,
                        })

    df = pd.DataFrame(rows)

    # Create boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    order = [DISPLAY_NAMES[m] for m in METABOLITE_NAMES]
    sns.boxplot(
        data=df,
        x="Metabolite",
        y="Concentration (μM)",
        order=order,
        ax=ax,
        palette="Set2",
        fliersize=2,
    )
    ax.set_title("Metabolite Concentrations Across Organoid Samples")
    ax.set_xlabel("")
    ax.set_ylabel("Concentration (μM)")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "metabolite_concentration_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
