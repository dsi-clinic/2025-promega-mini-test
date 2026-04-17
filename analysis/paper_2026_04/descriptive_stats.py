#!/usr/bin/env python3
"""
Reproduce dataset description numbers and metabolite summary table from the paper.

Outputs:
  - Console: dataset counts, label distribution, metabolite summary stats
  - analysis/outputs/figures/metabolite_summary_table.csv

Usage:
    make run ARGS="-m analysis.descriptive_stats"
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data_loader import FIGURE_DIR

ALL_DATA_PATH = Path("data/all_data.json")
SPLITS_CSV = Path("data/2026_winter_student_splits.csv")
OUTPUT_DIR = FIGURE_DIR

METABOLITE_NAMES = [
    "GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo", "MalateGlo"
]
DISPLAY_NAMES = {
    "GlucoseGlo": "Glucose concentration (μM)",
    "GlutamateGlo": "Glutamate concentration (μM)",
    "LactateGlo": "Lactate concentration (μM)",
    "PyruvateGlo": "Pyruvate concentration (μM)",
    "MalateGlo": "Malate concentration (μM)",
}


def extract_organoid_id(key: str) -> str:
    m = re.match(r"^(.*)\s+Dy\d+\s+(.*)$", key)
    return f"{m.group(1)} {m.group(2)}" if m else key


def main():
    with open(ALL_DATA_PATH) as f:
        all_data = json.load(f)

    print(f"Total records in all_data.json: {len(all_data)}")

    # Unique organoids
    org_ids = set()
    for key in all_data:
        org_ids.add(extract_organoid_id(key))
    print(f"Unique organoids (all batches): {len(org_ids)}")

    # Days
    days = sorted(set(v.get("dayID", "") for v in all_data.values()))
    print(f"Days: {days}")

    # Records with labels (Dy28/Dy30 with survey)
    labeled_records = 0
    labeled_days = set()
    for k, v in all_data.items():
        if "survey" in v and v["survey"].get("evaluations"):
            labeled_records += 1
            labeled_days.add(v.get("dayID"))
    print(f"Records with survey evaluations: {labeled_records}")
    print(f"Survey days: {sorted(labeled_days)}")

    # --- Load splits dataset for label distribution ---
    from pipeline.data_loader import OrganoidDataset
    ds = OrganoidDataset(str(ALL_DATA_PATH), splits_csv=str(SPLITS_CSV))
    print(f"\n--- Filtered dataset (paper config) ---")
    print(ds.summary())

    total_organoids = len(ds.organoid_ids)
    labels = Counter()
    for org_id in ds.organoid_ids:
        info = ds._organoids[org_id]
        labels[info["label"]] += 1
    acc = labels.get("Acceptable", 0)
    nacc = labels.get("Not Acceptable", 0)
    print(f"Label distribution: Acceptable={acc} ({100*acc/total_organoids:.1f}%), "
          f"Not Acceptable={nacc} ({100*nacc/total_organoids:.1f}%)")

    # Count total labeled image records across all days
    total_labeled_images = sum(
        len(info["records"]) for info in ds._organoids.values()
    )
    print(f"Total labeled image records across days: {total_labeled_images}")

    # --- Metabolite summary stats (Table 1) ---
    # Paper says "across organoid IDs and timepoints" — uses all filtered organoids
    print(f"\n--- Metabolite Summary Statistics (Table 1) ---")

    met_values = {m: [] for m in METABOLITE_NAMES}

    from pipeline.data_loader import CONDITIONAL_METABOLITES, _get_day_number

    for org_id in ds.organoid_ids:
        info = ds._organoids[org_id]
        for day, rec in info["records"].items():
            day_num = _get_day_number(day)
            mets = rec.get("metabolites", {})
            for m in METABOLITE_NAMES:
                # Apply conditional metabolite filtering (e.g. MalateGlo only for days > 10)
                if m in CONDITIONAL_METABOLITES:
                    if day_num is None or not CONDITIONAL_METABOLITES[m](day_num):
                        continue
                if m in mets:
                    conc = mets[m].get("concentration_uM")
                    if conc is not None:
                        met_values[m].append(conc)

    rows = []
    for m in METABOLITE_NAMES:
        vals = np.array(met_values[m])
        rows.append({
            "Metabolite": DISPLAY_NAMES[m],
            "Mean": round(vals.mean(), 3),
            "Min": round(vals.min(), 3),
            "Max": round(vals.max(), 3),
            "Std. Dev.": round(vals.std(), 3),
            "N": len(vals),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "metabolite_summary_table.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR / 'metabolite_summary_table.csv'}")

    # Compare with paper values
    paper = {
        "Glucose concentration (μM)":   {"Mean": 11.612, "Min": -0.083, "Max": 19.920, "Std. Dev.": 2.597},
        "Glutamate concentration (μM)": {"Mean": 1.289,  "Min": 0.024,  "Max": 22.902, "Std. Dev.": 1.008},
        "Lactate concentration (μM)":   {"Mean": 10.791, "Min": 0.059,  "Max": 46.680, "Std. Dev.": 6.647},
        "Pyruvate concentration (μM)":  {"Mean": 2.820,  "Min": -0.462, "Max": 6.738,  "Std. Dev.": 0.865},
        "Malate concentration (μM)":    {"Mean": 0.120,  "Min": -0.167, "Max": 26.818, "Std. Dev.": 0.772},
    }

    print(f"\n--- Comparison with paper values ---")
    for _, row in df.iterrows():
        name = row["Metabolite"]
        if name in paper:
            p = paper[name]
            diffs = []
            for col in ["Mean", "Min", "Max", "Std. Dev."]:
                diff = abs(row[col] - p[col])
                status = "OK" if diff < 0.01 else f"DIFF={diff:.3f}"
                diffs.append(f"{col}: {status}")
            print(f"  {name}: {', '.join(diffs)}")


if __name__ == "__main__":
    main()
