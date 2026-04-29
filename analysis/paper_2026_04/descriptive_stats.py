#!/usr/bin/env python3
"""Reproduce dataset description numbers and metabolite summary table (Table 1).

Uses ``OrganoidDataset.iter_organoids()`` for the filtered dataset; falls back
to a raw json.load only for the unfiltered total counts (those need to
include unlabeled organoids that ``OrganoidDataset`` drops).

Outputs:
  - Console: dataset counts, label distribution, metabolite summary
  - $ANALYSIS_OUTPUT_DIR/figures/metabolite_summary_table.csv

Usage:
    make run ARGS="-m analysis.paper_2026_04.descriptive_stats"
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data_loader import (
    CONDITIONAL_METABOLITES,
    FIGURE_DIR,
    OrganoidDataset,
    extract_organoid_id,
    get_day_int_floor,
)

ALL_DATA_PATH = Path("data/all_data.json")
SPLITS_CSV = Path("data/2026_winter_student_splits.csv")
OUTPUT_DIR = FIGURE_DIR

METABOLITE_NAMES = ["GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo", "MalateGlo"]
DISPLAY_NAMES = {
    "GlucoseGlo": "Glucose concentration (μM)",
    "GlutamateGlo": "Glutamate concentration (μM)",
    "LactateGlo": "Lactate concentration (μM)",
    "PyruvateGlo": "Pyruvate concentration (μM)",
    "MalateGlo": "Malate concentration (μM)",
}

PAPER_VALUES = {
    "Glucose concentration (μM)":   {"Mean": 11.612, "Min": -0.083, "Max": 19.920, "Std. Dev.": 2.597},
    "Glutamate concentration (μM)": {"Mean": 1.289,  "Min": 0.024,  "Max": 22.902, "Std. Dev.": 1.008},
    "Lactate concentration (μM)":   {"Mean": 10.791, "Min": 0.059,  "Max": 46.680, "Std. Dev.": 6.647},
    "Pyruvate concentration (μM)":  {"Mean": 2.820,  "Min": -0.462, "Max": 6.738,  "Std. Dev.": 0.865},
    "Malate concentration (μM)":    {"Mean": 0.120,  "Min": -0.167, "Max": 26.818, "Std. Dev.": 0.772},
}


def _print_raw_counts(all_data_path: Path) -> None:
    """Counts that include unlabeled organoids — needs raw json.load."""
    with open(all_data_path) as f:
        all_records = json.load(f)
    print(f"Total records in all_data.json: {len(all_records)}")
    org_ids = {extract_organoid_id(k) for k in all_records}
    print(f"Unique organoids (all batches): {len(org_ids)}")
    days = sorted({v.get("day", {}).get("id", "") for v in all_records.values()})
    print(f"Days: {days}")

    labeled_days = set()
    labeled_records = 0
    for v in all_records.values():
        if v.get("survey", {}).get("evaluations"):
            labeled_records += 1
            labeled_days.add(v.get("day", {}).get("id"))
    print(f"Records with survey evaluations: {labeled_records}")
    print(f"Survey days: {sorted(d for d in labeled_days if d)}")


def _print_filtered_summary(ds: OrganoidDataset) -> None:
    print("\n--- Filtered dataset (paper config) ---")
    print(ds.summary())

    labels = Counter(label for _, info in ds.iter_organoids() for label in [info["label"]])
    total = sum(labels.values())
    acc = labels.get("Acceptable", 0)
    nacc = labels.get("Not Acceptable", 0)
    print(f"Label distribution: Acceptable={acc} ({100 * acc / total:.1f}%), "
          f"Not Acceptable={nacc} ({100 * nacc / total:.1f}%)")

    total_records = sum(len(info["records"]) for _, info in ds.iter_organoids())
    print(f"Total labeled image records across days: {total_records}")


def _collect_metabolite_values(ds: OrganoidDataset) -> dict:
    """Map metabolite name → list of concentration_uM values across the filtered dataset."""
    values = {m: [] for m in METABOLITE_NAMES}
    for _, info in ds.iter_organoids():
        for day, rec in info["records"].items():
            day_num = get_day_int_floor(day)
            mets = rec.get("metabolite", {})
            for m in METABOLITE_NAMES:
                if m in CONDITIONAL_METABOLITES:
                    if day_num is None or not CONDITIONAL_METABOLITES[m](day_num):
                        continue
                if m in mets:
                    conc = mets[m].get("concentration_uM")
                    if conc is not None:
                        values[m].append(conc)
    return values


def _build_summary_table(values: dict) -> pd.DataFrame:
    rows = []
    for m in METABOLITE_NAMES:
        vals = np.array(values[m])
        rows.append({
            "Metabolite": DISPLAY_NAMES[m],
            "Mean": round(vals.mean(), 3),
            "Min": round(vals.min(), 3),
            "Max": round(vals.max(), 3),
            "Std. Dev.": round(vals.std(), 3),
            "N": len(vals),
        })
    return pd.DataFrame(rows)


def _compare_with_paper(df: pd.DataFrame) -> None:
    print("\n--- Comparison with paper values ---")
    for _, row in df.iterrows():
        name = row["Metabolite"]
        if name not in PAPER_VALUES:
            continue
        p = PAPER_VALUES[name]
        diffs = []
        for col in ["Mean", "Min", "Max", "Std. Dev."]:
            diff = abs(row[col] - p[col])
            status = "OK" if diff < 0.01 else f"DIFF={diff:.3f}"
            diffs.append(f"{col}: {status}")
        print(f"  {name}: {', '.join(diffs)}")


def main():
    _print_raw_counts(ALL_DATA_PATH)

    ds = OrganoidDataset(str(ALL_DATA_PATH), splits_csv=str(SPLITS_CSV))
    _print_filtered_summary(ds)

    print("\n--- Metabolite Summary Statistics (Table 1) ---")
    values = _collect_metabolite_values(ds)
    df = _build_summary_table(values)
    print(df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "metabolite_summary_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    _compare_with_paper(df)


if __name__ == "__main__":
    main()
