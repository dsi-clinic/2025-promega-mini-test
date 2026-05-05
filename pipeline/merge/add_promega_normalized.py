"""
Add Promega-normalized metabolite fields (win, win_vol_norm) to all_data.json.

Reads:  data/normalized/CONC_data_organoides_residualized_final.csv
Writes: data/all_data.json  (in-place enrichment — no structural changes)
Logs:   data/normalized/missing_report.txt

Run:
    python -m pipeline.merge.add_promega_normalized

Recovery:
    git checkout data/all_data.json
"""

import json
import math
import pandas as pd
from pathlib import Path

ALL_DATA_PATH = Path("data/all_data.json")
CONC_CSV_PATH = Path("data/normalized/CONC_data_organoides_residualized_final.csv")
REPORT_PATH = Path("data/normalized/missing_report.txt")

METABOLITES = ["GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo", "MalateGlo", "BCAAGlo"]

# Map integer day from CSV to canonical day.id used in all_data.json
DAY_INT_TO_ID = {
    3: "Dy3", 6: "Dy6", 8: "Dy8", 10: "Dy10", 13: "Dy13",
    15: "Dy15", 17: "Dy17", 21: "Dy20.5", 24: "Dy24", 28: "Dy28", 30: "Dy30",
}


def _null(val) -> float | None:
    """Return None for NaN/None, else the value."""
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except TypeError:
        pass
    return val


def build_lookup(df: pd.DataFrame) -> dict:
    """Build {(organoid_id, day_id): {metabolite: {win, win_vol_norm}}} from CSV."""
    lookup = {}
    for _, row in df.iterrows():
        organoid_id = row["Organoid"]
        day_int = int(row["Day"])
        day_id = DAY_INT_TO_ID.get(day_int)
        if day_id is None:
            continue
        met_data = {}
        for met in METABOLITES:
            met_data[met] = {
                "win": _null(row.get(f"{met}_win")),
                "win_vol_norm": _null(row.get(f"{met}_win_vol_norm")),
            }
        lookup[(organoid_id, day_id)] = met_data
    return lookup


def main():
    print(f"Loading {ALL_DATA_PATH} ...")
    with open(ALL_DATA_PATH) as f:
        all_data = json.load(f)

    print(f"Loading {CONC_CSV_PATH} ...")
    df = pd.read_csv(CONC_CSV_PATH)

    lookup = build_lookup(df)
    print(f"CSV lookup entries: {len(lookup)}")

    missing_from_csv = []       # in all_data.json with metabolites but no CSV match
    partial_in_csv = []         # matched but some win or win_vol_norm are null

    matched = 0
    for key, record in all_data.items():
        organoid_id = record.get("organoid_id")
        day_id = record.get("day", {}).get("id")
        met_block = record.get("metabolite")

        # only care about records that already have metabolite data
        if not met_block:
            continue

        csv_key = (organoid_id, day_id)
        if csv_key not in lookup:
            missing_from_csv.append(key)
            continue

        matched += 1
        csv_mets = lookup[csv_key]
        missing_fields = []

        for met, values in csv_mets.items():
            if met not in met_block:
                continue
            met_block[met]["win"] = values["win"]
            met_block[met]["win_vol_norm"] = values["win_vol_norm"]
            if values["win"] is None or values["win_vol_norm"] is None:
                missing_fields.append(met)

        if missing_fields:
            partial_in_csv.append((key, missing_fields))

    print(f"Matched and enriched: {matched} records")
    print(f"Missing from CSV entirely: {len(missing_from_csv)} records")
    print(f"Partial (some win/win_vol_norm null): {len(partial_in_csv)} records")

    print(f"Writing enriched data to {ALL_DATA_PATH} ...")
    with open(ALL_DATA_PATH, "w") as f:
        json.dump(all_data, f, indent=2)

    with open(REPORT_PATH, "w") as f:
        f.write("=== Records in all_data.json (with metabolites) but NOT in CONC CSV ===\n")
        f.write(f"Total: {len(missing_from_csv)}\n\n")
        for key in missing_from_csv:
            f.write(f"  {key}\n")

        f.write("\n=== Records matched in CSV but with null win or win_vol_norm ===\n")
        f.write(f"Total: {len(partial_in_csv)}\n\n")
        for key, fields in partial_in_csv:
            f.write(f"  {key}: {', '.join(fields)}\n")

    print(f"Missing report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
