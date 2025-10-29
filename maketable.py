import json
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import re
from typing import Dict


metabolite_names = [
    "GlucoseGlo",
    "GlutamateGlo",
    "MalateGlo",
    "BCAAGlo",
    "LactateGlo",
    "PyruvateGlo",
]


def summary_template() -> Dict[str, int]:
    """
    Create a base summary template for organoid and metabolite statistics.

    Returns
    -------
    Dict[str, int]
        A dictionary initialized with counters for key organoid metrics,
        including image counts, metabolite matches, splits, stitches,
        and survey votes.
    """
    return {
        "Number of images": 0,
        "Number of organoids": 0,
        "Metabolites match": 0,
        "Metabolites don't match": 0,
        "Metabolite Data Available": 0,
        "Number stitched": 0,
        "Number split": 0,
        "Survey 5-0": 0,
        "Survey 4-1": 0,
        "Survey 3-2": 0,
        "Survey 2-3": 0,
        "Survey 1-4": 0,
        "Survey 0-5": 0,
    }


def generate_summary_table(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Generate and save a summary statistics table from organoid data.

    Parameters
    ----------
    input_path : Path
        Path to the JSON file containing all organoid data (e.g., all_data.json).
    output_path : Path
        Path to save the generated summary CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing day-wise counts of organoids, images,
        metabolite matches, and survey evaluations.
    """
    # --- Step 1: Load data ---
    with open(input_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {input_path}")

    # --- Step 2: Initialize summary ---
    summary = defaultdict(summary_template)

    # --- Step 3: Loop through organoids ---
    for _, entry in data.items():
        day = entry.get("dayID", "Unknown")

        # base counters
        summary[day]["Number of organoids"] += 1
        summary[day]["Number of images"] += len(entry.get("all_files", []))

        # --- Metabolites ---
        metabolites = entry.get("metabolites", {})
        if any(m in metabolites for m in metabolite_names):
            summary[day]["Metabolite Data Available"] += 1

        for meta in metabolites.values():
            if meta.get("is_outlier", False):
                summary[day]["Metabolites don't match"] += 1
            else:
                summary[day]["Metabolites match"] += 1

        # --- Split / Stitched classification ---
        cls = entry.get("verification", {}).get("classification_verification", "").lower()
        if "split" in cls:
            summary[day]["Number split"] += 1
        if "stitch" in cls or "stitched" in cls:
            summary[day]["Number stitched"] += 1

        # --- Survey evaluations (aggregate as 5-0, 4-1, etc.) ---
        evaluations = entry.get("survey", {}).get("evaluations", [])
        if evaluations:
            votes = Counter({"Acceptable": 0, "Not Acceptable": 0})
            for e in evaluations:
                eval_result = e.get("evaluation", "").lower()
                if "not acceptable" in eval_result:
                    votes["Not Acceptable"] += 1
                elif "acceptable" in eval_result:
                    votes["Acceptable"] += 1

            a, n = votes["Acceptable"], votes["Not Acceptable"]
            label = f"Survey {a}-{n}"
            if label in summary[day]:
                summary[day][label] += 1
            else:
                summary[day][label] = 1  # in case of unexpected combos (like 6-0)

    # --- Step 4: Convert to DataFrame ---
    df = pd.DataFrame(summary).T
    df.index.name = "Day"

    regular_cols = [
        "Number of images",
        "Number of organoids",
        "Metabolites match",
        "Metabolites don't match",
        "Metabolite Data Available",
        "Number stitched",
        "Number split",
        "Survey 5-0",
        "Survey 4-1",
        "Survey 3-2",
        "Survey 2-3",
        "Survey 1-4",
        "Survey 0-5",
    ]
    df = df.reindex(columns=regular_cols)
    df = df.sort_index()

    # --- Step 5: Save ---
    df.to_csv(output_path, index=True)
    print(f"Summary table saved to: {output_path}")

    return df


def main():
    """
    Entry point for generating the summary table.

    This allows the script to be run directly from the command line.
    Example:
        python maketable.py
    """
    current_dir = Path(__file__).resolve().parent
    input_path = current_dir / "all_data.json"
    output_path = current_dir / "Summary_Table.csv"

    generate_summary_table(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
