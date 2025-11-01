import json
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import re
from typing import Dict


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
metabolite_names = [
    "GlucoseGlo", "GlutamateGlo", "MalateGlo",
    "BCAAGlo", "LactateGlo", "PyruvateGlo",
]
survey_patterns = ["5-0", "4-1", "3-2", "2-3", "1-4", "0-5"]
batch_names = ["BA1", "BA2", "BA3", "BA4"]  # add more if needed


# ---------------------------------------------------------
# Template for per-day summary
# ---------------------------------------------------------
def summary_template() -> Dict[str, int]:
    """
    Base summary template for organoid and metabolite statistics.
    """
    template = {
        "Number of images": 0,
        "Number of organoids": 0,
        "Metabolites match": 0,
        "Metabolites don't match": 0,
        "Metabolite Data Available": 0,
        "Number stitched": 0,
        "Number split": 0,
    }

    # global survey columns (aggregate across batches)
    for s in survey_patterns:
        template[f"Survey {s}"] = 0

    # batch-specific survey + total columns
    for b in batch_names:
        for s in survey_patterns:
            template[f"Survey {s} ({b})"] = 0
        template[f"{b} Total"] = 0

    return template


# ---------------------------------------------------------
# Generate summary table
# ---------------------------------------------------------
def generate_summary_table(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Generate a summary statistics table by day, including per-batch breakdowns.
    """
    # --- Load data ---
    with open(input_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {input_path}")

    summary = defaultdict(summary_template)

    # --- Iterate through all organoid entries ---
    for _, entry in data.items():
        day = entry.get("dayID", "Unknown")
        summary[day]["Number of organoids"] += 1
        summary[day]["Number of images"] += len(entry.get("all_files", []))

        # --- Detect batch robustly ---
        batch = None

        # Priority 1: dedicated "BA" field
        if "BA" in entry and isinstance(entry["BA"], str):
            match = re.search(r"BA\d+", entry["BA"], re.IGNORECASE)
            if match:
                batch = match.group(0).upper()

        # Priority 2: fallback search through main_id or verification fields
        if batch is None:
            possible_fields = [
                entry.get("main_id", ""),
                entry.get("verification", {}).get("main_id", ""),
                entry.get("processed", {}).get("main_id", ""),
            ]
            for f in possible_fields:
                match = re.search(r"BA\d+", str(f), re.IGNORECASE)
                if match:
                    batch = match.group(0).upper()
                    break

        # Default to Unknown if not found
        if batch is None:
            batch = "Unknown"

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

        # --- Survey evaluations ---
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

            # Global (day-level total)
            if label in summary[day]:
                summary[day][label] += 1
            else:
                summary[day][label] = 1

            # Batch-specific survey counts
            batch_label = f"{label} ({batch})"
            if batch_label in summary[day]:
                summary[day][batch_label] += 1
            else:
                summary[day][batch_label] = 1

        # --- Batch total ---
        if f"{batch} Total" in summary[day]:
            summary[day][f"{batch} Total"] += 1
        else:
            summary[day][f"{batch} Total"] = 1

    # --- Convert to DataFrame ---
    df = pd.DataFrame(summary).T
    df.index.name = "Day"

    # Define column order for clarity
    regular_cols = [
        "Number of images", "Number of organoids",
        "Metabolites match", "Metabolites don't match",
        "Metabolite Data Available", "Number stitched", "Number split",
    ]
    survey_cols = [f"Survey {s}" for s in survey_patterns]
    batch_survey_cols = [f"Survey {s} ({b})" for b in batch_names for s in survey_patterns]
    batch_total_cols = [f"{b} Total" for b in batch_names]

    df = df.reindex(columns=regular_cols + survey_cols + batch_survey_cols + batch_total_cols)
    df = df.sort_index()

    # --- Save output ---
    df.to_csv(output_path, index=True)
    print(f"Summary table saved to: {output_path}")

    return df


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
def main():
    current_dir = Path(__file__).resolve().parent
    input_path = current_dir / "all_data.json"
    output_path = current_dir / "Summary_Table.csv"
    generate_summary_table(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
