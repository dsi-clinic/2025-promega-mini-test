import json
import pandas as pd
from collections import defaultdict
import re

# --- Step 1: Load data ---
with open("all_data.json", "r") as f:
    data = json.load(f)

metabolite_names = ["GlucoseGlo", "GlutamateGlo", "MalateGlo", "BCAAGlo", "LactateGlo", "PyruvateGlo"]

# --- Step 2: Summary template ---
def summary_template():
    return {
        "Number of images": 0,
        "Number of organoids": 0,
        "Metabolites match": 0,
        "Metabolites don't match": 0,
        "Number stitched": 0,
        "Number split": 0,
        "Survey Acceptable votes": 0,
        "Survey Not Acceptable votes": 0,
    }

summary = defaultdict(summary_template)
all_batches = set()  # keep track of all batch names seen

# --- Step 3: Loop through organoids ---
for key, entry in data.items():
    day = entry.get("dayID", "Unknown")

    # detect batch (BA code) from main_id
    main_id = entry.get("main_id", "")
    batch_match = re.match(r"(BA\d+)", main_id)
    batch = batch_match.group(1) if batch_match else "Unknown"
    all_batches.add(batch)

    # ensure batch column exists for this day
    if f"Metabolite Data {batch}" not in summary[day]:
        summary[day][f"Metabolite Data {batch}"] = 0

    # base counters
    summary[day]["Number of organoids"] += 1
    summary[day]["Number of images"] += len(entry.get("all_files", []))

    # --- Metabolites ---
    metabolites = entry.get("metabolites", {})
    if any(m in metabolites for m in metabolite_names):
        summary[day][f"Metabolite Data {batch}"] += 1

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
    for e in entry.get("survey", {}).get("evaluations", []):
        eval_result = e.get("evaluation", "").lower()
        if "not acceptable" in eval_result:
            summary[day]["Survey Not Acceptable votes"] += 1
        elif "acceptable" in eval_result:
            summary[day]["Survey Acceptable votes"] += 1

# --- Step 4: Convert to DataFrame ---
df = pd.DataFrame(summary).T
df.index.name = "Day"

# reorder columns so batch columns come last
regular_cols = [
    "Number of images",
    "Number of organoids",
    "Metabolites match",
    "Metabolites don't match",
    "Number stitched",
    "Number split",
    "Survey Acceptable votes",
    "Survey Not Acceptable votes",
]
batch_cols = [f"Metabolite Data {b}" for b in sorted(all_batches)]
df = df.reindex(columns=regular_cols + batch_cols)

df = df.sort_index()

print(df)
df.to_csv("Summary_Table.csv")
