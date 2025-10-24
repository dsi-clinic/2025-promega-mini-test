import json
import pandas as pd
from collections import defaultdict

# --- Step 1: Load data ---
with open("all_data.json", "r") as f:
    data = json.load(f)

# --- Step 2: Summary template ---
summary = defaultdict(lambda: {
    "Number of images": 0,
    "Number of organoids": 0,
    "Metabolites match": 0,
    "Metabolites don't match": 0,
    "Number stitched": 0,
    "Number split": 0,
    "Survey Acceptable votes": 0,
    "Survey Not Acceptable votes": 0
})

# --- Step 3: Loop through each organoid ---
for key, entry in data.items():
    day = entry.get("dayID", "Unknown")

    summary[day]["Number of organoids"] += 1
    summary[day]["Number of images"] += len(entry.get("all_files", []))

    # --- Metabolites ---
    for meta in entry.get("metabolites", {}).values():
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
    survey = entry.get("survey", {})
    evaluations = survey.get("evaluations", [])
    for e in evaluations:
        eval_result = e.get("evaluation", "").lower()
        if "not acceptable" in eval_result:
            summary[day]["Survey Not Acceptable votes"] += 1
        elif "acceptable" in eval_result:
            summary[day]["Survey Acceptable votes"] += 1

# --- Step 4: Convert to DataFrame ---
df = pd.DataFrame(summary).T
df.index.name = "Day"
df = df.sort_index()

print(df)
df.to_csv("Summary_Table.csv")
