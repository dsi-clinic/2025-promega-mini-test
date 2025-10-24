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
    "Survey 5-0": 0,
    "Survey 4-1": 0,
    "Survey 3-2": 0,
    "Survey 2-3": 0,
    "Survey 1-4": 0,
    "Survey 0-5": 0,
    "Survey Quality - Good": 0,
    "Survey Quality - Reasonable": 0,
    "Survey Quality - Other": 0,
    "Average Acceptance Rate (%)": 0.0
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
    cls = entry.get("Classification", "").lower()
    if "split" in cls:
        summary[day]["Number split"] += 1
    if "stitched" in cls:
        summary[day]["Number stitched"] += 1

    # --- Survey evaluations ---
    survey = entry.get("survey", {})
    evaluations = survey.get("evaluations", [])
    acceptable = 0
    not_acceptable = 0
    total_reviews = len(evaluations)

    for e in evaluations:
        eval_result = e.get("evaluation", "").lower()
        if "not acceptable" in eval_result:
            not_acceptable += 1
        elif "acceptable" in eval_result:
            acceptable += 1

    # record the pattern of survey votes (e.g., 4 acceptable, 1 not acceptable)
    key_label = f"Survey {acceptable}-{not_acceptable}"
    if key_label in summary[day]:
        summary[day][key_label] += 1

    # calculate acceptance rate for this organoid (avoid divide by 0)
    if total_reviews > 0:
        acceptance_rate = acceptable / total_reviews * 100
        summary[day]["Average Acceptance Rate (%)"] += acceptance_rate

    # --- Survey quality scores ---
    for q in survey.get("quality_scores", []):
        q_val = q.get("quality", "").lower()
        if "good" in q_val:
            summary[day]["Survey Quality - Good"] += 1
        elif "reason" in q_val:
            summary[day]["Survey Quality - Reasonable"] += 1
        elif q_val:
            summary[day]["Survey Quality - Other"] += 1

# --- Step 4: Adjust average acceptance to actual mean per day ---
for day, stats in summary.items():
    if stats["Number of organoids"] > 0:
        stats["Average Acceptance Rate (%)"] /= stats["Number of organoids"]

# --- Step 5: Convert to DataFrame ---
df = pd.DataFrame(summary).T
df.index.name = "Day"
df = df.sort_index()

print(df)
df.to_csv("summary_by_day_with_survey_patterns.csv")
