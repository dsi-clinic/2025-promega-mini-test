import json
import pandas as pd
from collections import defaultdict

with open("all_data.json", "r") as f:
    data = json.load(f)

metabolite_names = ["GlucoseGlo", "GlutamateGlo", "MalateGlo", "BCAAGlo", "LactateGlo", "PyruvateGlo"]

def summary_template():
    base = {
        "Number of images": 0,
        "Number of organoids": 0,
        "Metabolites match": 0,
        "Metabolites don't match": 0,
        "Number stitched": 0,
        "Number split": 0,
        "Survey Acceptable votes": 0,
        "Survey Not Acceptable votes": 0,
    }
    for m in metabolite_names:
        base[f"{m} available"] = 0
    return base

summary = defaultdict(summary_template)

for key, entry in data.items():
    day = entry.get("dayID", "Unknown")

    summary[day]["Number of organoids"] += 1
    summary[day]["Number of images"] += len(entry.get("all_files", []))

    metabolites = entry.get("metabolites", {})
    for meta_name in metabolite_names:
        if meta_name in metabolites:
            summary[day][f"{meta_name} available"] += 1

    for meta in metabolites.values():
        if meta.get("is_outlier", False):
            summary[day]["Metabolites don't match"] += 1
        else:
            summary[day]["Metabolites match"] += 1

    cls = entry.get("verification", {}).get("classification_verification", "").lower()
    if "split" in cls:
        summary[day]["Number split"] += 1
    if "stitch" in cls or "stitched" in cls:
        summary[day]["Number stitched"] += 1

    survey = entry.get("survey", {})
    for e in survey.get("evaluations", []):
        eval_result = e.get("evaluation", "").lower()
        if "not acceptable" in eval_result:
            summary[day]["Survey Not Acceptable votes"] += 1
        elif "acceptable" in eval_result:
            summary[day]["Survey Acceptable votes"] += 1

df = pd.DataFrame(summary).T
df.index.name = "Day"
df = df.sort_index()

print(df)
df.to_csv("Summary_Table.csv")
