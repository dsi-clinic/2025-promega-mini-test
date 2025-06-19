import os
import json
from glob import glob
from tqdm import tqdm

# --- Paths ---
base_image_mapping_path = '/net/projects2/promega/data-analysis/output/image_mapping.json'
processed_root_dir = '/net/projects2/promega/data-analysis/output/processed_dataset_256x192'
survey_json_path = '/home/amandabrooke/2025-promega-mini-test/surveys/organoid_surveys_aggregated.json'
metabolite_json_path = '/net/projects2/promega/data-analysis/metabolite_data/metabolite_map.json'
output_path = 'final_combined_metadata.json'

# --- Load base image mapping ---
with open(base_image_mapping_path) as f:
    base_mapping = json.load(f)

# --- Load survey evaluations ---
with open(survey_json_path) as f:
    survey_data = json.load(f)

# --- Load metabolite data ---
with open(metabolite_json_path) as f:
    metabolite_data = json.load(f)

# --- Normalize survey data into mapping keyed by "BA1 Dy03 A1" ---
survey_lookup = {}
for org_id, survey_entry in survey_data.items():
    if 'evaluations' in survey_entry and survey_entry['evaluations']:
        image_id = survey_entry['evaluations'][0]['image_id']
    elif 'quality_scores' in survey_entry and survey_entry['quality_scores']:
        image_id = survey_entry['quality_scores'][0]['image_id']
    else:
        continue

    # Normalize image_id → e.g. "Ba2 96_2 Dy30 H11" → "BA2 Dy30 H11"
    parts = image_id.strip().split()
    if len(parts) >= 4:
        BA = parts[0].split('_')[0].upper()
        dayID = parts[2]
        wellID = parts[3]
        key = f"{BA} {dayID} {wellID}"
        survey_lookup[key] = survey_entry

# --- Load all nested processed mappings ---
processed_mapping = {}
for dirpath, _, filenames in os.walk(processed_root_dir):
    for file in filenames:
        if file.startswith("image_mapping_") and file.endswith("_processed.json"):
            full_path = os.path.join(dirpath, file)
            with open(full_path) as f:
                data = json.load(f)
                processed_mapping.update(data)

# --- Merge everything ---
final_data = {}

for key in base_mapping:
    entry = {}

    # Base mapping
    entry.update(base_mapping[key])

    # Processed mask info
    if key in processed_mapping:
        entry.update(processed_mapping[key])

    # Survey info
    if key in survey_lookup:
        entry["survey"] = survey_lookup[key]

    # Metabolite info
    if key in metabolite_data:
        entry["metabolites"] = metabolite_data[key]

    final_data[key] = entry

# --- Save ---
with open(output_path, 'w') as f:
    json.dump(final_data, f, indent=2)

print(f"Merged metadata written to: {output_path} ({len(final_data)} total entries)")
