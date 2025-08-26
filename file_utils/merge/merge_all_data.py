#!/usr/bin/env python3
import os
import json
import math
import pathlib
from glob import glob
from pathlib import Path
from tqdm import tqdm

# ───────── paths ─────────────────────────────────────────────────────────
from config import ORIGINAL_MAPPING, OUTPUT_FOLDER, BASE_PATH, METABOLITE_MAP_JSON, SURVEY_AGGREGATED_JSON
from file_utils.common.organoid_patterns import OrganoidNormalizer, norm_key, day_from_key

base_image_mapping_path = ORIGINAL_MAPPING
# processed_root_dir = INFER_AUTO_PROCESSED_DIR  # if/when you need it

metabolite_json_path = METABOLITE_MAP_JSON
survey_json_path    = SURVEY_AGGREGATED_JSON
processed_parent = str(OUTPUT_FOLDER)

output_path             = 'all_data.json'

# Regex patterns now centralized in organoid_patterns module

def to_mdl_day(day: int | None) -> float | None:
    if day is None:
        return None
    # collapse Dy20 and Dy21 to 20.5
    if day in (20, 21):
        return 20.5
    return float(day)

# norm_key and day_from_key now imported from organoid_patterns module


# ───────── read files & re-key with norm_key() ───────────────────────────
def load_json(path):
    with open(path) as f: return json.load(f)

def clean_nan_values(obj):
    """Recursively replace NaN values with None for valid JSON serialization"""
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

base_map     = {norm_key(k): v for k, v in load_json(base_image_mapping_path).items()}
metab_map    = {norm_key(k): v for k, v in load_json(metabolite_json_path).items()}


processed_map = {}


for p in pathlib.Path(processed_parent).rglob("image_mapping_*_processed.json"):
    if "auto_processed" in str(p):
        for k, v in load_json(p).items():
            norm_k = norm_key(k)
            resolution = OrganoidNormalizer.extract_resolution(str(p)) or "unknown"
            if norm_k not in processed_map:
                processed_map[norm_k] = {}
            processed_map[norm_k][resolution] = v


# survey – one file, keys are inside each record
survey_map = {}
for row in load_json(survey_json_path).values():
    # Try getting image_id from first evaluation
    iid = None
    if row.get("evaluations"):
        iid = row["evaluations"][0].get("image_id")
    elif row.get("quality_scores"):
        iid = row["quality_scores"][0].get("image_id")

    if iid:
        try:
            survey_map[norm_key(iid)] = row
        except ValueError as e:
            print(f"[SURVEY] Failed to normalize: {iid} — {e}")

# ───────── merge all sources ─────────────────────────────────────────────
all_keys   = set().union(base_map, processed_map, survey_map, metab_map)
combined   = {}

for k in tqdm(sorted(all_keys)):
    entry = {}
    if k in base_map:      entry.update(base_map[k])
    if k in processed_map: entry.update(processed_map[k])
    if k in survey_map:    entry["survey"]     = survey_map[k]
    if k in metab_map:     entry["metabolites"] = metab_map[k]
    
    _day = day_from_key(k)          # e.g., 20, 21, 30, ...
    entry["day_num"] = _day         # optional: raw numeric day
    entry["mdl_day"] = to_mdl_day(_day)  # 20/21 -> 20.5; others unchanged
    combined[k] = entry

# ───────── write out ─────────────────────────────────────────────────────
# Clean NaN values before writing to ensure valid JSON
combined_clean = clean_nan_values(combined)
with open(output_path, 'w') as f:
    json.dump(combined_clean, f, indent=2)
print(f"Wrote {len(combined):,} merged records → {output_path}")

