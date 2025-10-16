#!/usr/bin/env python3
import json, re
from pathlib import Path
from tqdm import tqdm
from file_utils.common.organoid_patterns import OrganoidNormalizer

from config import (
    ORIGINAL_MAPPING,
    INFER_RESIZED_DIR,
    METABOLITE_MAP_JSON,
    SURVEY_AGGREGATED_JSON,
    MANUAL_THRESHOLD_MAPPING,
    ALL_DATA_JSON,
)

OUTPUT_PATH = str(ALL_DATA_JSON)

# ---------- helpers ----------
def load_json(path: Path | str):
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

def sanitize_for_json(obj):
    """
    Recursively sanitize data to be JSON-safe.
    - Converts NaN, inf, -inf to None
    - Handles nested dicts and lists
    """
    import math
    
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif obj is None or isinstance(obj, (str, int, bool)):
        return obj
    else:
        # Handle pandas NA, numpy nan, etc.
        try:
            if hasattr(obj, 'isna') and obj.isna():
                return None
        except (TypeError, ValueError):
            pass
        # Try to convert to string as fallback
        return str(obj)

def normalized_parent_key(id_like: str) -> str:
    """Use OrganoidNormalizer to get consistent BA# 96_# Dy## A# format (no suffixes)."""
    try:
        return OrganoidNormalizer.normalize_key(id_like)
    except ValueError:
        # fallback: return a stripped clean version if parsing fails
        return OrganoidNormalizer.clean_string(id_like).upper()

# ---------- load sources ----------
print(f"Loading base mapping: {ORIGINAL_MAPPING}")
base_json = load_json(ORIGINAL_MAPPING)
base_map = base_json.get("entries", {})

print(f"Loading metabolite map: {METABOLITE_MAP_JSON}")
metab_map = load_json(METABOLITE_MAP_JSON)

print(f"Loading survey data: {SURVEY_AGGREGATED_JSON}")
survey_json = load_json(SURVEY_AGGREGATED_JSON)

print(f"Loading manual threshold mapping: {MANUAL_THRESHOLD_MAPPING}")
manual_mask_map = load_json(MANUAL_THRESHOLD_MAPPING)

# Build survey map keyed by image_id or parent
survey_map = {}
for row in survey_json.values():
    ids = []
    if row.get("evaluations"):
        ids += [ev["image_id"] for ev in row["evaluations"] if "image_id" in ev]
    if row.get("quality_scores"):
        ids += [qs["image_id"] for qs in row["quality_scores"] if "image_id" in qs]

    for iid in ids:
        try:
            norm_key = OrganoidNormalizer.normalize_key(iid)
        except ValueError:
            norm_key = OrganoidNormalizer.clean_string(iid).upper()
        survey_map[norm_key] = row

# Build manual mask map with normalized keys
manual_mask_normalized = {}
for raw_key, manual_data in manual_mask_map.items():
    try:
        norm_key = OrganoidNormalizer.normalize_key(raw_key)
    except ValueError:
        norm_key = OrganoidNormalizer.clean_string(raw_key).upper()
    manual_mask_normalized[norm_key] = manual_data

# ---------- load processed JSONs ----------
processed_map = {}
found_files = list(Path(INFER_RESIZED_DIR).rglob("image_mapping*_processed.json"))

for p in found_files:
    raw = load_json(p)
    processed_map.update(raw)

# ---------- merge ----------
# ---------- merge ----------
combined = {}
manual_mask_count = 0

# Add this helper function before the loop
def extract_mdl_day(day_id: str) -> float:
    """Extract numerical day from dayID (e.g., 'Dy17' -> 17.0, 'Dy20' or 'Dy21' -> 20.5)"""
    if not day_id:
        return None
    # Extract numbers from dayID
    match = re.search(r'(\d+(?:\.\d+)?)', day_id)
    if match:
        day_num = float(match.group(1))
        # Handle day 20/21 -> 20.5 for consistency
        if day_num in [20.0, 21.0]:
            return 20.5
        return day_num
    return None

for raw_k, payload in tqdm(base_map.items(), desc="Merging"):
    entry = dict(payload)
    
    # Add common_key for key consistency (from old structure)
    entry['common_key'] = raw_k
    
    # Extract day_num and mdl_day from dayID (from old structure)
    if 'dayID' in entry:
        day_id = entry['dayID']
        # Extract day_num (integer day)
        day_match = re.search(r'(\d+)', day_id)
        if day_match:
            entry['day_num'] = int(day_match.group(1))
        else:
            entry['day_num'] = None
        # Extract mdl_day (float day with special handling)
        entry['mdl_day'] = extract_mdl_day(day_id)
    
    # Debug: Check if fields are being added (only for first entry)
    if raw_k == list(base_map.keys())[0]:
        print(f"DEBUG: First entry fields after adding: {sorted(entry.keys())}")

    processed = processed_map.get(raw_k) or processed_map.get(normalized_parent_key(raw_k))
    if processed:
        entry["processed"] = processed
        entry["main_id"] = processed.get("main_id")

    norm_key_parent = normalized_parent_key(raw_k)
    
    if norm_key_parent in survey_map:
        entry["survey"] = survey_map[norm_key_parent]

    if norm_key_parent in metab_map:
        entry["metabolites"] = metab_map[norm_key_parent]

    # Add manual mask path if available
    if norm_key_parent in manual_mask_normalized:
        manual_data = manual_mask_normalized[norm_key_parent]
        entry["manual_mask_path"] = manual_data.get("MT Mask Path")
        manual_mask_count += 1

    combined[raw_k] = entry
    
    # Debug: Check final entry for first record
    if raw_k == list(base_map.keys())[0]:
        print(f"DEBUG: Final entry fields before sanitization: {sorted(entry.keys())}")
# ---------- sanitize and write output ----------
print("\nSanitizing data for JSON...")
combined_clean = sanitize_for_json(combined)

# Debug: Check first entry after sanitization
first_key = list(combined_clean.keys())[0]
print(f"DEBUG: First entry fields after sanitization: {sorted(combined_clean[first_key].keys())}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(combined_clean, f, indent=2)

print(f"\nWrote {len(combined_clean):,} merged records → {OUTPUT_PATH}")
print(f"Found {manual_mask_count:,} manual masks")