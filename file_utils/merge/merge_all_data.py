import json, math
from pathlib import Path
from tqdm import tqdm

from config import ORIGINAL_MAPPING, OUTPUT_FOLDER, METABOLITE_MAP_JSON, SURVEY_AGGREGATED_JSON
from file_utils.common.organoid_patterns import OrganoidNormalizer, day_from_key

norm_key_with_suffix = OrganoidNormalizer.normalize_key_with_suffix

# Paths
base_image_mapping_path = ORIGINAL_MAPPING
metabolite_json_path = METABOLITE_MAP_JSON
survey_json_path = SURVEY_AGGREGATED_JSON
processed_parent = Path(OUTPUT_FOLDER)
output_path = OUTPUT_FOLDER / "all_data.json"

# Utility functions
def load_json(path: Path):
    with open(path) as f:
        return json.load(f)

def clean_nan_values(obj):
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def is_organoid_key(key: str) -> bool:
    return isinstance(key, str) and key.strip().upper().startswith("BA")

def to_mdl_day(day: int | None) -> float | None:
    if day is None:
        return None
    return 20.5 if day in (20, 21) else float(day)

def _extract_infer_res(v: dict) -> str | None:
    for field in ("img_path", "mask_path"):
        s = v.get(field)
        if not isinstance(s, str):
            continue
        if "processed_dataset_" in s:
            return None
        res = OrganoidNormalizer.extract_resolution(s)
        if res:
            return res
    return None

def get_parent_key(key: str) -> str:
    """Extract parent key from a split key or return the key if not a split"""
    if " split_" in key:
        return key.split(" split_")[0]
    return key

def is_split_entry(key: str, entry: dict) -> bool:
    """Check if an entry is a split based on key suffix OR split_index field"""
    return " split_" in key or entry.get("split_index") is not None

def get_actual_split_key(key: str, entry: dict) -> str:
    """Generate the correct split key based on entry data"""
    if " split_" in key:
        return key  # Already has suffix
    
    split_idx = entry.get("split_index")
    if split_idx is not None:
        parent_key = key
        return f"{parent_key} split_{split_idx}"
    
    return key  # Not a split

# Load base image map
entries_data = load_json(base_image_mapping_path)
entries_data = entries_data.get("entries", entries_data)

# Stage 1: Normalize base map and fix split keys
base_map = {}
key_corrections = {}  # Map old keys to new keys

for k, v in entries_data.items():
    if not is_organoid_key(k):
        continue
    try:
        normk = norm_key_with_suffix(k)
        
        # Check if this needs to be converted to a split key
        if is_split_entry(normk, v):
            actual_key = get_actual_split_key(normk, v)
            base_map[actual_key] = v
            if actual_key != normk:
                key_corrections[normk] = actual_key
                print(f"[KEY CORRECTION] {normk} -> {actual_key} (split_index: {v.get('split_index')})")
        else:
            base_map[normk] = v
            
    except ValueError as e:
        print(f"[BASE] Failed to normalize: {k} — {e}")

# Stage 2: Processed masks and images (apply key corrections)
processed_map = {}
for p in processed_parent.rglob("image_mapping_*_processed.json"):
    data = load_json(p)
    for k, v in data.items():
        if not is_organoid_key(k):
            continue
        try:
            norm_k = norm_key_with_suffix(k)
            
            # Apply key correction if needed
            if norm_k in key_corrections:
                norm_k = key_corrections[norm_k]
            
            res = _extract_infer_res(v)
            if not res:
                continue
            processed_map.setdefault(norm_k, {})[res] = v
        except ValueError as e:
            print(f"[PROCESSED] Failed to normalize: {k} — {e}")

# Stage 3: Metabolites (only for parent keys)
metab_map = {}
for k, v in load_json(metabolite_json_path).items():
    if not is_organoid_key(k):
        continue
    try:
        norm_k = norm_key_with_suffix(k)
        
        # Apply key correction if needed
        if norm_k in key_corrections:
            norm_k = key_corrections[norm_k]
            
        parent_key = get_parent_key(norm_k)
        metab_map[parent_key] = v
    except ValueError as e:
        print(f"[METABOLITE] Failed to normalize: {k} — {e}")

# Stage 4: Surveys (apply key corrections)
survey_map = {}
survey_data = load_json(survey_json_path)
for row in survey_data.values():
    iid = None
    if row.get("evaluations"):
        iid = row["evaluations"][0].get("image_id")
    elif row.get("quality_scores"):
        iid = row["quality_scores"][0].get("image_id")
    if not iid:
        continue
    try:
        norm_k = OrganoidNormalizer.normalize_key(iid)
        split_idx = None
        if "evaluations" in row:
            idxs = {e.get("split_index") for e in row["evaluations"] if e.get("split_index") is not None}
            if len(idxs) == 1:
                split_idx = idxs.pop()
        if split_idx is None and "quality_scores" in row:
            idxs = {e.get("split_index") for e in row["quality_scores"] if e.get("split_index") is not None}
            if len(idxs) == 1:
                split_idx = idxs.pop()
        
        full_key = f"{norm_k} split_{split_idx}" if split_idx is not None else norm_k
        
        # Apply key correction if needed
        if full_key in key_corrections:
            full_key = key_corrections[full_key]
            
        survey_map[full_key] = row
    except ValueError as e:
        print(f"[SURVEY] Failed to normalize: {iid} — {e}")

# Find all keys and identify split relationships
all_keys = set(base_map) | set(processed_map) | set(metab_map) | set(survey_map)
parent_keys = set()
split_keys = set()

for k in all_keys:
    if " split_" in k:
        split_keys.add(k)
        parent_keys.add(get_parent_key(k))
    else:
        parent_keys.add(k)

print(f"Found {len(split_keys)} split entries and {len(parent_keys)} parent entries")

combined = {}
shared_fields = ["dayID", "BA", "wellID", "cellLine", "treatment"]
split_fields = [
    "split_index", "Classification", "um_per_px", "all_files", "Best Z",
    "Best Z Filename", "Actual Z Value", "Blank", "blank_area_frac"
]

# Process all entries
for k in tqdm(sorted(all_keys)):
    entry = {}
    base_entry = base_map.get(k, {})
    is_split = " split_" in k
    parent_key = get_parent_key(k)

    # Add shared fields from base entry
    for f in shared_fields:
        if f in base_entry:
            entry[f] = base_entry[f]

    # Add day information
    _day = day_from_key(k)
    entry["day_num"] = _day
    entry["mdl_day"] = to_mdl_day(_day)

    if is_split:
        # Split entry: add split-specific fields
        for f in split_fields:
            if f in base_entry:
                entry[f] = base_entry[f]
        
        # Add processed data nested under resolution key
        if k in processed_map:
            for res, proc_data in processed_map[k].items():
                entry[res] = proc_data
        
        # Add survey data
        if k in survey_map:
            entry["survey"] = survey_map[k]
    
    else:
        # Parent entry: check if it has splits
        child_splits = [sk for sk in split_keys if get_parent_key(sk) == k]
        
        if child_splits:
            # Has splits: only add metadata and metabolites
            entry["split_children"] = sorted(child_splits)
            if k in metab_map:
                entry["metabolites"] = metab_map[k]
        else:
            # No splits: add all data including technical fields
            for f in split_fields:
                if f in base_entry:
                    entry[f] = base_entry[f]
            
            # Add processed data nested under resolution key
            if k in processed_map:
                for res, proc_data in processed_map[k].items():
                    entry[res] = proc_data
            
            # Add survey data
            if k in survey_map:
                entry["survey"] = survey_map[k]
            
            # Add metabolites
            if k in metab_map:
                entry["metabolites"] = metab_map[k]

    combined[k] = entry

# Clean and dump
with open(output_path, "w") as f:
    json.dump(clean_nan_values(combined), f, indent=2)

print(f"Wrote {len(combined):,} merged records → {output_path}")

# Debug: Print some split relationships
print("\nSample split relationships:")
for parent in sorted(parent_keys)[:5]:
    children = [sk for sk in split_keys if get_parent_key(sk) == parent]
    if children:
        print(f"  {parent} -> {children}")
