#!/usr/bin/env python3
import json, math, pathlib, re
from pathlib import Path
from tqdm import tqdm

# ───────── paths ─────────────────────────────────────────────────────────
from config import ORIGINAL_MAPPING, OUTPUT_FOLDER, METABOLITE_MAP_JSON, SURVEY_AGGREGATED_JSON
from file_utils.common.organoid_patterns import OrganoidNormalizer, norm_key, day_from_key

base_image_mapping_path = ORIGINAL_MAPPING
metabolite_json_path    = METABOLITE_MAP_JSON
survey_json_path        = SURVEY_AGGREGATED_JSON
processed_parent        = Path(OUTPUT_FOLDER)             # where image_mapping_*_processed.json live
output_path             = OUTPUT_FOLDER / "all_data.json"

# ───────── helpers ───────────────────────────────────────────────────────
SPLIT_SUFFIX_RE    = re.compile(r'\bsplit_(\d+)\b', re.IGNORECASE)
STITCHED_SUFFIX_RE = re.compile(r'\bstitched_[A-Za-z0-9_]+\b', re.IGNORECASE)

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

def norm_key_with_suffix(raw_key: str) -> str:
    base = norm_key(raw_key)
    suffixes = []
    m = SPLIT_SUFFIX_RE.search(raw_key)
    if m:
        suffixes.append(f"split_{int(m.group(1))}")
    m2 = STITCHED_SUFFIX_RE.search(raw_key)
    if m2:
        suffixes.append(m2.group(0))
    return base if not suffixes else f"{base} {' '.join(suffixes)}"

def to_mdl_day(day: int | None) -> float | None:
    if day is None:
        return None
    return 20.5 if day in (20, 21) else float(day)

def _extract_infer_res(v: dict) -> str | None:
    """
    Return '512x384', '256x192', ... but ONLY for infer_resized_* paths.
    Ignore legacy processed_dataset_* outputs entirely.
    """
    for field in ("img_path", "mask_path"):
        s = v.get(field)
        if not isinstance(s, str):
            continue
        if "processed_dataset_" in s:
            return None  # skip old pipeline outputs
        # OrganoidNormalizer.extract_resolution should recognize infer_resized_(\d+x\d+)
        res = OrganoidNormalizer.extract_resolution(s)
        if res:
            return res
    return None

# ───────── read base mapping ─────────────────────────────────────────────
base_data = load_json(Path(base_image_mapping_path))
entries_data = base_data.get("entries", base_data)
base_map = {}

for k, v in entries_data.items():
    if not is_organoid_key(k):
        continue
    try:
        base_map[norm_key_with_suffix(k)] = v
    except ValueError as e:
        print(f"[BASE] Failed to normalize: {k} — {e}")

# ───────── read per-resolution (infer-resized ONLY) ─────────────────────
processed_map = {}
for p in processed_parent.rglob("image_mapping_*_processed.json"):
    data = load_json(p)
    for k, v in data.items():
        if not is_organoid_key(k):
            continue
        try:
            norm_k = norm_key_with_suffix(k)
            res = _extract_infer_res(v)   # returns None if not infer_resized_*
            if not res:
                continue
            processed_map.setdefault(norm_k, {})[res] = v
        except ValueError as e:
            print(f"[PROCESSED] Failed to normalize: {k} — {e}")

# ───────── metabolite & survey ───────────────────────────────────────────
metab_map = {}
for k, v in load_json(Path(metabolite_json_path)).items():
    if not is_organoid_key(k):
        continue
    try:
        metab_map[norm_key_with_suffix(k)] = v
    except ValueError as e:
        print(f"[METABOLITE] Failed to normalize: {k} — {e}")

survey_map = {}
for row in load_json(Path(survey_json_path)).values():
    iid = None
    if row.get("evaluations"):
        iid = row["evaluations"][0].get("image_id")
    elif row.get("quality_scores"):
        iid = row["quality_scores"][0].get("image_id")
    if not iid:
        continue
    try:
        survey_map[norm_key_with_suffix(iid)] = row
    except ValueError as e:
        print(f"[SURVEY] Failed to normalize: {iid} — {e}")

# ───────── merge ─────────────────────────────────────────────────────────
all_keys = set().union(base_map, processed_map, survey_map, metab_map)
combined = {}

for k in tqdm(sorted(all_keys)):
    entry = {}
    if k in base_map:      entry.update(base_map[k])
    if k in processed_map: entry.update(processed_map[k])     # adds {"512x384": {...}, ...}
    if k in survey_map:    entry["survey"]      = survey_map[k]
    if k in metab_map:     entry["metabolites"] = metab_map[k]
    _day = day_from_key(k)
    entry["day_num"] = _day
    entry["mdl_day"] = to_mdl_day(_day)
    combined[k] = entry

# ───────── write ─────────────────────────────────────────────────────────
combined_clean = clean_nan_values(combined)
with open(output_path, "w") as f:
    json.dump(combined_clean, f, indent=2)
print(f"Wrote {len(combined):,} merged records → {output_path}")
