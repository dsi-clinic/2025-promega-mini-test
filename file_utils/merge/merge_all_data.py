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
    ALL_DATA_JSON,
)

OUTPUT_PATH = str(ALL_DATA_JSON)

# ---------- helpers ----------
def load_json(path: Path | str):
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

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



# ---------- load processed JSONs ----------
processed_map = {}
found_files = list(Path(INFER_RESIZED_DIR).rglob("image_mapping*_processed.json"))

for p in found_files:
    raw = load_json(p)
    processed_map.update(raw)

# ---------- merge ----------
combined = {}
for raw_k, payload in tqdm(base_map.items(), desc="Merging"):
    entry = dict(payload)

    processed = processed_map.get(raw_k) or processed_map.get(normalized_parent_key(raw_k))
    if processed:
        entry["processed"] = processed
        entry["main_id"] = processed.get("main_id")

    norm_key_parent = normalized_parent_key(raw_k)
    if norm_key_parent in survey_map:
        entry["survey"] = survey_map[norm_key_parent]

    if norm_key_parent in metab_map:
        entry["metabolites"] = metab_map[norm_key_parent]

    combined[raw_k] = entry


# ---------- write output ----------
with open(OUTPUT_PATH, "w") as f:
    json.dump(combined, f, indent=2)

print(f"\nWrote {len(combined):,} merged records → {OUTPUT_PATH}")
