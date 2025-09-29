#!/usr/bin/env python3
import json, re
from pathlib import Path
from tqdm import tqdm

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

def parent_key(id_like: str) -> str:
    """Strip split/stitched tags for parent-level lookups."""
    return re.sub(r"\bsplit_\d+\b|\(stitched\)", "", id_like, flags=re.IGNORECASE).strip()

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
        survey_map[iid] = row
        survey_map[parent_key(iid)] = row

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

    # processed lookup
    processed = processed_map.get(raw_k)
    if not processed:
        pk = parent_key(raw_k)
        processed = processed_map.get(pk)
    if processed:
        entry["processed"] = processed
        entry["main_id"] = processed.get("main_id")

    # survey lookup
    pk_parent = parent_key(raw_k)
    if pk_parent in survey_map:
        entry["survey"] = survey_map[pk_parent]

    # metabolite lookup
    if pk_parent in metab_map:
        entry["metabolites"] = metab_map[pk_parent]

    combined[raw_k] = entry

# ---------- write output ----------
with open(OUTPUT_PATH, "w") as f:
    json.dump(combined, f, indent=2)

print(f"\nWrote {len(combined):,} merged records → {OUTPUT_PATH}")
