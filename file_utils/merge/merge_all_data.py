import os
import json
from glob import glob
from tqdm import tqdm
import re
#!/usr/bin/env python3
import json, os, re, pathlib
from glob import glob
from tqdm import tqdm

# ───────── paths ─────────────────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
base_image_mapping_path = os.getenv("ORIGINAL_MAPPING")
#processed_root_dir = os.getenv("PROCESSED_DATA_DIR")
metabolite_json_path = os.path.join(os.getenv("BASE_PATH"), "metabolite_data", "metabolite_map.json")
survey_json_path = "analysis/surveys/agreement_aggregations/organoid_surveys_aggregated.json"

output_path             = 'all_data.json'

_tok_ba    = re.compile(r'^BA\d+$',          re.IGNORECASE)
_tok_plate = re.compile(r'^(96_[12]|PT1)$',  re.IGNORECASE)
_tok_day   = re.compile(r'^DY\d+$',          re.IGNORECASE)
DAY_NUM_RE = re.compile(r'\bDy(\d{1,2})\b', re.IGNORECASE)

def day_from_key(norm_k: str) -> int | None:
    m = DAY_NUM_RE.search(norm_k)
    return int(m.group(1)) if m else None

def to_mdl_day(day: int | None) -> float | None:
    if day is None:
        return None
    # collapse Dy20 and Dy21 to 20.5
    if day in (20, 21):
        return 20.5
    return float(day)

def norm_key(id_like: str) -> str:
    """
    Normalise an ID of the form
        'Ba2 96_1 Dy30 H11'  -> 'BA2 96_1 Dy30 H11'
        'Ba1 Dy06 A1'        -> 'BA1 Dy06 A1'
    """
    parts = id_like.strip().split()
    if not parts or not _tok_ba.match(parts[0]):
        raise ValueError(f"Bad BA token in {id_like!r}")

    ba      = parts[0].upper()            # BA1, BA2, …
    idx     = 1

    # optional plate designator (96_1 / 96_2 / Pt1)
    plate   = ''
    if idx < len(parts) and _tok_plate.match(parts[idx]):
        plate = parts[idx]
        idx  += 1

    # day token (Dy##)
    if idx >= len(parts) or not _tok_day.match(parts[idx]):
        raise ValueError(f"Cannot find day token in {id_like!r}")
    day = parts[idx]
    idx += 1

    # remaining token is the well ID
    if idx >= len(parts):
        raise ValueError(f"Cannot find well token in {id_like!r}")
    well = parts[idx]

    # final normalised key
    ba_full = f"{ba} {plate}".strip()     # keep plate for BA2 / BA3
    return f"{ba_full} {day} {well}"


# ───────── read files & re-key with norm_key() ───────────────────────────
def load_json(path):
    with open(path) as f: return json.load(f)

base_map     = {norm_key(k): v for k, v in load_json(base_image_mapping_path).items()}
metab_map    = {norm_key(k): v for k, v in load_json(metabolite_json_path).items()}

# # processed – iterate over many small files
# processed_map = {}
# for p in pathlib.Path(processed_root_dir).rglob("image_mapping_*_processed.json"):
#     for k, v in load_json(p).items():
#         processed_map[norm_key(k)] = v

processed_map = {}
processed_parent = os.getenv("PROCESSED_PARENT_DIR")

for p in pathlib.Path(processed_parent).rglob("image_mapping_*_processed.json"):
    if "auto_processed" in str(p):
        for k, v in load_json(p).items():
            norm_k = norm_key(k)
            resolution = re.search(r'processed_dataset_(\d+x\d+)', str(p))
            resolution = resolution.group(1) if resolution else "unknown"
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
with open(output_path, 'w') as f:
    json.dump(combined, f, indent=2)
print(f"Wrote {len(combined):,} merged records → {output_path}")

