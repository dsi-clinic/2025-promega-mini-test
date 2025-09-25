#!/usr/bin/env python3
import json, re
from pathlib import Path
from tqdm import tqdm

from config import (
    ORIGINAL_MAPPING,
    INFER_RESIZED_DIR,
    METABOLITE_MAP_JSON,
    SURVEY_AGGREGATED_JSON,
    ALL_DATA_JSON
)

OUTPUT_PATH = str(ALL_DATA_JSON)

# regex helpers
_tok_ba    = re.compile(r"^BA\d+$", re.IGNORECASE)
_tok_plate = re.compile(r"^(96_[12]|PT1)$", re.IGNORECASE)
_tok_day   = re.compile(r"^DY\d+$", re.IGNORECASE)
DAY_NUM_RE = re.compile(r"\bDy(\d{1,2})\b", re.IGNORECASE)
SPLIT_TOKEN = re.compile(r"\bsplit_(\d+)\b", re.IGNORECASE)


def load_json(path):
    with open(path) as f:
        return json.load(f)

def parent_key(id_like: str) -> str:
    """Strip split/stitched from an ID for parent-level lookups."""
    k = SPLIT_TOKEN.sub("", id_like)
    k = re.sub(r"\(?stitched\)?", "", k, flags=re.IGNORECASE)
    return " ".join(k.split()).strip()

def norm_key(id_like: str) -> str:
    """
    Normalize IDs for consistent parent-level comparisons.
        'Ba2 96_1 Dy30 H11' -> 'BA2 96_1 Dy30 H11'
    """
    parts = id_like.strip().split()
    if not parts or not _tok_ba.match(parts[0]):
        raise ValueError(f"Bad BA token in {id_like!r}")

    ba = parts[0].upper()
    idx = 1

    plate = ""
    if idx < len(parts) and _tok_plate.match(parts[idx]):
        plate = parts[idx]; idx += 1

    if idx >= len(parts) or not _tok_day.match(parts[idx]):
        raise ValueError(f"Cannot find day token in {id_like!r}")
    day = parts[idx]; idx += 1

    if idx >= len(parts):
        raise ValueError(f"Cannot find well token in {id_like!r}")
    well = parts[idx]

    ba_full = f"{ba} {plate}".strip()
    return f"{ba_full} {day} {well}"

def day_from_key(k: str):
    m = DAY_NUM_RE.search(k)
    return int(m.group(1)) if m else None

def to_mdl_day(d):
    if d is None: return None
    return 20.5 if d in (20, 21) else float(d)

def make_common_key(raw_key: str, norm_k: str, payload: dict):
    """
    Build ba[_plate]_DyXX_Well_splitX_stitched
    - 'ba' lowercased (e.g., 'ba2'); plate kept if present (e.g., '96_1')
    """
    parts = norm_k.split()
    ba_tok = parts[0].lower() if parts else ""  # <-- lowercased BA
    plate_tok = parts[1] if (len(parts) >= 4 and _tok_plate.match(parts[1])) else ""
    day  = parts[-2] if len(parts) >= 2 else ""
    well = parts[-1] if len(parts) >= 1 else ""

    ba_part = f"{ba_tok}_{plate_tok}" if plate_tok else ba_tok

    # split detection
    split_match = SPLIT_TOKEN.search(raw_key)
    split_str = f"split{split_match.group(1)}" if split_match else "nosplit"

    # stitched detection
    stitched_str = "nostitch"
    if "stitched" in raw_key.lower():
        stitched_str = "stitched"
    elif payload.get("Classification", "").lower() == "stitched":
        stitched_str = "stitched"
    elif any("stitched" in str(v).lower() for v in payload.get("all_files", [])):
        stitched_str = "stitched"
    elif "stitched" in str(payload.get("Best Z Filename", "")).lower():
        stitched_str = "stitched"

    return f"{ba_part}_{day}_{well}_{split_str}_{stitched_str}"


# ───────── Load sources ─────────
base_json   = load_json(ORIGINAL_MAPPING)
base_map    = base_json.get("entries", {})   # all raw entries

metab_map   = {norm_key(k): v for k,v in load_json(METABOLITE_MAP_JSON).items()}

# processed (keys may include split/stitched; store per-resolution)
processed_map = {}
found_files = list(Path(INFER_RESIZED_DIR).rglob("image_mapping*_processed.json"))
print(f"Found {len(found_files)} processed JSONs under {INFER_RESIZED_DIR}")

for p in found_files:
    raw = load_json(p)
    print(f"  - {p} ({len(raw)} entries)")
    for k, v in raw.items():
        # use the raw key directly, e.g. "BA2 96_1 Dy28 C1 split_1"
        processed_map[k] = v

# surveys
survey_map = {}
survey_json = load_json(SURVEY_AGGREGATED_JSON)
for row in survey_json.values():
    ids = []
    if row.get("evaluations"):
        ids += [ev["image_id"] for ev in row["evaluations"] if "image_id" in ev]
    if row.get("quality_scores"):
        ids += [qs["image_id"] for qs in row["quality_scores"] if "image_id" in qs]
    for iid in ids:
        survey_map[iid] = row
        survey_map[parent_key(iid)] = row   # also allow parent-level match


# ───────── Merge ─────────
combined = {}
for raw_k, payload in base_map.items():
    entry = dict(payload)

    # processed lookup:
    # first try exact (works for splits + normal wells)
    if raw_k in processed_map:
        entry["processed"] = processed_map[raw_k]
    else:
        # stitched fallback: strip "(stitched)" only
        pk_stitchless = re.sub(r"\(stitched\)", "", raw_k, flags=re.IGNORECASE).strip()
        if pk_stitchless in processed_map:
            entry["processed"] = processed_map[pk_stitchless]

    # survey lookup: parent key (strip split + stitched)
    pk_parent = parent_key(raw_k)
    if pk_parent in survey_map:
        entry["survey"] = survey_map[pk_parent]

    # metabolite lookup: parent key only
    if pk_parent in metab_map:
        entry["metabolites"] = metab_map[pk_parent]

    # add common key
    entry["common_key"] = make_common_key(raw_k, norm_key(raw_k), payload)

    combined[raw_k] = entry



# ───────── Write out ─────────
with open(OUTPUT_PATH,"w") as f:
    json.dump(combined, f, indent=2)
print(f"Wrote {len(combined):,} merged records → {OUTPUT_PATH}")
