#!/usr/bin/env python3
import json, re, csv
from pathlib import Path
from tqdm import tqdm

from config import (
    ORIGINAL_MAPPING,
    INFER_RESIZED_DIR,
    METABOLITE_MAP_JSON,
    SURVEY_AGGREGATED_JSON,
    ALL_DATA_JSON,
    IMAGE_VERIFICATION_FORM,  # optional CSV with "main id"
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
    Normalize IDs for consistent comparisons:
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

def parse_norm_key(norm_k: str):
    """
    From a normalized key 'BA[ plate]? Dy## Well' return (ba, plate, dayID, well)
    Example: 'BA2 96_1 Dy28 B3' -> ('BA2', '96_1', 'Dy28', 'B3')
             'BA1 Dy03 A1'      -> ('BA1', '',     'Dy03', 'A1')
    """
    parts = norm_k.split()
    if not parts: return "", "", "", ""
    ba = parts[0]
    i = 1
    plate = ""
    # plate present only if we still have at least 2 tokens for day/well
    if i < len(parts) - 1 and _tok_plate.match(parts[i]):
        plate = parts[i]; i += 1
    dayID = parts[i] if i < len(parts) else ""
    i += 1
    well = parts[i] if i < len(parts) else ""
    return ba, plate, dayID, well

def day_from_key(k: str):
    m = DAY_NUM_RE.search(k)
    return int(m.group(1)) if m else None

def to_mdl_day(d):
    if d is None: return None
    return 20.5 if d in (20, 21) else float(d)

def compute_first_split_days(base_entries: dict) -> dict:
    """
    Earliest split day per (BA, plate, well) using ONLY ORIGINAL_MAPPING.
    We consider a day 'split' if either:
      - 'split_' appears in the entry key, OR
      - entry has a non-None 'split_index' field.
    Returns {(BA, plate, well): first_split_day_num}
    """
    first_split = {}

    for raw_k, rec in base_entries.items():
        # detect split from key or value
        has_split = ("split_" in raw_k.lower()) or (
            isinstance(rec, dict) and rec.get("split_index") is not None
        )
        if not has_split:
            continue

        nk = norm_key(raw_k)                 # 'BA[ plate]? Dy## Well'
        ba, plate, dayID, well = parse_norm_key(nk)
        dnum = day_from_key(dayID)
        if dnum is None:
            continue

        key = (ba, plate, well)
        cur = first_split.get(key)
        first_split[key] = dnum if cur is None else min(cur, dnum)

    return first_split

def make_common_key(raw_key: str, norm_k: str, payload: dict, first_split_map: dict):
    """
    Build BA[_plate]_DyXX_Well_{split|nosplit|presplit}_{stitched|nostitch}
    - BA stays uppercase
    - presplit: if this well splits later, and this day < first split day
    """
    ba, plate, dayID, well = parse_norm_key(norm_k)
    ba_part = f"{ba}_{plate}" if plate else ba

    # split detection
    split_match = SPLIT_TOKEN.search(raw_key)
    if split_match:
        split_str = f"split{split_match.group(1)}"
    else:
        # presplit if this day is earlier than the well's first split day
        dnum = day_from_key(dayID)
        fsd = first_split_map.get((ba, plate, well))
        if (dnum is not None) and (fsd is not None) and (dnum < fsd):
            split_str = "presplit"
        else:
            split_str = "nosplit"

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

    return f"{ba_part}_{dayID}_{well}_{split_str}_{stitched_str}"

def load_verification_csv(csv_path: str | Path) -> set[str]:
    """
    Optional: read lab CSV with a 'main id' column (case-insensitive).
    Returns set of main ids as-is (no presplit->nosplit remapping).
    """
    p = Path(csv_path)
    if not p.exists():
        print(f"⚠️  Verification CSV not found: {p}")
        return set()

    main_ids = set()
    with p.open(newline="") as f:
        reader = csv.DictReader(f)

        def get_cell(row, *names):
            for n in names:
                for key in row.keys():
                    if key.strip().lower() == n.strip().lower():
                        return row[key]
            return None

        raw_rows = 0
        for row in reader:
            raw_rows += 1
            main_id = get_cell(row, "main id", "main_id", "common_key")
            if main_id:
                main_ids.add(main_id.strip())

    print(f"Loaded {len(main_ids)} verification 'main id' rows from {p}")
    return main_ids

# ───────── Load sources ─────────
base_json   = load_json(ORIGINAL_MAPPING)
base_map    = base_json.get("entries", {})   # all raw entries

# first pass: compute earliest split day per (BA, plate, well)
first_split_map = compute_first_split_days(base_map)

metab_map   = {norm_key(k): v for k,v in load_json(METABOLITE_MAP_JSON).items()}

# processed (keys may include split/stitched)
processed_map = {}
found_files = list(Path(INFER_RESIZED_DIR).rglob("image_mapping*_processed.json"))
print(f"Found {len(found_files)} processed JSONs under {INFER_RESIZED_DIR}")

for p in found_files:
    raw = load_json(p)
    print(f"  - {p} ({len(raw)} entries)")
    for k, v in raw.items():
        processed_map[k] = v  # raw keys include split variants

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
for raw_k, payload in tqdm(base_map.items(), desc="Merging"):
    entry = dict(payload)

    # processed lookup:
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

    # add common key with presplit logic
    entry["common_key"] = make_common_key(raw_k, norm_key(raw_k), payload, first_split_map)

    # day helpers (if you still want them)
    dnum = day_from_key(entry.get("dayID") or norm_key(raw_k))
    entry["day_num"] = dnum
    entry["mdl_day"] = to_mdl_day(dnum)

    combined[raw_k] = entry

# ───────── Optional: verification against lab CSV ─────────
csv_ids = set()
if IMAGE_VERIFICATION_FORM:
    try:
        csv_ids = load_verification_csv(IMAGE_VERIFICATION_FORM)
    except Exception as e:
        print(f"⚠️  Failed to load verification CSV: {e}")

if csv_ids:
    merged_keys = {v.get("common_key") for v in combined.values()
                   if isinstance(v, dict) and "common_key" in v}
    missing_in_data = sorted(k for k in csv_ids if k not in merged_keys)
    extra_in_data   = sorted(k for k in merged_keys if k not in csv_ids)

    print("\nVerification cross-check")
    print(f"  CSV main ids          : {len(csv_ids)}")
    print(f"  merged common_keys    : {len(merged_keys)}")
    print(f"  CSV ids missing in merged data : {len(missing_in_data)}")
    print(f"  merged keys not in CSV        : {len(extra_in_data)} (OK if CSV is a subset)")

    if missing_in_data:
        from pathlib import Path
        report = Path("/tmp/verification_missing_in_data.json")
        with report.open("w") as f:
            json.dump(missing_in_data, f, indent=2)
        raise AssertionError(
            f"Some CSV 'main id' values were not found in merged data "
            f"(count={len(missing_in_data)}). Wrote details to {report}. "
            f"Examples: {missing_in_data[:10]}"
        )

# ───────── Write out ─────────
with open(OUTPUT_PATH,"w") as f:
    json.dump(combined, f, indent=2)
print(f"Wrote {len(combined):,} merged records → {OUTPUT_PATH}")
