#!/usr/bin/env python3
import json, re, sys
from glob import glob
from pathlib import Path

# Locate repo root
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

from config import RAW_IMAGE_MAPPING_JSON, MANUAL_MASKS_DIR, MANUAL_THRESHOLD_MAPPING as OUTPUT_PATH

ALLOWED_EXT = {".tif", ".tiff", ".png"}

def load_raw_mapping(json_path: Path) -> dict:
    data = json.loads(Path(json_path).read_text())
    if isinstance(data, dict) and "_base_folder" in data and "entries" in data:
        base = Path(data["_base_folder"])
        entries = data["entries"]
        for v in entries.values():
            if "Best Z Filename" in v:
                v["Best Z Filename"] = str(base / v["Best Z Filename"])
            if "all_files" in v and isinstance(v["all_files"], list):
                v["all_files"] = [str(base / p) for p in v["all_files"]]
        return entries
    return data

def flex_chunk(s: str) -> str:
    toks = re.findall(r'[A-Za-z0-9]+', (s or "").lower())
    return r'[\W_]*'.join(map(re.escape, toks)) if toks else ''

def discover_batch_dirs(root: Path):
    batch_dirs = [Path(p) for p in glob(str(root / "masks-batch-*")) if Path(p).is_dir()]
    print("[DISCOVER] batch dirs:", [b.name for b in batch_dirs])
    return batch_dirs

def list_mask_files(batch_dirs):
    files = []
    per_batch_counts = []
    for bdir in batch_dirs:
        subdirs = [d for d in (bdir / "manual", bdir / "threshold") if d.is_dir()]
        cnt = 0
        for sd in subdirs:
            for f in sd.rglob("*"):
                if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
                    files.append(f)
                    cnt += 1
        per_batch_counts.append((bdir.name, cnt))
    for name, cnt in per_batch_counts:
        print(f"[INFO] {name}: {cnt} mask files")
    print(f"[INFO] total masks: {len(files)}")
    return files

# Load mapping
mapping = load_raw_mapping(RAW_IMAGE_MAPPING_JSON)

# Filter to Regular and Stitched only (exclude Split)
filtered_mapping = {
    k: v for k, v in mapping.items() 
    if v.get("Classification") in ["Regular", "Stitched"]
}

print(f"[INFO] Total entries in raw mapping: {len(mapping)}")
print(f"[INFO] Regular entries: {sum(1 for v in mapping.values() if v.get('Classification') == 'Regular')}")
print(f"[INFO] Stitched entries: {sum(1 for v in mapping.values() if v.get('Classification') == 'Stitched')}")
print(f"[INFO] Split entries (EXCLUDED): {sum(1 for v in mapping.values() if v.get('Classification') == 'Split')}")
print(f"[INFO] Using for mapping: {len(filtered_mapping)}")

batch_dirs = discover_batch_dirs(Path(MANUAL_MASKS_DIR))
mask_paths = list_mask_files(batch_dirs)

if not mask_paths:
    print("[FATAL] Found 0 mask files. Check MANUAL_MASKS_DIR.")
    sys.exit(1)

new_mapping = {}
skipped_no_match = 0

for key, info in filtered_mapping.items():  # Use filtered_mapping
    ba   = info.get('BA')
    day  = info.get('dayID')
    well = info.get('wellID')
    if not (ba and day and well):
        continue

    # Build flexible patterns that handle old naming variations
    ba_pat  = flex_chunk(ba)

    m = re.search(r'(\d+)', day or "")
    if m:
        day_num = int(m.group(1))
        day_pat = rf'(?:dy|day)[\W_]*0*{day_num}(?!\d)'
    else:
        day_pat = flex_chunk(day)

    wl = well[0].lower()
    wn = int(well[1:])
    # Match both "D11" and "D11(#)" or "D11(1)%" patterns
    well_pat = rf'(?<![a-z0-9]){wl}0?{wn}(?:\([^)]*\))?(?!\d)'

    best_z = info.get('Best Z')
    def score(s: str) -> int:
        s = s.lower()
        pts = 0
        if re.search(rf'(?<![a-z0-9]){wl}{wn}(?!\d)', s): pts += 2
        if best_z is not None and re.search(rf'(?<!\d){best_z}(?!\d)', s): pts += 1
        return pts

    matches = []
    for p in mask_paths:
        s = str(p).lower()
        if re.search(ba_pat, s) and re.search(day_pat, s) and re.search(well_pat, s):
            matches.append(p)

    if matches:
        matches.sort(key=lambda p: score(str(p)), reverse=True)
        mt_path = str(matches[0].resolve())
        new_mapping[key] = {
            "dayID": info.get("dayID"),
            "BA": info.get("BA"),
            "wellID": info.get("wellID"),
            "Best Z Filename": info.get("Best Z Filename"),
            "MT Mask Path": mt_path,
        }
    else:
        skipped_no_match += 1

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(new_mapping, f, indent=2)

print(f"[OK] Saved {len(new_mapping)} entries to: {OUTPUT_PATH}")
print(f"[INFO] Skipped {skipped_no_match} entries with no matching masks")
print(f"[INFO] Excluded ALL split entries from mapping (due to naming inconsistencies)")