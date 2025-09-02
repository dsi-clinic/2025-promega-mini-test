#!/usr/bin/env python3
import json, re, sys
from glob import glob
from pathlib import Path

# Locate repo root: must contain both paths.py and .env
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

# Import ONLY root paths
from config import RAW_IMAGE_MAPPING_JSON, MANUAL_MASKS_DIR, MANUAL_THRESHOLD_MAPPING as OUTPUT_PATH

ALLOWED_EXT = {".tif", ".tiff", ".png"}

def load_raw_mapping(json_path: Path) -> dict:
    data = json.loads(Path(json_path).read_text())
    # New wrapped format?
    if isinstance(data, dict) and "_base_folder" in data and "entries" in data:
        base = Path(data["_base_folder"])
        entries = data["entries"]
        # Re-hydrate relative paths to absolute strings for consumers
        for v in entries.values():
            if "Best Z Filename" in v:
                v["Best Z Filename"] = str(base / v["Best Z Filename"])
            if "all_files" in v and isinstance(v["all_files"], list):
                v["all_files"] = [str(base / p) for p in v["all_files"]]
        return entries
    # Legacy flat mapping
    return data

def flex_chunk(s: str) -> str:
    toks = re.findall(r'[A-Za-z0-9]+', (s or "").lower())
    return r'[\W_]*'.join(map(re.escape, toks)) if toks else ''

def discover_batch_dirs(root: Path):
    # Find masks-batch-* at the correct level
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
                    files.append(f)     # full path
                    cnt += 1
        per_batch_counts.append((bdir.name, cnt))
    for name, cnt in per_batch_counts:
        print(f"[INFO] {name}: {cnt} mask files")
    print(f"[INFO] total masks: {len(files)}")
    return files

# --- load once ---
mapping = load_raw_mapping(RAW_IMAGE_MAPPING_JSON)

batch_dirs = discover_batch_dirs(Path(MANUAL_MASKS_DIR))
mask_paths = list_mask_files(batch_dirs)

if not mask_paths:
    print("[FATAL] Found 0 mask files. Check MANUAL_MASKS_DIR.")
    sys.exit(1)

new_mapping = {}

for key, info in mapping.items():
    ba   = info.get('BA')
    day  = info.get('dayID')
    well = info.get('wellID')
    if not (ba and day and well):
        continue

    ba_pat  = flex_chunk(ba)

    m = re.search(r'(\d+)', day or "")
    if m:
        day_num = int(m.group(1))
        day_pat = rf'(?:dy|day)[\W_]*0*{day_num}(?!\d)'
    else:
        day_pat = flex_chunk(day)

    wl = well[0].lower()
    wn = int(well[1:])
    well_pat = rf'(?<![a-z0-9]){wl}0?{wn}(?!\d)'

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

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(new_mapping, f, indent=2)
print(f"[OK] Saved {len(new_mapping)} entries to: {OUTPUT_PATH}")
