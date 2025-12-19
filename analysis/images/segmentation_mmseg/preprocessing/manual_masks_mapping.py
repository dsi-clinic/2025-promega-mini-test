#!/usr/bin/env python3
import argparse
import json
import logging
import re
import sys
from glob import glob
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

# Locate repo root
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

# Constants
ALLOWED_EXT = {".tif", ".tiff", ".png"}
EXPECTED_RECORDS_NUM = 5168

def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Map manual masks to image mapping JSON'
    )
    parser.add_argument(
        '--image-json',
        type=Path,
        help='Path to the image mapping JSON file'
    )
    parser.add_argument(
        '--masks-dir',
        type=Path,
        help='Path to the masks directory'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=None,
        help='Path to the output JSON file'
    )
    args = parser.parse_args()

    # Validate required paths
    if not args.image_json:
        parser.error("--image-json is required")

    if args.output_file is None:
        args.output_file = args.image_json.parent.parent / "masks" / "manual_masks_mapping.json"

    return args

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
    logging.info("[DISCOVER] batch dirs: %s", ", ".join([b.name for b in batch_dirs]))
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
        logging.info("[INFO] %s: %d mask files", name, cnt)
    logging.info("[INFO] total masks: %d", len(files))
    return files

def main():
    args = get_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)

    # Load mapping
    mapping = load_raw_mapping(args.image_json)

    # Filter to Regular and Stitched only (exclude Split)
    filtered_mapping = {
        k: v for k, v in mapping.items()
        if v.get("Classification") in ["Regular", "Stitched"]
    }

    skipped_split = sum(1 for v in mapping.values() if v.get('Classification') == 'Split')
    skipped_other_classification = sum(1 for v in mapping.values()
                                   if v.get('Classification') not in ["Regular", "Stitched", "Split"])    # Classification=SplitStitched, BA2 96_2 Dy30 D12 split_2
    logging.info("[INFO] Total entries in raw mapping: %d", len(mapping))
    logging.info("[INFO] Regular entries: %d", sum(1 for v in mapping.values() if v.get('Classification') == 'Regular'))
    logging.info("[INFO] Stitched entries: %d", sum(1 for v in mapping.values() if v.get('Classification') == 'Stitched'))
    logging.info("[INFO] Split entries (EXCLUDED): %d", skipped_split)
    logging.info("[INFO] Other classification entries (EXCLUDED): %d", skipped_other_classification)
    logging.info("[INFO] Using for mapping: %d", len(filtered_mapping))

    batch_dirs = discover_batch_dirs(args.masks_dir)
    mask_paths = list_mask_files(batch_dirs)

    if not mask_paths:
        logging.error("[FATAL] Found 0 mask files. Check --masks-dir.")
        sys.exit(1)

    new_mapping = {}
    skipped_no_match = 0

    for key, info in filtered_mapping.items():  # Use filtered_mapping
        ba   = info.get('BA')
        day  = info.get('dayID')
        well = info.get('wellID')
        if not (ba and day and well):
            print(f"[WARN] Skipping {key} because of missing BA, day, or well")
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

    actual_records_num = len(new_mapping) + skipped_no_match + skipped_split + skipped_other_classification
    assert actual_records_num == EXPECTED_RECORDS_NUM, f"Expected {EXPECTED_RECORDS_NUM} records, got {actual_records_num}"

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(new_mapping, f, indent=2)

    logging.info("[OK] Saved %d entries to: %s", len(new_mapping), args.output_file)
    logging.info("[INFO] Skipped %d entries with no matching masks", skipped_no_match)
    logging.info("[INFO] Excluded ALL split entries from mapping (due to naming inconsistencies)")

if __name__ == "__main__":
    main()
