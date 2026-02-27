#!/usr/bin/env python3
import argparse
import json
import logging
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=logging.INFO
)

ALLOWED_EXT = {".tif", ".tiff", ".png"}
EXPECTED_RECORDS_NUM = 5168

# Interp choices (match old working script)
MASK_INTERP = cv2.INTER_NEAREST

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Map + preprocess manual masks')

    parser.add_argument('--image-json', type=Path, required=True,
                        help='Path to the image mapping JSON file (the one with processed_image + verification.blank).')
    parser.add_argument('--masks-dir', type=Path, required=True,
                        help='Root of masks dir containing masks-batch-* folders.')
    parser.add_argument('--output-file', type=Path, default=None,
                        help='Output JSON file. Default: <image_json>/../masks/manual_masks_mapping.json')

    # NEW: preprocessing options
    parser.add_argument('--target-width', type=int, default=512)
    parser.add_argument('--target-height', type=int, default=384)

    parser.add_argument('--processed-masks-dir', type=Path, default=None,
                        help=('Where to write resized/binarized masks. '
                              'Default: <output_file parent>/manual_processed_<WxH>/'))

    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite already-written processed masks.')

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.image_json.parent.parent / "masks" / "manual_masks_mapping.json"

    if args.processed_masks_dir is None:
        args.processed_masks_dir = args.output_file.parent / f"manual_processed_{args.target_width}x{args.target_height}"

    return args

def load_raw_mapping(json_path: Path) -> Dict[str, Dict]:
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

def discover_batch_dirs(root: Path) -> List[Path]:
    batch_dirs = [Path(p) for p in glob(str(root / "masks-batch-*")) if Path(p).is_dir()]
    logging.info("[DISCOVER] batch dirs: %s", ", ".join([b.name for b in batch_dirs]))
    return batch_dirs

def list_mask_files(batch_dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
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

def get_score(s: str, well: str, info: Dict[str, Any]) -> int:
    wn = int(well[1:])
    best_z = info.get('Best Z')
    s = s.lower()
    pts = 0
    # NOTE: this logic looks a bit odd but leaving as you had it
    if re.search(rf'(?<![a-z0-9]){well}{wn}(?!\d)', s):
        pts += 2
    if best_z is not None and re.search(rf'(?<!\d){best_z}(?!\d)', s):
        pts += 1
    return pts

def read_mask_grayscale(path: Path) -> Optional[np.ndarray]:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return arr

def write_processed_mask(
    raw_mask_path: Path,
    out_path: Path,
    target_wh: Tuple[int, int],
    overwrite: bool
) -> Path:
    """
    Reads raw mask, resizes (NEAREST), binarizes to {0,1}, writes uint8 PNG/TIF.
    Returns out_path.
    """
    if out_path.exists() and not overwrite:
        return out_path

    msk = read_mask_grayscale(raw_mask_path)
    if msk is None:
        raise RuntimeError(f"Failed to read mask: {raw_mask_path}")

    tw, th = target_wh  # width, height
    msk_rs = cv2.resize(msk, (tw, th), interpolation=MASK_INTERP)
    msk_bin = (msk_rs > 0).astype(np.uint8)  # 0/1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), msk_bin)
    if not ok:
        raise RuntimeError(f"Failed to write processed mask: {out_path}")
    return out_path

def write_blank_mask(out_path: Path, target_wh: Tuple[int, int], overwrite: bool) -> Path:
    if out_path.exists() and not overwrite:
        return out_path
    tw, th = target_wh
    blank = np.zeros((th, tw), dtype=np.uint8)  # (H,W)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), blank)
    if not ok:
        raise RuntimeError(f"Failed to write blank mask: {out_path}")
    return out_path

def main() -> None:
    args = get_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)

    mapping = load_raw_mapping(args.image_json)

    # Filter to Regular + Stitched; exclude Split from mask matching
    filtered_mapping = {
        k: v for k, v in mapping.items()
        if v.get("Classification") in ["Regular", "Stitched"]
    }

    skipped_split = sum(1 for v in mapping.values() if v.get('Classification') == 'Split')
    skipped_other_classification = sum(
        1 for v in mapping.values()
        if v.get('Classification') not in ["Regular", "Stitched", "Split"]
    )

    logging.info("[INFO] Total entries in raw mapping: %d", len(mapping))
    logging.info("[INFO] Using for mask matching: %d", len(filtered_mapping))
    logging.info("[INFO] Split entries (excluded from mask matching): %d", skipped_split)
    logging.info("[INFO] Other classification entries (excluded): %d", skipped_other_classification)

    batch_dirs = discover_batch_dirs(args.masks_dir)
    mask_paths = list_mask_files(batch_dirs)
    if not mask_paths:
        logging.error("[FATAL] Found 0 mask files. Check --masks-dir.")
        sys.exit(1)

    target_wh = (args.target_width, args.target_height)
    processed_masks_dir = args.processed_masks_dir
    processed_masks_dir.mkdir(parents=True, exist_ok=True)

    new_mapping: Dict[str, Dict[str, Any]] = {}
    skipped_no_match = 0
    processed_written = 0
    blanks_added = 0

    # ---- SINGLE PASS: Regular/Stitched records
    for record_id, info in filtered_mapping.items():
        ba = info.get('BA')
        day = info.get('dayID')
        well = info.get('wellID')
        if not (ba and day and well):
            logging.warning("[WARN] Skipping %s because missing BA/day/well", record_id)
            skipped_no_match += 1
            continue

        # --- BLANK HANDLING FIRST ---
        is_blank = bool(info.get("verification", {}).get("blank", False))
        if is_blank:
            out_mask_path = (processed_masks_dir / "masks" / f"{record_id}_mask.png").resolve()
            try:
                write_blank_mask(out_mask_path, target_wh, overwrite=args.overwrite)
            except Exception as e:
                logging.error("[ERROR] Failed writing blank mask for %s: %s", record_id, e)
                skipped_no_match += 1
                continue

            new_mapping[record_id] = {
                "dayID": info.get("dayID"),
                "BA": info.get("BA"),
                "wellID": info.get("wellID"),
                "Best Z Filename": info.get("Best Z Filename"),
                "processed_image": info.get("processed_image"),
                "manual_mask_path": str(out_mask_path),
                "manual_mask_path_original": None,
                "blank": True,
            }
            blanks_added += 1
            continue

        # --- NON-BLANK: match a real mask file ---
        ba_pat = flex_chunk(ba)

        m = re.search(r'(\d+)', day or "")
        if m:
            day_num = int(m.group(1))
            day_pat = rf'(?:dy|day)[\W_]*0*{day_num}(?!\d)'
        else:
            day_pat = flex_chunk(day)

        wl = well[0].lower()
        wn = int(well[1:])
        well_pat = rf'(?<![a-z0-9]){wl}0?{wn}(?:\([^)]*\))?(?!\d)'

        matches: List[Path] = []
        for p in mask_paths:
            s = str(p).lower()
            if re.search(ba_pat, s) and re.search(day_pat, s) and re.search(well_pat, s):
                matches.append(p)

        if not matches:
            skipped_no_match += 1
            continue

        matches.sort(key=lambda p: get_score(str(p), well, info), reverse=True)
        raw_mask_path = matches[0].resolve()

        out_mask_path = (processed_masks_dir / "masks" / f"{record_id}_mask.png").resolve()
        try:
            write_processed_mask(raw_mask_path, out_mask_path, target_wh, overwrite=args.overwrite)
            processed_written += 1
        except Exception as e:
            logging.error("[ERROR] Failed processing %s mask %s: %s", record_id, raw_mask_path, e)
            skipped_no_match += 1
            continue

        new_mapping[record_id] = {
            "dayID": info.get("dayID"),
            "BA": info.get("BA"),
            "wellID": info.get("wellID"),
            "Best Z Filename": info.get("Best Z Filename"),
            "processed_image": info.get("processed_image"),
            "manual_mask_path": str(out_mask_path),
            "manual_mask_path_original": str(raw_mask_path),
            "blank": False,
        }

    # Accounting: every filtered record is either mapped or counted as no-match
    accounted = len(new_mapping) + skipped_no_match
    expected = len(filtered_mapping)
    assert accounted == expected, f"accounted {accounted} != expected {expected}"


    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(new_mapping, f, indent=2)

    logging.info("[OK] Saved %d entries to: %s", len(new_mapping), args.output_file)
    logging.info("[INFO] Processed masks written (non-blank): %d", processed_written)
    logging.info("[INFO] Blanks added: %d", blanks_added)
    logging.info("[INFO] Skipped %d entries with no matching/processable masks", skipped_no_match)
    logging.info("[INFO] Processed masks dir: %s", processed_masks_dir)

if __name__ == "__main__":
    main()
