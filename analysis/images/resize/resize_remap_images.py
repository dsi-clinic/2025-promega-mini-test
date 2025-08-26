#!/usr/bin/env python3
import json
import argparse
import shutil
import sys
from pathlib import Path
import cv2

HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

from config import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    TARGET_SIZE,
    RAW_IMAGE_MAPPING_JSON,            # <-- use this
    INFER_AUTO_PROCESSED_DIR as OUTPUT_DIR
)

# --------------- configuration -----------------
INTERPOLATION = cv2.INTER_LINEAR

# --------------- helpers -----------------------
def norm(s: str) -> str:
    return s.lower().replace(' ', '') if isinstance(s, str) else ''

def ba_match(json_ba: str, batch_id: str) -> bool:
    return norm(json_ba).startswith(norm(batch_id))

# --------------- main functions ----------------
def process_batch(batch_num: int, day_num: int):
    # was: with ORIGINAL_MAPPING.open() as f:
    with RAW_IMAGE_MAPPING_JSON.open() as f:
        mapping = json.load(f)

    # Handle all BA* matches in mapping
    all_ba_keys = set(norm(info['BA']) for info in mapping.values() if 'BA' in info)
    batch_prefix = norm(f"BA{batch_num}")
    batch_ids = sorted(b for b in all_ba_keys if b.startswith(batch_prefix))

    for batch_id in batch_ids:
        create_mapping(mapping, batch_id, day_num)

def create_mapping(mapping: dict, batch_id: str, day_num: int):
    day_id     = f"Dy{day_num:02d}"
    safe_batch = batch_id.replace(' ', '_')
    output_dir = OUTPUT_DIR / f"{safe_batch}_{day_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"image_mapping_{safe_batch}_{day_id}_processed.json"
    if output_json.exists():
        print(f"Mapping exists: {output_json}")
        return

    # select images for this batch/day
    matches = [
        (img_id, info['Best Z Filename'], info.get('um_per_px'))
        for img_id, info in mapping.items()
        if norm(info.get('dayID', '')) == norm(day_id)
        and ba_match(info.get('BA', ''), batch_id)
    ]
    print(f"BA/day filter found {len(matches)} candidates.")

    new_mapping = {}
    for img_id, img_path_str, orig_um_px in matches:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"Skipped missing: {img_path}")
            continue

        img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_raw is None:
            print(f"Skipped unreadable: {img_path}")
            continue

        orig_h, orig_w = img_raw.shape[:2]
        img_final = cv2.resize(img_raw, TARGET_SIZE, INTERPOLATION)

        um_x = um_y = None
        final_um_px_x = final_um_px_y = None
        if orig_um_px is not None:
            # handle single float or (x, y) tuple
            if isinstance(orig_um_px, (list, tuple)) and len(orig_um_px) == 2:
                um_x, um_y = orig_um_px
            else:
                um_x = um_y = orig_um_px

            scale_x = orig_w / TARGET_WIDTH
            scale_y = orig_h / TARGET_HEIGHT
            final_um_px_x = um_x * scale_x
            final_um_px_y = um_y * scale_y

        out_path = output_dir / f"{img_id.replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), img_final)

        new_mapping[img_id] = {
            "img_path":            str(out_path),
            "orig_width_px":       orig_w,
            "orig_height_px":      orig_h,
            "orig_um_per_px_x":    um_x,
            "orig_um_per_px_y":    um_y,
            "final_um_per_px_x":   final_um_px_x,
            "final_um_per_px_y":   final_um_px_y
        }

    with output_json.open('w') as f:
        json.dump(new_mapping, f, indent=2)

    print(f"Created mapping for {batch_id} with {len(new_mapping)} images")
    print(f"Images saved to: {output_dir}")
    print(f"Mapping saved to: {output_json}")

# --------------- CLI entrypoint ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batches',
        required=True,
        help='Comma-separated batch numbers, e.g. 1,2,3'
    )
    parser.add_argument(
        '--days',
        required=True,
        help='Comma-separated day numbers, e.g. 3,6,8'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove existing processed folders first'
    )
    args = parser.parse_args()

    batches = [int(x) for x in args.batches.split(',')]
    days    = [int(x) for x in args.days.split(',')]

    for batch in batches:
        for day in days:
            print(f"\nProcessing Batch {batch}, Day {day}")
            if args.overwrite:
                sub_ids = ["96_1","96_2"] if batch == 2 else [None]
                for sub in sub_ids:
                    fname = (
                        f"BA{batch}_{sub}_Dy{day:02d}"
                        if sub else f"BA{batch}_Dy{day:02d}"
                    )
                    folder = OUTPUT_DIR / fname
                    if folder.exists():
                        shutil.rmtree(folder)
                        print(f"  Removed old folder {folder}")
            process_batch(batch, day)

    print("\nAll done.")
