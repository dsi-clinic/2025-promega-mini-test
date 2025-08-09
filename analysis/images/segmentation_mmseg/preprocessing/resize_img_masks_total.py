#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from paths import TARGET_SIZE, PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR, MANUAL_THRESHOLD_MAPPING, ORIGINAL_MAPPING

# ======== HARD-CODED CONFIG ========
IMAGE_INTERP  = cv2.INTER_LINEAR
MASK_INTERP   = cv2.INTER_NEAREST

# ======== PARSE ARGS ========
parser = argparse.ArgumentParser(
    description="Resize images and masks for training from day-specific mappings."
)
parser.add_argument(
    '--mapping',
    nargs='+',
    default=[str(MANUAL_THRESHOLD_MAPPING)],
    help='One or more JSON mapping files (manual masks).'
)
# NEW: image mapping with 'Blank' flag
parser.add_argument(
    '--image-mapping',
    default=str(ORIGINAL_MAPPING),  # <- default to your env path
    help='Path to image mapping JSON that contains "Blank" flags.'
)
parser.add_argument(
    '--output',
    default=None,
    help='Optional output base folder. Defaults to sibling of first JSON: processed_<WxH>.'
)

args = parser.parse_args()

# ======== SET UP OUTPUT FOLDERS ========
first_json = Path(args.mapping[0])
out_base = Path(args.output) if args.output else PROCESSED_IMAGES_DIR.parent
images_out = PROCESSED_IMAGES_DIR
masks_out  = PROCESSED_MASKS_DIR

for d in (out_base, images_out, masks_out):
    d.mkdir(parents=True, exist_ok=True)
print(f"Writing processed data to: {out_base}")

# ======== LOAD & MERGE MANUAL MASK MAPPINGS ========
master_map = {}
for jm in args.mapping:
    jm = Path(jm)
    if not jm.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {jm}")
    master_map.update(json.loads(jm.read_text()))
print(f"Loaded {len(master_map)} manual-mask entries from {len(args.mapping)} JSON(s)")

# ======== (OPTIONAL) LOAD IMAGE MAPPING WITH BLANK FLAGS ========
image_map = {}
if args.image_mapping:
    imap_p = Path(args.image_mapping)
    if not imap_p.exists():
        raise FileNotFoundError(f"Image mapping JSON not found: {imap_p}")
    image_map = json.loads(imap_p.read_text())
    print(f"Loaded {len(image_map)} entries from image mapping with Blank flags")

# ======== PROCESS MANUAL MASKS FIRST ========
new_map = {}
proc = skip = 0

for img_id, info in master_map.items():
    img_raw  = info.get('Best Z Filename', '')  # source image
    mask_raw = info.get('MT Mask Path', '')     # manual mask
    img_p = Path(img_raw)
    msk_p = Path(mask_raw)

    if not img_p.exists():
        skip += 1
        continue

    img = cv2.imread(str(img_p))
    if img is None:
        skip += 1
        continue

    if not msk_p.exists():
        # manual map should have masks; if not, just skip here (we'll let the blank pass handle later if flagged)
        skip += 1
        continue

    msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
    if msk is None:
        skip += 1
        continue

    # resize
    img_rs  = cv2.resize(img, TARGET_SIZE, interpolation=IMAGE_INTERP)
    msk_rs  = cv2.resize(msk, TARGET_SIZE, interpolation=MASK_INTERP)
    msk_bin = (msk_rs > 0).astype(np.uint8)

    # save
    out_img = images_out / f"{img_id}.png"
    out_msk = masks_out  / f"{img_id}_mask.png"
    cv2.imwrite(str(out_img),  img_rs)
    cv2.imwrite(str(out_msk), msk_bin)

    # record
    new_map[img_id] = {
        'img_path':  str(out_img.resolve()),
        'mask_path': str(out_msk.resolve()),
        'dayID':     info.get('dayID'),
        'BA':        info.get('BA'),
        'wellID':    info.get('wellID'),
    }
    proc += 1

# ======== SECOND PASS: ADD BLANKS FROM IMAGE MAPPING ========
blank_added = 0
blank_skipped = 0

if image_map:
    for img_id, info in image_map.items():
        # only add if flagged Blank == True and not already present (i.e., no manual mask processed)
        if not info.get('Blank', False):
            continue
        if img_id in new_map:
            # already has manual annotation; skip
            blank_skipped += 1
            continue

        img_raw = info.get('Best Z Filename', '')
        img_p = Path(img_raw)
        if not img_p.exists():
            blank_skipped += 1
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            blank_skipped += 1
            continue

        # resize + make blank mask
        img_rs = cv2.resize(img, TARGET_SIZE, interpolation=IMAGE_INTERP)
        msk_bin = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)  # (H,W)

        out_img = images_out / f"{img_id}.png"
        out_msk = masks_out  / f"{img_id}_mask.png"
        cv2.imwrite(str(out_img), img_rs)
        cv2.imwrite(str(out_msk), msk_bin)

        new_map[img_id] = {
            'img_path':  str(out_img.resolve()),
            'mask_path': str(out_msk.resolve()),
            'dayID':     info.get('dayID'),
            'BA':        info.get('BA'),
            'wellID':    info.get('wellID'),
        }
        blank_added += 1

# ======== DUMP NEW MAPPING ========
new_json = out_base / f"mapping_processed_total_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.json"
with open(new_json, 'w') as f:
    json.dump(new_map, f, indent=2)

# ======== SUMMARY ========
print(f"Processed (manual masks): {proc}, Skipped: {skip}")
if image_map:
    print(f"Added blanks from image mapping: {blank_added}, Skipped blanks: {blank_skipped}")
print(f"New mapping JSON: {new_json}")
