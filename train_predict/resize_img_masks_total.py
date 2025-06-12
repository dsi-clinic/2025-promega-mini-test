#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from paths import TARGET_SIZE, PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR


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
    required=True,
    help='One or more JSON mapping files (e.g. image_mapping_day03_manual.json ...).'
)
parser.add_argument(
    '--days',
    nargs='+',
    metavar='DAY',
    help='Optional subset of day numbers to include, e.g. 03 06 10'
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

# ======== LOAD & MERGE MAPPINGS ========
master_map = {}
for jm in args.mapping:
    jm = Path(jm)
    if not jm.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {jm}")
    master_map.update(json.loads(jm.read_text()))
print(f"Loaded {len(master_map)} total entries from {len(args.mapping)} JSON(s)")

# ======== DAY FILTER (OPTIONAL) ========
if args.days:
    allowed = {f"Dy{int(d):02d}" for d in args.days}
    master_map = {k: v for k, v in master_map.items() if v.get('dayID') in allowed}
    print(f"Filtered to {len(master_map)} entries for days: {sorted(allowed)}")

# ======== PROCESSING LOOP ========
new_map = {}
proc = skip = 0

for img_id, info in master_map.items():
    # support both lowercase and Title-case keys
    img_raw  = info.get('img_path') or info.get('Best Z Filename', '')
    mask_raw = info.get('mask_path') or info.get('Mask Path', '')
    img_p = Path(img_raw)
    msk_p = Path(mask_raw)

    if not img_p.exists() or not msk_p.exists():
        skip += 1
        continue

    img = cv2.imread(str(img_p))
    msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        skip += 1
        continue

    # resize
    img_rs  = cv2.resize(img,  TARGET_SIZE, interpolation=IMAGE_INTERP)
    msk_rs  = cv2.resize(msk,  TARGET_SIZE, interpolation=MASK_INTERP)
    # binarize → 0 or 1
    msk_bin = (msk_rs > 0).astype(np.uint8)

    # save
    out_img = images_out / f"{img_id}.png"
    out_msk = masks_out  / f"{img_id}_mask.png"
    cv2.imwrite(str(out_img),  img_rs)
    cv2.imwrite(str(out_msk), msk_bin)


    # record in new map
    new_map[img_id] = {
        'img_path':  str(out_img.resolve()),
        'mask_path': str(out_msk.resolve()),
        'dayID':     info.get('dayID'),
        'BA':        info.get('BA'),
        'wellID':    info.get('wellID'),
    }
    proc += 1

# ======== DUMP NEW MAPPING ========
new_json = out_base / f"mapping_processed_total_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.json"

with open(new_json, 'w') as f:
    json.dump(new_map, f, indent=2)

# ======== SUMMARY ========
print(f"Processed: {proc}, Skipped: {skip}")
print(f"New mapping JSON: {new_json}")