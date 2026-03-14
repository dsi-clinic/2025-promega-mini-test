#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

# Locate repo root: must contain both paths.py and .env
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")


# --- canonical paths from root ---
from config import (
    TARGET_SIZE,
    RAW_IMAGE_MAPPING_JSON,
    TRAIN_MANUAL_PROCESSED_DIR,
    MANUAL_THRESHOLD_MAPPING,
)

# derive training output subdirs
PROCESSED_IMAGES_DIR = TRAIN_MANUAL_PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR = TRAIN_MANUAL_PROCESSED_DIR / "masks"


# ======== CONFIG ========
IMAGE_INTERP = cv2.INTER_LINEAR
MASK_INTERP = cv2.INTER_NEAREST

# ======== ARGS ========
parser = argparse.ArgumentParser(
    description="Resize images and masks for training from manual+threshold mappings; optionally add blanks from raw mapping."
)
parser.add_argument(
    "--mapping",
    nargs="+",
    default=[str(MANUAL_THRESHOLD_MAPPING)],
    help="One or more JSON mapping files (manual/threshold masks).",
)
parser.add_argument(
    "--image-mapping",
    default=str(RAW_IMAGE_MAPPING_JSON),
    help='Path to the raw image mapping JSON that contains "blank" flags.',
)
parser.add_argument(
    "--output",
    default=None,
    help="Optional output base folder. Defaults to TRAIN_MANUAL_PROCESSED_DIR.",
)


def main():
    args = parser.parse_args()

    # ======== OUTPUT FOLDERS ========
    out_base = Path(args.output) if args.output else TRAIN_MANUAL_PROCESSED_DIR
    images_out = PROCESSED_IMAGES_DIR
    masks_out = PROCESSED_MASKS_DIR

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
    print(
        f"Loaded {len(master_map)} manual-mask entries from {len(args.mapping)} JSON(s)"
    )

    # ======== LOAD IMAGE MAPPING (with nested entries if _base_folder format) ========
    image_map = {}
    if args.image_mapping:
        imap_p = Path(args.image_mapping)
        if not imap_p.exists():
            raise FileNotFoundError(f"Image mapping JSON not found: {imap_p}")

        raw_data = json.loads(imap_p.read_text())

        # Handle new wrapped format with _base_folder
        if (
            isinstance(raw_data, dict)
            and "_base_folder" in raw_data
            and "entries" in raw_data
        ):
            base = Path(raw_data["_base_folder"])
            entries = raw_data["entries"]
            # Re-hydrate relative paths to absolute
            for v in entries.values():
                if "Best Z Filename" in v:
                    v["Best Z Filename"] = str(base / v["Best Z Filename"])
            image_map = entries
        else:
            # Legacy flat format
            image_map = raw_data

        print(f"Loaded {len(image_map)} entries from raw image mapping")

    # ======== PROCESS MANUAL MASKS FIRST ========
    new_map = {}
    proc = skip = 0

    print("Processing manual masks...")
    for img_id, info in tqdm(master_map.items(), desc="Manual masks"):
        img_raw = info.get("Best Z Filename", "")
        mask_raw = info.get("MT Mask Path", "")
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
            skip += 1
            continue

        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            skip += 1
            continue

        # resize
        img_rs = cv2.resize(img, TARGET_SIZE, interpolation=IMAGE_INTERP)
        msk_rs = cv2.resize(msk, TARGET_SIZE, interpolation=MASK_INTERP)
        msk_bin = (msk_rs > 0).astype(np.uint8)

        # save
        out_img = images_out / f"{img_id}.png"
        out_msk = masks_out / f"{img_id}_mask.png"
        cv2.imwrite(str(out_img), img_rs)
        cv2.imwrite(str(out_msk), msk_bin)

        new_map[img_id] = {
            "img_path": str(out_img.resolve()),
            "mask_path": str(out_msk.resolve()),
            "dayID": info.get("dayID"),
            "BA": info.get("BA"),
            "wellID": info.get("wellID"),
        }
        proc += 1

    # ======== SECOND PASS: ADD BLANKS ========
    blank_added = 0
    blank_skipped = 0

    if image_map:
        print("Processing blanks...")
        for img_id, info in tqdm(image_map.items(), desc="Blanks"):
            is_blank = info.get("verification", {}).get("blank", False)

            if not is_blank:
                continue
            if img_id in new_map:
                blank_skipped += 1
                continue

            img_raw = info.get("Best Z Filename", "")
            img_p = Path(img_raw)
            if not img_p.exists():
                blank_skipped += 1
                continue

            img = cv2.imread(str(img_p))
            if img is None:
                blank_skipped += 1
                continue

            # resize + make blank mask (all zeros)
            img_rs = cv2.resize(img, TARGET_SIZE, interpolation=IMAGE_INTERP)
            msk_bin = np.zeros(
                (TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8
            )  # (H, W)

            out_img = images_out / f"{img_id}.png"
            out_msk = masks_out / f"{img_id}_mask.png"
            cv2.imwrite(str(out_img), img_rs)
            cv2.imwrite(str(out_msk), msk_bin)

            new_map[img_id] = {
                "img_path": str(out_img.resolve()),
                "mask_path": str(out_msk.resolve()),
                "dayID": info.get("dayID"),
                "BA": info.get("BA"),
                "wellID": info.get("wellID"),
                "blank": True,  # Flag this as a blank
            }
            blank_added += 1

    # ======== DUMP NEW MAPPING ========
    new_json = (
        out_base / f"mapping_processed_total_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.json"
    )
    with open(new_json, "w") as f:
        json.dump(new_map, f, indent=2)

    # ======== SUMMARY ========
    print(f"Processed (manual masks): {proc}, Skipped: {skip}")
    if image_map:
        print(
            f"Added blanks from image mapping: {blank_added}, Skipped blanks: {blank_skipped}"
        )
    print(f"Total entries in output: {len(new_map)}")
    print(f"New mapping JSON: {new_json}")


if __name__ == "__main__":
    main()
