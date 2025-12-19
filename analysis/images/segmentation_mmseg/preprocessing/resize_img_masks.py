#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

# Locate repo root: must contain both paths.py and .env
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

# Constants
DEFAULT_TARGET_WIDTH = 512
DEFAULT_TARGET_HEIGHT = 384
EXPECTED_RECORDS_NUM = 5168
IMAGE_INTERP = cv2.INTER_LINEAR
MASK_INTERP  = cv2.INTER_NEAREST

def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - image_json: Path to the image mapping JSON file
            - masks_dir: Path to the masks directory
            - output_file: Path to the output JSON file (defaults to masks/manual_masks_mapping.json)

    Raises:
        SystemExit: If --image-json is not provided
    """
    parser = argparse.ArgumentParser(
        description='Map manual masks to image mapping JSON'
    )
    parser.add_argument(
        '--mask-json',
        nargs="+",
        help='List of paths to the mask mapping JSON files'
    )
    parser.add_argument(
        '--image-json',
        type=Path,
        help='Path to the image mapping JSON file'
    )
    parser.add_argument(
        '--output-images-dir',
        type=Path,
        default=None,
        help='Path to the output images directory'
    )
    parser.add_argument(
        '--output-masks-dir',
        type=Path,
        default=None,
        help='Path to the output masks directory'
    )
    parser.add_argument(
        '--target-width',
        type=int,
        default=DEFAULT_TARGET_WIDTH,
        help='Target width of the images/masks (pixels)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=DEFAULT_TARGET_HEIGHT,
        help='Target height of the images/masks (pixels)'
    )
    args = parser.parse_args()

    # Validate required paths
    if not args.image_json:
        parser.error("--image-json is required")
    if not args.mask_json:
        parser.error("--mask-json is required")

    return args

def main():
    start_time = datetime.datetime.now()

    args = get_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)
    target_size = (args.target_width, args.target_height)
    logging.info("Target size: %s", target_size)

    mask_jsons = [Path(mj) for mj in args.mask_json]

    # ======== OUTPUT FOLDERS ========
    output_dir_name = f"resized_{target_size[0]}x{target_size[1]}"
    if args.output_images_dir is None:
        images_out = args.image_json.parent / output_dir_name
    else:
        images_out = args.output_images_dir
    images_out.mkdir(parents=True, exist_ok=True)
    logging.info("Writing processed images to: %s", images_out)

    if args.output_masks_dir is None:
        masks_out = mask_jsons[0].parent / output_dir_name
    else:
        masks_out  = args.output_masks_dir
    masks_out.mkdir(parents=True, exist_ok=True)
    logging.info("Writing processed masks to: %s", masks_out)

    # ======== LOAD & MERGE MANUAL MASK MAPPINGS ========
    master_map = {}
    for jm in mask_jsons:
        if not jm.exists():
            raise FileNotFoundError(f"Mapping JSON not found: {jm}")
        master_map.update(json.loads(jm.read_text()))
    logging.info("Loaded %d manual-mask entries from %d JSON(s)", len(master_map), len(mask_jsons))

    # ======== LOAD IMAGE MAPPING (with nested entries if _base_folder format) ========
    image_map = {}
    if args.image_json:
        if not args.image_json.exists():
            raise FileNotFoundError(f"Image mapping JSON not found: {args.image_json}")

        raw_data = json.loads(args.image_json.read_text())

        # Handle new wrapped format with _base_folder
        if isinstance(raw_data, dict) and "_base_folder" in raw_data and "entries" in raw_data:
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

        logging.info("Loaded %d entries from raw image mapping", len(image_map))

    # ======== PROCESS MANUAL MASKS FIRST ========
    new_map = {}
    proc = skip = 0

    logging.info("Processing manual masks...")
    for img_id, info in tqdm(master_map.items(), desc="Manual masks"):
        img_raw  = info.get("Best Z Filename", "")
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
        img_rs  = cv2.resize(img, target_size, interpolation=IMAGE_INTERP)
        msk_rs  = cv2.resize(msk, target_size, interpolation=MASK_INTERP)
        msk_bin = (msk_rs > 0).astype(np.uint8)

        # save
        updated_img_id = f"{info.get('BA')} {info.get('dayID')} {info.get('wellID')}"    # Preserve actual day to make images/masks consistent with originals
        out_img = images_out / f"{updated_img_id}.png"
        out_msk = masks_out  / f"{updated_img_id}_mask.png"
        cv2.imwrite(str(out_img), img_rs)
        cv2.imwrite(str(out_msk), msk_bin)

        new_map[img_id] = {
            "img_path":  str(out_img.resolve()),
            "mask_path": str(out_msk.resolve()),
            "dayID":     info.get("dayID"),
            "BA":        info.get("BA"),
            "wellID":    info.get("wellID"),
        }
        proc += 1

    # ======== SECOND PASS: ADD BLANKS ========
    blank_added = 0
    blank_skipped = 0
    non_blank_skipped = 0
    if image_map:
        logging.info("Processing blanks...")
        for img_id, info in tqdm(image_map.items(), desc="Blanks"):
            is_blank = info.get("verification", {}).get("blank", False)

            if not is_blank:
                if not img_id in new_map:
                    non_blank_skipped += 1
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
            img_rs  = cv2.resize(img, target_size, interpolation=IMAGE_INTERP)
            msk_bin = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)  # (H, W)

            updated_img_id = f"{info.get('BA')} {info.get('dayID')} {info.get('wellID')}"    # Preserve actual day to make images/masks consistent with originals
            out_img = images_out / f"{updated_img_id}.png"
            out_msk = masks_out  / f"{updated_img_id}_mask.png"
            cv2.imwrite(str(out_img), img_rs)
            cv2.imwrite(str(out_msk), msk_bin)

            new_map[img_id] = {
                "img_path":  str(out_img.resolve()),
                "mask_path": str(out_msk.resolve()),
                "dayID":     info.get("dayID"),
                "BA":        info.get("BA"),
                "wellID":    info.get("wellID"),
                "blank":     True,  # Flag this as a blank
            }
            blank_added += 1

    assert len(new_map) + non_blank_skipped == EXPECTED_RECORDS_NUM, "Number of entries in new map does not match expected number"
    assert proc + blank_added == len(new_map), "Number of entries in new map does not match expected number"

    # ======== DUMP NEW MAPPING ========
    new_json = images_out.parent / f"mapping_processed_total_{target_size[0]}x{target_size[1]}.json"
    with open(new_json, "w") as f:
        json.dump(new_map, f, indent=2)

    # ======== SUMMARY ========
    logging.info("Processed (manual masks): %d, Skipped: %d", proc, skip)
    if image_map:
        logging.info("Added blanks from image mapping: %d, Skipped blanks: %d", blank_added, blank_skipped)
        logging.info("Skipped non-blank entries: %d", non_blank_skipped)
    logging.info("Total entries in output: %d", len(new_map))
    logging.info("New mapping JSON: %s", new_json)

    end_time = datetime.datetime.now()
    logging.info("Elapsed time: %s", end_time - start_time)

if __name__ == "__main__":
    main()
