#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from tqdm import tqdm

# -- Constants --
EXPECTED_RECORDS_NUM = 5168

# --- sizes (single source of truth) ---
TARGET_WIDTH = 512
TARGET_HEIGHT = 384

# --- match original behavior ---
INTERPOLATION = cv2.INTER_LINEAR


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


def safe_record_filename(main_id: str) -> str:
    # Original behavior was basically "spaces to underscores"; keep it stable and safe-ish.
    s = (
        main_id.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    return f"{s}.png"


def get_base_folder(mapping: Dict[str, Any]) -> Path:
    """
    Supports a few possible key names so you don't have to fight schema drift.
    (Keep this: it's purely about robustness of the mapping schema.)
    """
    for key in ("_base_folder", "base_folder", "_raw_base_folder", "_images_base_folder", "base_dir"):
        if key in mapping and mapping[key]:
            return Path(mapping[key])
    raise KeyError(
        "Could not find base folder key in mapping JSON (expected one of: "
        "_base_folder, base_folder, _raw_base_folder, _images_base_folder, base_dir)"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resize + remap images into a flat PNG folder and write a processed mapping JSON "
                    "(matches original preprocessing exactly: color read, INTER_LINEAR resize, PNG write)."
    )

    p.add_argument("--image-mapping-json", type=Path, required=True, help="Input mapping JSON (from image_mapper).")
    p.add_argument("--mask-mapping-json", type=Path, required=True, help="Manual masks mapping JSON (from manual masks mapping).")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for processed PNG images.")
    p.add_argument("--out-mapping-json", type=Path, required=True, help="Output mapping JSON with processed_image fields.")

    p.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    p.add_argument("--target-height", type=int, default=TARGET_HEIGHT)

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing processed images.")
    p.add_argument("--smoke", type=int, default=None, help="Process only N records (debug).")

    return p.parse_args()

def get_mask_path(record_id: str, mask_entries: Dict[str, Any]) -> Optional[str]:
    """
    Get the mask path for a given record ID.

    Args:
        record_id: The record ID.
        mask_entries: The mask entries dictionary.

    Returns:
        Optional[str]: The mask path, or None if not found.
    """
    return mask_entries.get(record_id, {}).get("MT Mask Path", None)

def main() -> None:
    args = parse_args()

    logging.info("image_mapping_json: %s", args.image_mapping_json)
    logging.info("out_dir: %s", args.out_dir)
    logging.info("out_mapping_json: %s", args.out_mapping_json)
    logging.info("target: %dx%d", args.target_width, args.target_height)
    logging.info("interpolation: INTER_LINEAR")
    logging.info("overwrite=%s smoke=%s", args.overwrite, args.smoke)

    image_mapping: Dict[str, Any] = json.loads(args.image_mapping_json.read_text())
    base_folder = get_base_folder(image_mapping)

    image_entries: Dict[str, Dict[str, Any]] = image_mapping.get("entries", {})
    if not isinstance(image_entries, dict) or not image_entries:
        raise RuntimeError("Mapping JSON has no 'entries' dict or it's empty.")

    mask_entries: Dict[str, Any] = json.loads(args.mask_mapping_json.read_text())
    if not isinstance(mask_entries, dict) or not mask_entries:
        raise RuntimeError("Mapping JSON has no 'entries' dict or it's empty.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    record_ids = list(image_entries.keys())
    if args.smoke is not None and args.smoke > 0:
        record_ids = record_ids[: args.smoke]

    processed_entries: Dict[str, Dict[str, Any]] = {}
    skipped_exists = 0
    failed = 0
    no_masks = 0

    for record_id in tqdm(record_ids, desc="Processing records"):
        entry = image_entries[record_id]
        try:
            # Match your earlier convention: prefer "Best Z Filename"
            rel_img = (
                entry.get("Best Z Filename")
                or entry.get("image")
                or entry.get("img")
                or entry.get("image_path")
            )
            if not rel_img:
                raise KeyError("Entry missing Best Z Filename (or equivalent image path field)")

            img_path = base_folder / str(rel_img)
            if not img_path.exists():
                raise FileNotFoundError(str(img_path))

            main_id = entry.get("main_id") or record_id
            out_img_path = args.out_dir / safe_record_filename(str(main_id))

            if out_img_path.exists() and not args.overwrite:
                new_entry = dict(entry)
                new_entry["processed_image"] = str(out_img_path)
                new_entry["processed_image_record_id"] = record_id
                mask_path = get_mask_path(record_id, mask_entries)
                if mask_path is None:
                    no_masks += 1
                    logging.debug("record_id=%s has no manual mask path", record_id)
                new_entry["manual_mask_path"] = mask_path
                processed_entries[record_id] = new_entry
                skipped_exists += 1
                continue

            # --- ORIGINAL behavior: read COLOR (3-channel) ---
            img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_raw is None:
                raise ValueError(f"cv2 failed to read image: {img_path}")

            # --- ORIGINAL behavior: resize with INTER_LINEAR ---
            img_final = cv2.resize(
                img_raw,
                (args.target_width, args.target_height),
                interpolation=INTERPOLATION,
            )

            ok = cv2.imwrite(str(out_img_path), img_final)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed: {out_img_path}")

            new_entry = dict(entry)
            new_entry["processed_image"] = str(out_img_path)
            new_entry["processed_image_record_id"] = record_id
            mask_path = get_mask_path(record_id, mask_entries)
            if mask_path is None:
                no_masks += 1
                logging.debug("record_id=%s has no manual mask path", record_id)
            processed_entries[record_id] = new_entry

        except Exception:
            failed += 1
            logging.exception("Skipping record_id=%s due to error", record_id)
            continue

    out_mapping: Dict[str, Any] = {
        "_processed_base_folder": str(args.out_dir),
        "preprocess_params": {
            "target_width": args.target_width,
            "target_height": args.target_height,
            "interpolation": "INTER_LINEAR",
            "read_mode": "IMREAD_COLOR",
            "format": "png",
        },
        "summary": {
            "input_entries": len(image_entries),
            "processed_entries": len(processed_entries),
            "skipped_exists": skipped_exists,
            "failed": failed,
            "no_manual_masks": no_masks,
        },
        "entries": processed_entries,
    }

    assert len(processed_entries) == EXPECTED_RECORDS_NUM, \
        f"Expected {EXPECTED_RECORDS_NUM} records, got {len(processed_entries)}"

    args.out_mapping_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_mapping_json.write_text(json.dumps(out_mapping, indent=2))
    logging.info("Processed mapping saved to: %s", args.out_mapping_json)
    logging.info("Done. processed=%d skipped_exists=%d failed=%d no_manual_masks=%d",
                 len(processed_entries), skipped_exists, failed, no_masks)


if __name__ == "__main__":
    main()