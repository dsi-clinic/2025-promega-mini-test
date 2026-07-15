#!/usr/bin/env python3
import argparse
import dataclasses
import datetime
import json
import logging
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


# Constants
EXPECTED_RECORDS_NUM = 5168


@dataclasses.dataclass
class Config:
    image_mapping_json: Path = dataclasses.field(metadata={
        "help": "Path to image mapping JSON file created by resize remap images operations"
    })
    overlay_dir: Path = dataclasses.field(metadata={
        "help": "Directory to store image overlay results"
    })
    overwrite: bool = dataclasses.field(default=False, metadata={
        "help": "Overwrite existing mask files"
    })
    def __post_init__(self):
        if not self.image_mapping_json.exists():
            raise ValueError(f"Image mapping JSON does not exist: {self.image_mapping_json}")
        if not self.overlay_dir.exists():
            self.overlay_dir.mkdir(parents=True, exist_ok=True)

def get_args():
    arg_parser = create_args()
    args = arg_parser.parse_args()
    args_dict = vars(args)
    cfg = Config(**args_dict)
    return cfg

def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Build outline overlays for all processed image/mask pairs and update mapping JSONs with overlay_path.")

    for field in dataclasses.fields(Config):
        # Build argument flag and help message
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {
            "help": field.metadata.get("help", ""),
            "default": field.default
        }

        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type
        parser.add_argument(*flags, **kwargs)

    return parser

def load_json(p: Path):
    with p.open("r") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2)

def read_image_bgr(p: Path) -> np.ndarray | None:
    """Read as BGR (OpenCV) with PIL fallback."""
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        try:
            img = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]  # RGB->BGR
        except Exception:
            return None
    return img

def read_mask_bin(p: Path) -> np.ndarray | None:
    """Read mask as binary uint8 (0/1), with PIL fallback."""
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        try:
            m = np.array(Image.open(p).convert("L"))
        except Exception:
            return None
    return (m > 0).astype(np.uint8)

def ensure_gray_binary(mask: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)

def derive_overlay_path(mask_path_str: str) -> Path:
    """
    predictions/<batch>/<day>/image_mask_overlays/<basename>_overlay.png
    mirroring 'predicted_masks' placement.
    """
    mp = Path(mask_path_str)
    mask_dir = mp.parent
    day_dir = mask_dir.parent               # e.g., .../day28
    overlays_dir = day_dir / "image_mask_overlays"
    stem = mp.stem  # e.g., BA2_96_1_Dy28_B9_predmask
    out_stem = re.sub(r"_predmask$", "", stem, flags=re.IGNORECASE) + "_overlay"
    return overlays_dir / f"{out_stem}.png"

def draw_outline_overlay(img_bgr: np.ndarray, mask_bin: np.ndarray, color=(0,255,0), thickness=2) -> np.ndarray:
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img_bgr.copy()
    if contours:
        cv2.drawContours(out, contours, contourIdx=-1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out

def initialize_totals() -> Counter:
    """Initialize the statistics counter."""
    return Counter({
        "overlays_created": 0,
        "overlays_skipped": 0,
        "overlays_skipped_existing": 0,
        "pairs_total": 0,
        "missing_imgs": 0,
        "missing_masks": 0,
        "decode_imgs": 0,
        "decode_masks": 0,
        "write_fails": 0,
        "processed": 0
    })

def validate_paths_in_json(record_id: str, record: dict) -> tuple[bool, str | None]:
    """
    Validate that image and mask paths exist in JSON record.

    Returns:
        (is_valid, error_message): Tuple indicating if paths are valid and error message if not.
    """
    img_path = record.get("processed_image")
    mask_path = record.get("predicted_mask_path")

    if not img_path or not mask_path:
        error_msg = f"missing in json: [img_path:{img_path}] or [mask_path:{mask_path}]"
        logging.warning(f"Record {record_id} has no image or mask path")
        return False, error_msg

    return True, None

def validate_files_on_disk(img_path: str, mask_path: str) -> tuple[bool, str | None]:
    """
    Validate that image and mask files exist on disk.

    Returns:
        (is_valid, error_message): Tuple indicating if files exist and error message if not.
    """
    img_p = Path(img_path)
    mask_p = Path(mask_path)

    miss_img = not img_p.exists()
    miss_msk = not mask_p.exists()

    if miss_img or miss_msk:
        reason_parts = []
        if miss_img:
            reason_parts.append(f"[img:{img_p}]")
        if miss_msk:
            reason_parts.append(f"[mask:{mask_p}]")
        error_msg = f"missing on disk: {' '.join(reason_parts)}"
        return False, error_msg

    return True, None

def load_image_and_mask(img_path: str, mask_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load and decode image and mask files.

    Returns:
        (image, mask): Tuple of loaded arrays, or (None, None) if loading failed.
    """
    img_p = Path(img_path)
    mask_p = Path(mask_path)

    img = read_image_bgr(img_p)
    mask_bin = read_mask_bin(mask_p)

    return img, mask_bin

def create_or_skip_overlay(
    record_id: str,
    record: dict,
    img: np.ndarray,
    mask_bin: np.ndarray,
    img_path: str,
    overlay_dir: Path,
    overwrite: bool,
    totals: Counter,
    missing_pairs: list
) -> tuple[bool, bool]:
    """
    Create overlay image or skip if it already exists.

    Returns:
        (overlay_created, json_updated): Tuple indicating if overlay was created and if JSON was updated.
    """
    img_p = Path(img_path)
    out_path = overlay_dir / f"{img_p.stem}_overlay.png"

    overlay_created = False
    json_updated = False

    if out_path.exists() and not overwrite:
        # Overlay exists, just update JSON if needed
        if record.get("overlay_path") != str(out_path):
            record["overlay_path"] = str(out_path)
            json_updated = True
        totals.update({
            "overlays_skipped_existing": 1,
            "processed": 1,
        })
    else:
        # Create new overlay
        overlay = draw_outline_overlay(img, mask_bin, color=(0,255,0), thickness=2)
        ok = cv2.imwrite(str(out_path), overlay)

        if not ok:
            totals.update({
                "write_fails": 1,
                "overlays_skipped": 1,
            })
            missing_pairs.append((record_id, f"write failed: {out_path}"))
            return False, False

        record["overlay_path"] = str(out_path)
        json_updated = True
        overlay_created = True
        totals.update({
            "overlays_created": 1,
            "processed": 1,
        })

    return overlay_created, json_updated

def validate_data_integrity(totals: Counter, mapping_file_name: str):
    """Validate data integrity by checking assertions."""
    assert totals["overlays_created"] + totals["overlays_skipped_existing"] == totals["pairs_total"], (
        f"{mapping_file_name}: overlays_created({totals['overlays_created']}) + overlays_skipped_existing({totals['overlays_skipped_existing']}) != pairs_total({totals['pairs_total']})"
    )
    assert totals["overlays_created"] + totals["overlays_skipped_existing"] == EXPECTED_RECORDS_NUM, (
        f"{mapping_file_name}: overlays_created({totals['overlays_created']}) + overlays_skipped_existing({totals['overlays_skipped_existing']}) != EXPECTED_RECORDS_NUM({EXPECTED_RECORDS_NUM})"
    )

def print_summary(totals: Counter):
    """Print summary statistics."""
    logging.info("overlays created          : %d", totals["overlays_created"])
    logging.info("overlays skipped          : %d", totals["overlays_skipped"])
    logging.info("overlays skipped existing : %d", totals["overlays_skipped_existing"])
    logging.info("valid (img+mask) pairs    : %d", totals["pairs_total"])
    logging.info("overlays write fails      : %d", totals["write_fails"])
    logging.info("total overlays processed  : %d", totals["processed"])

def save_results(totals: Counter, missing_pairs: list, overlay_dir: Path):
    """Save summary statistics to JSON file."""
    totals["missing_pairs"] = missing_pairs
    summary_path = overlay_dir / "summary.json"
    save_json(summary_path, totals)
    logging.info("summary saved to: %s", summary_path)

def update_mapping_json(mapping: dict, mapping_json_path: Path, json_updated: bool):
    """Update mapping JSON file if any records were modified."""
    if json_updated:
        new_json = Path(mapping_json_path.parent / (mapping_json_path.stem + "_overlay.json"))
        save_json(new_json, mapping)
        logging.info("Updated mapping JSON: %s", new_json)

def process_record(
    record_id: str,
    record: dict,
    overlay_dir: Path,
    overwrite: bool,
    totals: Counter,
    missing_pairs: list
) -> bool:
    """
    Process a single record: validate, load, and create overlay.

    Returns:
        json_updated: Boolean indicating if the record was updated in JSON.
    """
    # Step 1: Validate paths in JSON
    is_valid, error_msg = validate_paths_in_json(record_id, record)
    if not is_valid:
        totals.update({
            "missing_imgs": 1,
            "missing_masks": 1,
            "overlays_skipped": 1,
        })
        missing_pairs.append((record_id, error_msg))
        return False

    img_path = record.get("processed_image")
    mask_path = record.get("predicted_mask_path")

    # Step 2: Validate files exist on disk
    is_valid, error_msg = validate_files_on_disk(img_path, mask_path)
    if not is_valid:
        img_p = Path(img_path)
        mask_p = Path(mask_path)
        miss_img = not img_p.exists()
        miss_msk = not mask_p.exists()
        totals.update({
            "missing_imgs": 1 if miss_img else 0,
            "missing_masks": 1 if miss_msk else 0,
            "overlays_skipped": 1,
        })
        missing_pairs.append((record_id, error_msg))
        return False

    # Step 3: Load image and mask
    img, mask_bin = load_image_and_mask(img_path, mask_path)
    if img is None or mask_bin is None:
        totals.update({
            "decode_imgs": 1 if img is None else 0,
            "decode_masks": 1 if mask_bin is None else 0,
            "overlays_skipped": 1,
        })
        missing_pairs.append((record_id, f"decode failed: img({img is None}), mask({mask_bin is None})"))
        return False

    # Step 4: Create or skip overlay
    totals.update({"pairs_total": 1})
    _, json_updated = create_or_skip_overlay(
        record_id, record, img, mask_bin, img_path,
        overlay_dir, overwrite, totals, missing_pairs
    )

    return json_updated

def main():
    start = datetime.datetime.now()
    args = get_args()
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    mapping = load_json(args.image_mapping_json)
    logging.info(f"Found {len(mapping.get('entries', {}))} records in: %s", args.image_mapping_json)

    totals = initialize_totals()
    missing_pairs = []
    json_updated = False

    for record_id, record in tqdm(mapping.get('entries', {}).items(), desc="Processing records"):
        record_updated = process_record(
            record_id, record, args.overlay_dir, args.overwrite, totals, missing_pairs
        )
        if record_updated:
            json_updated = True

    validate_data_integrity(totals, args.image_mapping_json.name)
    print_summary(totals)
    save_results(totals, missing_pairs, args.overlay_dir)
    update_mapping_json(mapping, args.image_mapping_json, json_updated)

    end = datetime.datetime.now()
    logging.info("time taken: %s", end - start)

if __name__ == "__main__":
    main()
