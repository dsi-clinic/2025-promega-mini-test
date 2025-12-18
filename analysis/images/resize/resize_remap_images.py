#!/usr/bin/env python3
"""Resize images and create processed mapping JSON files."""
import argparse
import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Optional, Tuple

import cv2

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

# --------------- Configuration -----------------
INTERPOLATION = cv2.INTER_LINEAR
DEFAULT_BATCHES = "1,2,3,4"
DEFAULT_DAYS = "3,6,8,10,13,15,17,20,21,24,28,30"
DEFAULT_TARGET_WIDTH = 512
DEFAULT_TARGET_HEIGHT = 384

# --------------- Utility Functions -----------------
def norm(s: str) -> str:
    """Normalize string: lowercase and remove spaces."""
    return s.lower().replace(' ', '') if isinstance(s, str) else ''


def ba_match(json_ba: str, batch_id: str) -> bool:
    """Check if JSON BA matches batch ID (case-insensitive, space-insensitive)."""
    return norm(json_ba).startswith(norm(batch_id))


def parse_comma_separated_ints(value: str) -> list[int]:
    """Parse comma-separated string into list of integers."""
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def ensure_output_directories(output_dir: Path) -> None:
    """Create output directory structure if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)


def get_output_paths(output_dir: Path, batch_id: str, day_num: int) -> Tuple[Path, Path]:
    """Generate output paths for JSON mapping and images directory.

    Args:
        output_dir: Base output directory
        batch_id: Batch identifier (e.g., "ba196_1")
        day_num: Day number

    Returns:
        Tuple of (json_path, images_dir)
    """
    day_id = f"Dy{day_num:02d}"
    safe_batch = batch_id.replace(' ', '_')
    json_path = output_dir / "json" / f"image_mapping_{safe_batch}_{day_id}_processed.json"
    images_dir = output_dir / "images"
    return json_path, images_dir


def load_image_mapping(input_json: Path) -> Tuple[dict, Path]:
    """Load image mapping JSON and extract entries and base folder.

    Args:
        input_json: Path to input JSON file

    Returns:
        Tuple of (entries_dict, base_folder_path)
    """
    with input_json.open() as f:
        full_mapping = json.load(f)
    entries = full_mapping.get("entries", {})
    base_folder = Path(full_mapping.get("_base_folder", ""))
    return entries, base_folder


def find_batch_ids(entries: dict, batch_num: int) -> list[str]:
    """Find all batch ID variants matching the given batch number.

    Args:
        entries: Dictionary of image entries
        batch_num: Batch number to match

    Returns:
        Sorted list of matching batch IDs
    """
    batch_prefix = norm(f"BA{batch_num}")
    all_ba_keys = {norm(info['BA']) for info in entries.values() if 'BA' in info}
    return sorted(b for b in all_ba_keys if b.startswith(batch_prefix))


def filter_entries_by_day_and_batch(
    entries: dict,
    day_id: str,
    batch_id: str
) -> list[Tuple[str, str, Optional[float]]]:
    """Filter entries matching day and batch criteria.

    Args:
        entries: Dictionary of image entries
        day_id: Day identifier (e.g., "Dy03")
        batch_id: Batch identifier

    Returns:
        List of tuples: (img_id, best_z_filename, um_per_px)
    """
    return [
        (img_id, info['Best Z Filename'], info.get('um_per_px'))
        for img_id, info in entries.items()
        if norm(info.get('dayID', '')) == norm(day_id)
        and ba_match(info.get('BA', ''), batch_id)
    ]


def calculate_um_per_px(
    orig_um_px: Optional[float | Tuple[float, float]],
    orig_width: int,
    orig_height: int,
    target_width: int,
    target_height: int
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Calculate original and final um_per_px values.

    Args:
        orig_um_px: Original um_per_px (scalar or tuple)
        orig_width: Original image width
        orig_height: Original image height
        target_width: Target image width
        target_height: Target image height

    Returns:
        Tuple of (um_x, um_y, final_um_per_px_x, final_um_per_px_y)
    """
    if orig_um_px is None:
        return None, None, None, None

    if isinstance(orig_um_px, (list, tuple)) and len(orig_um_px) == 2:
        um_x, um_y = orig_um_px
    else:
        um_x = um_y = orig_um_px

    scale_x = orig_width / target_width
    scale_y = orig_height / target_height
    final_um_per_px_x = um_x * scale_x
    final_um_per_px_y = um_y * scale_y

    return um_x, um_y, final_um_per_px_x, final_um_per_px_y


def process_single_image(
    img_path: Path,
    main_id: str,
    output_images_dir: Path,
    target_size: Tuple[int, int],
    orig_um_px: Optional[float | Tuple[float, float]]
) -> Optional[dict]:
    """Process a single image: load, resize, save, and return metadata.

    Args:
        img_path: Path to source image
        main_id: Main identifier for the image
        output_images_dir: Directory to save processed image
        target_size: Target (width, height) for resizing
        orig_um_px: Original um_per_px value

    Returns:
        Dictionary with image metadata, or None if processing failed
    """
    if not img_path.exists():
        logging.warning(f"Skipped missing: {img_path}")
        return None

    img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_raw is None:
        logging.warning(f"Skipped unreadable: {img_path}")
        return None

    orig_h, orig_w = img_raw.shape[:2]
    img_final = cv2.resize(img_raw, target_size, INTERPOLATION)

    um_x, um_y, final_um_per_px_x, final_um_per_px_y = calculate_um_per_px(
        orig_um_px, orig_w, orig_h, target_size[0], target_size[1]
    )

    out_path = output_images_dir / f"{main_id}.png"
    cv2.imwrite(str(out_path), img_final)

    return {
        "img_path": str(out_path),
        "main_id": main_id,
        "orig_width_px": orig_w,
        "orig_height_px": orig_h,
        "orig_um_per_px_x": um_x,
        "orig_um_per_px_y": um_y,
        "final_um_per_px_x": final_um_per_px_x,
        "final_um_per_px_y": final_um_per_px_y
    }


def extract_main_id(entry_info: dict, img_id: str) -> str:
    """Extract main_id from entry, with fallback to normalized img_id.

    Args:
        entry_info: Entry information dictionary
        img_id: Original image ID

    Returns:
        Main identifier string
    """
    verification = entry_info.get("verification", {})
    main_id = verification.get("main_id")
    if main_id:
        return main_id
    return img_id.replace(' ', '_')


# --------------- Main Processing Functions -----------------
def create_mapping(
    entries: dict,
    base_folder: Path,
    batch_id: str,
    day_num: int,
    target_size: Tuple[int, int],
    output_dir: Path
) -> None:
    """Create a mapping for a given batch and day.

    Args:
        entries: Dictionary of entries
        base_folder: Base folder of the images
        batch_id: Batch ID
        day_num: Day number
        target_size: Target size of the images
        output_dir: Path to the output directory
    """
    day_id = f"Dy{day_num:02d}"
    output_json, output_images_dir = get_output_paths(output_dir, batch_id, day_num)

    if output_json.exists():
        logging.warning(f"Skipping because mapping already exists: {output_json}")
        logging.info(f"Size: {output_json.stat().st_size} bytes")
        return

    matches = filter_entries_by_day_and_batch(entries, day_id, batch_id)
    new_mapping = {}

    for img_id, img_path_str, orig_um_px in matches:
        entry_info = entries[img_id]
        main_id = extract_main_id(entry_info, img_id)

        img_path = base_folder / img_path_str
        metadata = process_single_image(
            img_path, main_id, output_images_dir, target_size, orig_um_px
        )

        if metadata:
            new_mapping[img_id] = metadata

    with output_json.open('w') as f:
        json.dump(new_mapping, f, indent=2)

    logging.info(f"Created mapping for {batch_id} with {len(new_mapping)} images")
    logging.info(f"Images saved to: {output_dir}")
    logging.info(f"Mapping saved to: {output_json}")


def process_batch(
    input_json: Path,
    output_dir: Path,
    batch_num: int,
    day_num: int,
    target_size: Tuple[int, int]
) -> None:
    """Process a batch of images and create mappings.

    Args:
        input_json: Path to the input JSON file
        output_dir: Path to the output directory
        batch_num: Batch number
        day_num: Day number
        target_size: Target size of the images
    """
    entries, base_folder = load_image_mapping(input_json)
    batch_ids = find_batch_ids(entries, batch_num)

    for batch_id in batch_ids:
        create_mapping(entries, base_folder, batch_id, day_num, target_size, output_dir)


# --------------- CLI Functions -----------------
def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Resize and remap images for a given batch and day'
    )
    parser.add_argument(
        '--batches',
        default=DEFAULT_BATCHES,
        help='Comma-separated batch numbers, e.g. 1,2,3'
    )
    parser.add_argument(
        '--days',
        default=DEFAULT_DAYS,
        help='Comma-separated day numbers, e.g. 3,6,8'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove existing processed folders first'
    )
    parser.add_argument(
        '--image-json',
        type=Path,
        help='Path to the image mapping JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Path to the output directory'
    )
    parser.add_argument(
        '--target-width',
        type=int,
        default=DEFAULT_TARGET_WIDTH,
        help='Target width of the images (pixels)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=DEFAULT_TARGET_HEIGHT,
        help='Target height of the images (pixels)'
    )
    args = parser.parse_args()

    # Validate required paths
    if not args.image_json:
        parser.error("--image-json is required (or set RAW_IMAGE_MAPPING_JSON in config)")
    if not args.output_dir:
        parser.error("--output-dir is required (or set INFER_RESIZED_DIR in config)")

    return args


def main() -> None:
    """Main entry point."""
    start_time = datetime.datetime.now()
    args = get_args()

    # Print configuration
    for key, val in vars(args).items():
        logging.info(f"{key}: {val}")

    # Setup output directories
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    ensure_output_directories(args.output_dir)

    target_size = (args.target_width, args.target_height)
    logging.info(f"Target size: {target_size}")

    batches = parse_comma_separated_ints(args.batches)
    days = parse_comma_separated_ints(args.days)

    for batch in batches:
        for day in days:
            logging.info(f"Processing Batch {batch}, Day {day}")
            process_batch(
                args.image_json,
                args.output_dir,
                batch,
                day,
                target_size
            )

    end_time = datetime.datetime.now()
    logging.info(f"Elapsed time: {end_time - start_time} seconds")


# --------------- CLI Entrypoint -----------------
if __name__ == "__main__":
    main()
