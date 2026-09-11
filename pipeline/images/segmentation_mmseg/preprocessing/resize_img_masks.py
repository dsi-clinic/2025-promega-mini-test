#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

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


def load_mask_mappings(mask_json_paths: list[Path]) -> dict[str, dict[str, Any]]:
    """
    Load and merge mask mapping JSON files.

    Args:
        mask_json_paths: List of paths to mask mapping JSON files.

    Returns:
        Dict[str, Dict[str, Any]]: Merged dictionary of all mask mappings,
            keyed by image ID.

    Raises:
        FileNotFoundError: If any mask JSON file doesn't exist.
    """
    master_map = {}
    for jm in mask_json_paths:
        if not jm.exists():
            raise FileNotFoundError(f"Mapping JSON not found: {jm}")
        master_map.update(json.loads(jm.read_text()))
    logging.info("Loaded %d manual-mask entries from %d JSON(s)", len(master_map), len(mask_json_paths))
    return master_map


def load_image_mapping(image_json_path: Path) -> dict[str, dict[str, Any]]:
    """
    Load image mapping JSON file, handling both wrapped and flat formats.

    Args:
        image_json_path: Path to the image mapping JSON file.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of image mapping entries,
            keyed by image ID.

    Raises:
        FileNotFoundError: If the image JSON file doesn't exist.
    """
    if not image_json_path.exists():
        raise FileNotFoundError(f"Image mapping JSON not found: {image_json_path}")

    raw_data = json.loads(image_json_path.read_text())

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
    return image_map


def setup_output_directories(
    image_json_path: Path,
    mask_json_paths: list[Path],
    target_size: tuple[int, int],
    output_images_dir: Path | None,
    output_masks_dir: Path | None
) -> tuple[Path, Path]:
    """
    Set up output directories for processed images and masks.

    Args:
        image_json_path: Path to the image JSON file (used for default output location).
        mask_json_paths: List of mask JSON paths (used for default output location).
        target_size: Target size tuple (width, height).
        output_images_dir: Optional explicit output directory for images.
        output_masks_dir: Optional explicit output directory for masks.

    Returns:
        tuple[Path, Path]: Tuple of (images_output_dir, masks_output_dir).
    """
    output_dir_name = f"resized_{target_size[0]}x{target_size[1]}"

    if output_images_dir is None:
        images_out = image_json_path.parent / output_dir_name
    else:
        images_out = output_images_dir
    images_out.mkdir(parents=True, exist_ok=True)
    logging.info("Writing processed images to: %s", images_out)

    if output_masks_dir is None:
        masks_out = mask_json_paths[0].parent / output_dir_name
    else:
        masks_out = output_masks_dir
    masks_out.mkdir(parents=True, exist_ok=True)
    logging.info("Writing processed masks to: %s", masks_out)

    return images_out, masks_out


def build_output_filename(info: dict[str, Any]) -> str:
    """
    Build output filename from info dictionary, preserving actual dayID.

    Args:
        info: Dictionary containing 'BA', 'dayID', and 'wellID' keys.

    Returns:
        str: Formatted filename string in the format "BA dayID wellID".
    """
    return f"{info.get('BA')} {info.get('dayID')} {info.get('wellID')}"


def load_image(img_path: Path) -> np.ndarray | None:
    """
    Load an image file.

    Args:
        img_path: Path to the image file.

    Returns:
        Optional[np.ndarray]: Loaded image array, or None if loading failed.
    """
    if not img_path.exists():
        return None
    return cv2.imread(str(img_path))


def load_mask(mask_path: Path) -> np.ndarray | None:
    """
    Load a mask file as grayscale.

    Args:
        mask_path: Path to the mask file.

    Returns:
        Optional[np.ndarray]: Loaded mask array, or None if loading failed.
    """
    if not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def resize_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize an image to target size.

    Args:
        img: Input image array (BGR format).
        target_size: Target size tuple (width, height).

    Returns:
        np.ndarray: Resized image array (BGR format).
    """
    return cv2.resize(img, target_size, interpolation=IMAGE_INTERP)


def resize_mask(msk: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize a mask to target size and binarize it.

    Args:
        msk: Input mask array (grayscale).
        target_size: Target size tuple (width, height).

    Returns:
        np.ndarray: Binarized resized mask (uint8, values 0 or 1).
    """
    msk_rs = cv2.resize(msk, target_size, interpolation=MASK_INTERP)
    return (msk_rs > 0).astype(np.uint8)


def create_blank_mask(target_size: tuple[int, int]) -> np.ndarray:
    """
    Create a blank (all zeros) mask of target size.

    Args:
        target_size: Target size tuple (width, height).

    Returns:
        np.ndarray: Blank mask array (uint8, all zeros) with shape (height, width).
    """
    return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)  # (H, W)


def save_resized_pair(
    img: np.ndarray,
    msk: np.ndarray,
    filename_base: str,
    images_out: Path,
    masks_out: Path
) -> tuple[Path, Path]:
    """
    Save resized image and mask pair to disk.

    Args:
        img: Resized image array (BGR format).
        msk: Resized mask array (binary, uint8).
        filename_base: Base filename (without extension).
        images_out: Output directory for images.
        masks_out: Output directory for masks.

    Returns:
        Tuple[Path, Path]: Tuple of (saved_image_path, saved_mask_path),
            both as resolved absolute paths.
    """
    out_img = images_out / f"{filename_base}.png"
    out_msk = masks_out / f"{filename_base}_mask.png"
    cv2.imwrite(str(out_img), img)
    cv2.imwrite(str(out_msk), msk)
    return out_img.resolve(), out_msk.resolve()


def create_mapping_entry(
    img_path: Path,
    mask_path: Path,
    info: dict[str, Any],
    is_blank: bool = False
) -> dict[str, Any]:
    """
    Create a mapping dictionary entry.

    Args:
        img_path: Path to the processed image.
        mask_path: Path to the processed mask.
        info: Original info dictionary containing metadata.
        is_blank: Whether this is a blank entry.

    Returns:
        Dict[str, Any]: Mapping entry dictionary with keys:
            'img_path', 'mask_path', 'dayID', 'BA', 'wellID',
            and optionally 'blank' if is_blank is True.
    """
    entry = {
        "img_path": str(img_path),
        "mask_path": str(mask_path),
        "dayID": info.get("dayID"),
        "BA": info.get("BA"),
        "wellID": info.get("wellID"),
    }
    if is_blank:
        entry["blank"] = True
    return entry


def process_mask_entry(
    img_id: str,
    info: dict[str, Any],
    images_out: Path,
    masks_out: Path,
    target_size: tuple[int, int]
) -> dict[str, Any] | None:
    """
    Process a single entry with mask: load, resize, and save.

    Args:
        img_id: Image identifier (key from mapping).
        info: Entry info dictionary containing 'Best Z Filename' and 'MT Mask Path'.
        images_out: Output directory for images.
        masks_out: Output directory for masks.
        target_size: Target size tuple (width, height).

    Returns:
        Optional[Dict[str, Any]]: Mapping entry dictionary if successful,
            None if image or mask loading failed.
    """
    img_path = Path(info.get("Best Z Filename", ""))
    mask_path = Path(info.get("MT Mask Path", ""))

    img = load_image(img_path)
    if img is None:
        return None

    msk = load_mask(mask_path)
    if msk is None:
        return None

    # Resize
    img_rs = resize_image(img, target_size)
    msk_bin = resize_mask(msk, target_size)

    # Save
    filename_base = build_output_filename(info)
    out_img, out_msk = save_resized_pair(img_rs, msk_bin, filename_base, images_out, masks_out)

    return create_mapping_entry(out_img, out_msk, info, is_blank=False)


def process_blank_entry(
    img_id: str,
    info: dict[str, Any],
    images_out: Path,
    masks_out: Path,
    target_size: tuple[int, int]
) -> dict[str, Any] | None:
    """
    Process a single blank entry: load image, create blank mask, and save.

    Args:
        img_id: Image identifier (key from mapping).
        info: Entry info dictionary containing 'Best Z Filename'.
        images_out: Output directory for images.
        masks_out: Output directory for masks.
        target_size: Target size tuple (width, height).

    Returns:
        Optional[Dict[str, Any]]: Mapping entry dictionary if successful,
            None if image loading failed.
    """
    img_path = Path(info.get("Best Z Filename", ""))

    img = load_image(img_path)
    if img is None:
        return None

    # Resize image and create blank mask
    img_rs = resize_image(img, target_size)
    msk_bin = create_blank_mask(target_size)

    # Save
    filename_base = build_output_filename(info)
    out_img, out_msk = save_resized_pair(img_rs, msk_bin, filename_base, images_out, masks_out)

    return create_mapping_entry(out_img, out_msk, info, is_blank=True)


def save_mapping_json(
    new_map: dict[str, dict[str, Any]],
    images_out: Path,
    target_size: tuple[int, int]
) -> Path:
    """
    Save the final mapping JSON file.

    Args:
        new_map: Dictionary of processed entries, keyed by image ID.
        images_out: Output directory for images (used to determine JSON location).
        target_size: Target size tuple (width, height).

    Returns:
        Path: Path to the saved JSON file.
    """
    new_json = images_out.parent / f"mapping_processed_total_{target_size[0]}x{target_size[1]}.json"
    with open(new_json, "w") as f:
        json.dump(new_map, f, indent=2)
    return new_json


def log_summary(
    proc: int,
    skip: int,
    blank_added: int,
    blank_skipped: int,
    non_blank_skipped: int,
    new_map: dict[str, dict[str, Any]],
    new_json: Path,
    elapsed_time: datetime.timedelta
) -> None:
    """
    Log processing summary statistics.

    Args:
        proc: Number of entries processed with masks.
        skip: Number of entries skipped during mask processing.
        blank_added: Number of blank entries added.
        blank_skipped: Number of blank entries skipped.
        non_blank_skipped: Number of non-blank entries skipped.
        new_map: Final mapping dictionary of processed entries.
        new_json: Path to the saved JSON file.
        elapsed_time: Total elapsed time for processing.
    """
    logging.info("Processed (manual masks): %d, Skipped: %d", proc, skip)
    logging.info("Added blanks from image mapping: %d, Skipped blanks: %d", blank_added, blank_skipped)
    logging.info("Skipped non-blank entries: %d", non_blank_skipped)
    logging.info("Total entries in output: %d", len(new_map))
    logging.info("New mapping JSON: %s", new_json)
    logging.info("Elapsed time: %s", elapsed_time)

def main() -> None:
    """
    Main entry point for resizing images and masks.

    Orchestrates the complete workflow:
    1. Parse command-line arguments
    2. Setup output directories
    3. Load mask and image mappings
    4. Process entries with manual masks
    5. Process blank entries
    6. Validate results
    7. Save mapping JSON
    8. Log summary statistics
    """
    start_time = datetime.datetime.now()

    args = get_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)
    target_size = (args.target_width, args.target_height)
    logging.info("Target size: %s", target_size)

    mask_jsons = [Path(mj) for mj in args.mask_json]

    # Setup output directories
    images_out, masks_out = setup_output_directories(
        args.image_json,
        mask_jsons,
        target_size,
        args.output_images_dir,
        args.output_masks_dir
    )

    # Load mappings
    master_map = load_mask_mappings(mask_jsons)
    image_map = load_image_mapping(args.image_json)

    # Process manual masks
    new_map = {}
    proc = skip = 0

    logging.info("Processing manual masks...")
    for img_id, info in tqdm(
        master_map.items(),
        desc="Manual masks",
        position=0,
        leave=True,
        ncols=100,
        mininterval=0.5
    ):
        entry = process_mask_entry(img_id, info, images_out, masks_out, target_size)
        if entry is None:
            skip += 1
            continue
        new_map[img_id] = entry
        proc += 1

    # Process blank entries
    blank_added = 0
    blank_skipped = 0
    non_blank_skipped = 0

    if image_map:
        logging.info("Processing blanks...")
        for img_id, info in tqdm(
            image_map.items(),
            desc="Blanks",
            position=0,
            leave=True,
            ncols=100,
            mininterval=0.1
        ):
            is_blank = info.get("verification", {}).get("blank", False)

            if not is_blank:
                if img_id not in new_map:
                    non_blank_skipped += 1
                continue

            if img_id in new_map:
                blank_skipped += 1
                continue

            entry = process_blank_entry(img_id, info, images_out, masks_out, target_size)
            if entry is None:
                blank_skipped += 1
                continue

            new_map[img_id] = entry
            blank_added += 1

    # Validate results
    assert len(new_map) + non_blank_skipped == EXPECTED_RECORDS_NUM, \
        f"Expected {EXPECTED_RECORDS_NUM} records, got {len(new_map) + non_blank_skipped}"
    assert proc + blank_added == len(new_map), \
        f"Processed count mismatch: {proc} + {blank_added} != {len(new_map)}"

    # Save mapping JSON
    new_json = save_mapping_json(new_map, images_out, target_size)

    # Log summary
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    log_summary(proc, skip, blank_added, blank_skipped, non_blank_skipped, new_map, new_json, elapsed_time)

if __name__ == "__main__":
    main()
