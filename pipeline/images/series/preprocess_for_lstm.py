import argparse
import json
import logging
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

# Target physical scale and dimensions
COMPLETE_SERIES = 'complete_series_data_no_blanks.json'
TARGET_UM_PER_PX = 6.0
TARGET_SIZE = 768

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Preprocess images and masks for LSTM training with uniform physical scale'
    )
    parser.add_argument('--complete-series', type=Path, default=COMPLETE_SERIES,
                       help=f'Complete series data file (default: {COMPLETE_SERIES})')
    parser.add_argument('--raw-image-dir', type=Path,
                       help='Base folder for raw images')
    parser.add_argument('--out-dir', type=Path,
                       help='Output directory')
    parser.add_argument('--target-um-per-px', type=float, default=TARGET_UM_PER_PX,
                       help=f'Target physical scale in um per pixel (default: {TARGET_UM_PER_PX})')
    parser.add_argument('--target-size', type=int, default=TARGET_SIZE,
                       help=f'Target square size in pixels (default: {TARGET_SIZE})')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip dimension analysis and proceed directly')
    parser.add_argument('--save-debug', action='store_true',
                       help='Save debug images for first 5 entries')
    args = parser.parse_args()
    return args

def load_json(p: Path):
    with open(p) as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def calculate_target_dimensions(orig_width, orig_height, orig_um_per_px, target_um_per_px):
    """
    Calculate new dimensions to achieve target physical scale
    Returns (new_width, new_height) maintaining aspect ratio
    """
    scale_factor = orig_um_per_px / target_um_per_px
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    return new_width, new_height

def find_max_target_dimensions(data, target_um_per_px):
    """Find the maximum target dimensions needed across all images"""
    max_width = 0
    max_height = 0

    for entry in data.values():
        orig_um_per_px = entry.get('um_per_px')
        if isinstance(orig_um_per_px, (list, tuple)):
            orig_um_per_px = orig_um_per_px[0]

        if orig_um_per_px and 'processed' in entry:
            orig_w = entry['processed'].get('orig_width_px')
            orig_h = entry['processed'].get('orig_height_px')

            if orig_w and orig_h:
                target_w, target_h = calculate_target_dimensions(
                    orig_w, orig_h, orig_um_per_px, target_um_per_px
                )
                max_width = max(max_width, target_w)
                max_height = max(max_height, target_h)

    return max_width, max_height

def analyze_target_dimension_distribution(data, target_um_per_px):
    """Analyze the distribution of target dimensions"""
    widths = []
    heights = []

    for entry in data.values():
        orig_um_per_px = entry.get('um_per_px')
        if isinstance(orig_um_per_px, (list, tuple)):
            orig_um_per_px = orig_um_per_px[0]

        if orig_um_per_px and 'orig_width_px' in entry and 'orig_height_px' in entry:
            orig_w = entry['orig_width_px']
            orig_h = entry['orig_height_px']

            if orig_w and orig_h:
                target_w, target_h = calculate_target_dimensions(
                    orig_w, orig_h, orig_um_per_px, target_um_per_px
                )
                widths.append(target_w)
                heights.append(target_h)

    widths = np.array(widths)
    heights = np.array(heights)

    logging.info("Target dimension statistics (at %.4f um/px):", target_um_per_px)
    logging.info("  Width  - min: %d, median: %d, max: %d", widths.min(), int(np.median(widths)), widths.max())
    logging.info("  Height - min: %d, median: %d, max: %d", heights.min(), int(np.median(heights)), heights.max())
    logging.info("  95th percentile - width: %d, height: %d", int(np.percentile(widths, 95)), int(np.percentile(heights, 95)))

    # Count how many exceed different thresholds
    logging.info("Images exceeding various sizes:")
    for size in [512, 768, 1024, 1200]:
        exceeding = np.sum((widths > size) | (heights > size))
        pct = 100*exceeding/len(widths) if len(widths) > 0 else 0
        logging.info("  %d×%d: %d images (%.1f%%)", size, size, exceeding, pct)

    return widths, heights

def pad_to_square(image, target_size, is_mask=False):
    """
    Pad image to square dimensions, centering the content
    Uses white (255) for images, black (0) for masks
    """
    if len(image.shape) == 2:
        h, w = image.shape
        channels = None
    else:
        h, w, channels = image.shape

    # Simple: white (255) for images, black (0) for masks
    pad_value = 0 if is_mask else 255

    # Calculate padding
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if channels is None:
        padded = np.pad(image,
                       ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode='constant', constant_values=pad_value)
    else:
        padded = np.pad(image,
                       ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                       mode='constant', constant_values=pad_value)

    return padded

def resize_and_pad(image, target_width, target_height, target_size, is_mask=False):
    """
    Resize image to target dimensions and pad to square
    Converts everything to uint8 for consistent padding
    """
    # Resize to target physical scale
    if is_mask:
        # Use nearest neighbor for masks to preserve label values
        resized = resize(image, (target_height, target_width),
                        order=0, preserve_range=True, anti_aliasing=False)
        resized = np.clip(resized, 0, 255).astype(np.uint8)
    else:
        # Use bilinear for images
        resized = resize(image, (target_height, target_width),
                        order=1, preserve_range=True, anti_aliasing=True)

        # Convert to uint8 range (handle both uint8 and uint16 inputs)
        if resized.max() > 255:
            # Normalize from uint16 range (0-65535) to uint8 range (0-255)
            resized = (resized / 65535.0 * 255.0).astype(np.uint8)
        else:
            resized = np.clip(resized, 0, 255).astype(np.uint8)

    # Pad to square (white for images, black for masks)
    padded = pad_to_square(resized, target_size, is_mask=is_mask)

    # Crop if somehow larger than target
    if padded.shape[0] > target_size or padded.shape[1] > target_size:
        if len(padded.shape) == 2:
            padded = padded[:target_size, :target_size]
        else:
            padded = padded[:target_size, :target_size, :]

    return padded

def process_entry(key, entry, base_folder, output_images_dir, output_masks_dir, stats, target_um_per_px, target_size, save_debug=False):
    """
    Process a single entry: load raw image and mask, resize, pad, save
    Returns lstm_processed metadata to add to entry
    """
    try:
        # Get metadata
        main_id = entry.get('verification', {}).get('main_id')
        if not main_id:
            stats['errors'].append(f"{key}: No main_id")
            return None

        orig_um_per_px = entry.get('um_per_px')
        if orig_um_per_px is None:
            stats['errors'].append(f"{key}: No um_per_px")
            return None

        # Handle um_per_px as list or scalar
        if isinstance(orig_um_per_px, (list, tuple)):
            orig_um_per_px = orig_um_per_px[0]  # Use X dimension

        # Get paths
        raw_image_path_rel = entry.get('Best Z Filename')
        if not raw_image_path_rel:
            stats['errors'].append(f"{key}: No Best Z Filename")
            return None

        raw_image_path = base_folder / raw_image_path_rel
        if not raw_image_path.exists():
            stats['errors'].append(f"{key}: Raw image not found at {raw_image_path}")
            return None

        mask_path = None
        if 'predicted_mask_path' in entry:
            mask_path = Path(entry['predicted_mask_path'])
            if not mask_path.exists():
                stats['warnings'].append(f"{key}: Mask not found at {mask_path}")
                mask_path = None

        # Load raw image
        raw_image = imread(str(raw_image_path))
        if raw_image is None or raw_image.size == 0:
            stats['errors'].append(f"{key}: Could not read raw image")
            return None

        orig_height, orig_width = raw_image.shape[:2]

        # Load and resize mask to raw image dimensions
        mask_resized = None
        if mask_path:
            mask = imread(str(mask_path))
            if mask is not None and mask.size > 0:
                # Resize mask to raw image dimensions (unstretching it)
                # This aligns the mask with the raw image
                mask_resized = resize(mask, (orig_height, orig_width),
                                    order=0, preserve_range=True, anti_aliasing=False)
                mask_resized = mask_resized.astype(np.uint8)

                # DEBUG: Save intermediate steps for first few entries
                if save_debug and stats['processed'] < 5:
                    debug_dir = output_images_dir.parent / 'debug'
                    debug_dir.mkdir(exist_ok=True)

                    # Save raw image
                    imsave(str(debug_dir / f"{main_id}_1_raw.png"),
                          (raw_image / raw_image.max() * 255).astype(np.uint8) if raw_image.max() > 255 else raw_image,
                          check_contrast=False)
                    # Save original stretched mask
                    imsave(str(debug_dir / f"{main_id}_2_mask_stretched.png"), mask, check_contrast=False)
                    # Save unstretched mask
                    imsave(str(debug_dir / f"{main_id}_3_mask_unstretched.png"), mask_resized, check_contrast=False)
                    # Save overlay
                    if len(raw_image.shape) == 3:
                        overlay = raw_image.copy()
                        if overlay.max() > 255:
                            overlay = (overlay / 65535.0 * 255.0).astype(np.uint8)
                        overlay = overlay.astype(np.uint8)
                        overlay[mask_resized > 0] = [255, 0, 0]  # Red where mask is
                        imsave(str(debug_dir / f"{main_id}_4_overlay.png"), overlay, check_contrast=False)

                # Verify resize worked
                if mask_resized.shape[:2] != (orig_height, orig_width):
                    stats['warnings'].append(
                        f"{key}: Mask resize failed! Expected {(orig_height, orig_width)}, "
                        f"got {mask_resized.shape[:2]}"
                    )
                    mask_resized = None

        # Calculate target dimensions for target um/px
        target_width, target_height = calculate_target_dimensions(
            orig_width, orig_height, orig_um_per_px, target_um_per_px
        )

        # Resize and pad image (white padding for well background)
        image_processed = resize_and_pad(raw_image, target_width, target_height,
                                        target_size, is_mask=False)

        # Resize and pad mask (black padding for background label)
        mask_processed = None
        if mask_resized is not None:
            mask_processed = resize_and_pad(mask_resized, target_width, target_height,
                                           target_size, is_mask=True)

        # Save processed image
        image_output_path = output_images_dir / f"{main_id}.png"
        imsave(str(image_output_path), image_processed, check_contrast=False)

        # Save processed mask
        mask_output_path = None
        if mask_processed is not None:
            mask_output_path = output_masks_dir / f"{main_id}.png"
            imsave(str(mask_output_path), mask_processed, check_contrast=False)

        # Also save debug final if requested
        if save_debug and stats['processed'] < 5 and mask_processed is not None:
            debug_dir = output_images_dir.parent / 'debug'
            imsave(str(debug_dir / f"{main_id}_5_final_image.png"), image_processed, check_contrast=False)
            imsave(str(debug_dir / f"{main_id}_6_final_mask.png"), mask_processed, check_contrast=False)

        stats['processed'] += 1

        # Return metadata to add to entry
        return {
            'image_path': str(image_output_path),
            'mask_path': str(mask_output_path) if mask_output_path else None,
            'target_um_per_px': target_um_per_px,
            'final_size': target_size,
            'target_width_before_pad': target_width,
            'target_height_before_pad': target_height,
            'padding_type': 'white (255) for images, black (0) for masks'
        }

    except Exception as e:
        stats['errors'].append(f"{key}: {str(e)}")
        return None

def main():
    args = get_args()
    for key, value in args.__dict__.items():
        logging.info(f"{key}: {value}")

    logging.info("%s", '='*70)
    logging.info("LSTM IMAGE PREPROCESSING")
    logging.info("%s", '='*70)
    logging.info(f"Target physical scale: {args.target_um_per_px} um/px")
    logging.info(f"Target dimensions: {args.target_size}×{args.target_size} px")
    logging.info("Padding: WHITE (255) for images, BLACK (0) for masks")

    # Load data
    data_path = args.complete_series
    if not data_path.exists():
        logging.error("%s not found. Run filter_complete_series.py first.", data_path)
        return

    logging.info("Loading data from %s", data_path)
    data = load_json(data_path)
    logging.info("Loaded %d entries", len(data))

    # Determine base folder for raw images
    base_folder = args.raw_image_dir
    if not base_folder.exists():
        logging.error("Base folder does not exist: %s", base_folder)
        return

    # Analyze target dimensions
    if not args.skip_analysis:
        logging.info("%s", '='*70)
        logging.info("ANALYZING TARGET DIMENSIONS")
        logging.info("%s", '='*70)

        max_w, max_h = find_max_target_dimensions(data, args.target_um_per_px)
        logging.info("Maximum target dimensions:")
        logging.info("  Width:  %dpx", max_w)
        logging.info("  Height: %dpx", max_h)

        recommended_size = max(max_w, max_h)
        logging.info("Recommended TARGET_SIZE: %dpx", recommended_size)

        # Show distribution
        analyze_target_dimension_distribution(data, args.target_um_per_px)

        # Warn if TARGET_SIZE is too small
        if args.target_size < recommended_size:
            logging.info("%s", '='*70)
            logging.warning("[WARNING] Current TARGET_SIZE (%d) is SMALLER than needed!", args.target_size)
            logging.info("%s", '='*70)
            logging.info("  %dpx will be CROPPED from largest images", recommended_size - args.target_size)
            logging.info("  This means losing organoid content at the edges!")
            logging.info("Recommendation: Use --target-size %d", recommended_size)
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                logging.info("Aborted. Rerun with larger --target-size")
                return
        else:
            logging.info("[OK] TARGET_SIZE %d is sufficient for all images", args.target_size)

    # Create output directories
    output_base = args.out_dir / 'lstm_ready'
    output_images_dir = output_base / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir = output_base / 'masks'
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    logging.info("%s", '='*70)
    logging.info("PROCESSING IMAGES AND MASKS")
    logging.info("%s", '='*70)
    logging.info("Output directories:")
    logging.info("  Images: %s", output_images_dir)
    logging.info("  Masks:  %s", output_masks_dir)
    if args.save_debug:
        logging.info("  Debug:  %s", output_base / 'debug')

    # Process all entries
    logging.info("Processing %d entries...", len(data))

    stats = {
        'processed': 0,
        'errors': [],
        'warnings': []
    }

    orig_scales = []

    for key, entry in tqdm(data.items(), desc="Processing"):
        lstm_metadata = process_entry(key, entry, base_folder, output_images_dir,
                                      output_masks_dir, stats, args.target_um_per_px,
                                       args.target_size, save_debug=args.save_debug)
        if lstm_metadata:
            # Add lstm_processed field to entry
            entry['lstm_processed'] = lstm_metadata

            # Track original scale for stats
            orig_um_per_px = entry.get('um_per_px')
            if orig_um_per_px:
                if isinstance(orig_um_per_px, (list, tuple)):
                    orig_um_per_px = orig_um_per_px[0]
                orig_scales.append(orig_um_per_px)

    # Save updated data back to JSON
    logging.info("Saving updated data to %s", data_path)
    save_json(data_path, data)

    # logging.info summary
    logging.info("%s", '='*70)
    logging.info("PROCESSING SUMMARY")
    logging.info("%s", '='*70)
    logging.info("Successfully processed: %d / %d", stats['processed'], len(data))
    logging.info("Errors: %d", len(stats['errors']))
    logging.info("Warnings: %d", len(stats['warnings']))

    if stats['errors']:
        logging.info("First 10 errors:")
        for error in stats['errors'][:10]:
            logging.info("  - %s", error)

    if stats['warnings']:
        logging.info("First 10 warnings:")
        for warning in stats['warnings'][:10]:
            logging.info("  - %s", warning)

    logging.info("Output saved to:")
    logging.info("  Images: %s (white padding)", output_images_dir)
    logging.info("  Masks: %s (black padding)", output_masks_dir)
    logging.info("  Updated data: %s (added 'lstm_processed' field)", data_path)

    # Calculate final statistics
    if orig_scales:
        logging.info("%s", '='*70)
        logging.info("PHYSICAL SCALE VERIFICATION")
        logging.info("%s", '='*70)
        logging.info("Original scales ranged from %.4f to %.4f um/px", min(orig_scales), max(orig_scales))
        logging.info("All images now uniformly at %.4f um/px", args.target_um_per_px)
        logging.info("All images now uniformly %d×%d pixels", args.target_size, args.target_size)
        logging.info("[OK] Ready for LSTM training!")
        logging.info("For LSTM training:")
        logging.info("  1. Load complete_series_metadata_no_blanks.json for sequences")
        logging.info("  2. Load complete_series_data_no_blanks.json for entry data")
        logging.info("  3. Access processed images via entry['lstm_processed']['image_path']")

if __name__ == "__main__":
    main()
