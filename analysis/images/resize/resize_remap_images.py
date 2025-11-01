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
    TARGET_WIDTH, TARGET_HEIGHT, TARGET_SIZE,
    RAW_IMAGE_MAPPING_JSON, INFER_AUTO_PROCESSED_DIR as OUTPUT_DIR
)

# Import for metadata reading
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ WARNING: PIL not available, metadata extraction disabled")

# --------------- configuration -----------------
INTERPOLATION = cv2.INTER_LINEAR

# --------------- metadata extraction -----------------------
def extract_physical_size(img_path: Path):
    """Extract physical dimensions from ImageJ TIFF metadata."""
    if not PIL_AVAILABLE:
        return None, None, None, None
    
    try:
        with Image.open(img_path) as img:
            width_px, height_px = img.size
            
            if hasattr(img, 'tag_v2'):
                # Get ImageDescription
                image_desc = img.tag_v2.get(270, "")
                if isinstance(image_desc, bytes):
                    image_desc = image_desc.decode('utf-8', errors='ignore')
                
                # Check for ImageJ micron unit (handle escaped unicode)
                has_micron_unit = (
                    'unit=µm' in image_desc or 
                    'unit=um' in image_desc.lower() or
                    'unit=\\u00b5m' in image_desc.lower() or  # Escaped!
                    'unit=\u00b5m' in image_desc.lower() or
                    'unit=micron' in image_desc.lower()
                )
                
                # Get resolution (should be pixels per micron for ImageJ)
                x_resolution = img.tag_v2.get(282)
                y_resolution = img.tag_v2.get(283)
                resolution_unit = img.tag_v2.get(296)
                
                # ImageJ format: ResolutionUnit=1, resolution is pixels/micron
                if resolution_unit == 1 and has_micron_unit and x_resolution and y_resolution:
                    if x_resolution > 0 and y_resolution > 0:
                        um_per_px_x = 1.0 / float(x_resolution)
                        um_per_px_y = 1.0 / float(y_resolution)
                        
                        # Sanity check (0.1-100 um/px is reasonable)
                        if 0.1 <= um_per_px_x <= 100 and 0.1 <= um_per_px_y <= 100:
                            width_um = width_px * um_per_px_x
                            height_um = height_px * um_per_px_y
                            return width_um, height_um, um_per_px_x, um_per_px_y
            
    except Exception:
        pass
    
    return None, None, None, None

def is_stitched(img_id: str, info: dict) -> bool:
    """Determine if an image is stitched based on naming or metadata."""
    # Add your stitching detection logic here
    # Common patterns: "stitched", "montage", "mosaic" in filename
    filename = info.get('Best Z Filename', '').lower()
    img_id_lower = img_id.lower()
    
    return (
        'stitched' in filename or 
        'stitched' in img_id_lower
    )

# --------------- helpers -----------------------
def norm(s: str) -> str:
    return s.lower().replace(' ', '') if isinstance(s, str) else ''

def ba_match(json_ba: str, batch_id: str) -> bool:
    return norm(json_ba).startswith(norm(batch_id))

# --------------- main functions ----------------
def process_batch(batch_num: int, day_num: int, overwrite: bool, rewrite_json_only: bool):
    with RAW_IMAGE_MAPPING_JSON.open() as f:
        full_mapping = json.load(f)
    entries = full_mapping.get("entries", {})
    base_folder = full_mapping.get("_base_folder")

    batch_prefix = norm(f"BA{batch_num}")
    all_ba_keys = set(norm(info['BA']) for info in entries.values() if 'BA' in info)
    batch_ids = sorted(b for b in all_ba_keys if b.startswith(batch_prefix))

    for batch_id in batch_ids:
        create_mapping(entries, base_folder, batch_id, day_num, overwrite, rewrite_json_only)


def create_mapping(entries: dict, base_folder: str, batch_id: str, day_num: int, 
                   overwrite: bool, rewrite_json_only: bool):
    day_id = f"Dy{day_num:02d}"
    safe_batch = batch_id.replace(' ', '_')
    output_dir = OUTPUT_DIR / f"{safe_batch}_{day_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"image_mapping_{safe_batch}_{day_id}_processed.json"

    if overwrite and not rewrite_json_only and output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if output_json.exists() and not rewrite_json_only and not overwrite:
        print(f"Skipping because mapping already exists: {output_json}")
        return

    # If rewrite_json_only, load existing mapping to preserve img_path
    existing_mapping = {}
    if rewrite_json_only and output_json.exists():
        with output_json.open() as f:
            existing_mapping = json.load(f)
        print(f"Rewriting JSON only (preserving existing images): {output_json}")

    matches = [
        (img_id, info['Best Z Filename'], info.get('um_per_px'))
        for img_id, info in entries.items()
        if norm(info.get('dayID', '')) == norm(day_id) and ba_match(info.get('BA', ''), batch_id)
    ]

    new_mapping = {}
    stitched_missing_calibration = []
    
    for img_id, img_path_str, orig_um_px in matches:
        info = entries[img_id]
        verification = info.get("verification", {})
        main_id = verification.get("main_id", img_id.replace(' ', '_'))

        img_path = Path(base_folder) / img_path_str
        if not img_path.exists():
            print(f"Skipped missing: {img_path}")
            continue

        # Check if this is a stitched image
        is_stitched_img = is_stitched(img_id, info)

        # Get image dimensions
        if rewrite_json_only:
            # Load from existing mapping to get orig dimensions
            if img_id in existing_mapping:
                orig_w = existing_mapping[img_id].get('orig_width_px')
                orig_h = existing_mapping[img_id].get('orig_height_px')
            else:
                # Need to read image to get dims
                img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_raw is None:
                    print(f"Skipped unreadable: {img_path}")
                    continue
                orig_h, orig_w = img_raw.shape[:2]
        else:
            # Full processing mode
            img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_raw is None:
                print(f"Skipped unreadable: {img_path}")
                continue
            orig_h, orig_w = img_raw.shape[:2]

        # STEP 1: Try to extract from metadata (for ALL images)
        width_um_meta, height_um_meta, um_px_x_meta, um_px_y_meta = extract_physical_size(img_path)
        
        if um_px_x_meta is not None:
            # SUCCESS: Use metadata
            um_x = um_px_x_meta
            um_y = um_px_y_meta
            calibration_source = "metadata"
            if is_stitched_img:
                print(f"✓ {main_id} [STITCHED]: {width_um_meta:.1f}×{height_um_meta:.1f}μm ({um_x:.3f}×{um_y:.3f} μm/px) from metadata")
            else:
                print(f"  {main_id}: {width_um_meta:.1f}×{height_um_meta:.1f}μm from metadata")
                
        elif not is_stitched_img and orig_um_px is not None:
            # FALLBACK: Non-stitched can use JSON um/px
            if isinstance(orig_um_px, (list, tuple)) and len(orig_um_px) == 2:
                um_x, um_y = orig_um_px
            else:
                um_x = um_y = orig_um_px
            calibration_source = "json"
            print(f"  {main_id}: Using JSON um/px ({um_x:.3f})")
            
        else:
            # NO CALIBRATION
            if is_stitched_img:
                stitched_missing_calibration.append(main_id)
                print(f"⚠️  {main_id} [STITCHED]: NO CALIBRATION FOUND!")
            else:
                print(f"⚠️  {main_id}: No calibration available")
            um_x = um_y = None
            calibration_source = "none"

        # Process or reuse image path
        if rewrite_json_only:
            # Keep existing image path
            if img_id in existing_mapping:
                out_path = existing_mapping[img_id]['img_path']
            else:
                out_path = str(output_dir / f"{main_id}.png")
        else:
            # Resize and save image
            img_final = cv2.resize(img_raw, TARGET_SIZE, INTERPOLATION)
            out_path = output_dir / f"{main_id}.png"
            cv2.imwrite(str(out_path), img_final)
            out_path = str(out_path)

        # Calculate final um/px after resize
        final_um_per_px_x = final_um_per_px_y = None
        if um_x is not None and um_y is not None:
            scale_x = orig_w / TARGET_WIDTH
            scale_y = orig_h / TARGET_HEIGHT
            final_um_per_px_x = um_x * scale_x
            final_um_per_px_y = um_y * scale_y

        new_mapping[img_id] = {
            "img_path": out_path,
            "main_id": main_id,
            "orig_width_px": orig_w,
            "orig_height_px": orig_h,
            "orig_um_per_px_x": um_x,
            "orig_um_per_px_y": um_y,
            "final_um_per_px_x": final_um_per_px_x,
            "final_um_per_px_y": final_um_per_px_y,
            "calibration_source": calibration_source,
            "is_stitched": is_stitched_img
        }

    # Save updated mapping
    with output_json.open('w') as f:
        json.dump(new_mapping, f, indent=2)

    mode_str = "Updated JSON for" if rewrite_json_only else "Processed"
    print(f"\n✓ {mode_str} {batch_id}: {len(new_mapping)} images")
    print(f"  Mapping: {output_json}")
    
    # Summary statistics
    meta_count = sum(1 for v in new_mapping.values() if v['calibration_source'] == 'metadata')
    json_count = sum(1 for v in new_mapping.values() if v['calibration_source'] == 'json')
    none_count = sum(1 for v in new_mapping.values() if v['calibration_source'] == 'none')
    stitched_count = sum(1 for v in new_mapping.values() if v.get('is_stitched'))
    
    print(f"  Calibration sources: {meta_count} metadata, {json_count} JSON, {none_count} missing")
    print(f"  Stitched images: {stitched_count}")
    
    # Warning for stitched images without calibration
    if stitched_missing_calibration:
        print(f"\nWARNING: {len(stitched_missing_calibration)} stitched images missing calibration:")
        for main_id in stitched_missing_calibration:
            print(f"    - {main_id}")

# --------------- CLI entrypoint ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process microscopy images and extract calibration metadata"
    )
    parser.add_argument('--batches', required=True, 
                       help='Comma-separated batch numbers, e.g. 1,2,3')
    parser.add_argument('--days', required=True, 
                       help='Comma-separated day numbers, e.g. 3,6,8')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Remove existing processed folders first')
    parser.add_argument('--rewrite-json-only', action='store_true',
                       help='Only update JSON files without reprocessing images')
    args = parser.parse_args()

    if args.rewrite_json_only:
        print("JSON-ONLY MODE: Will update metadata without reprocessing images\n")
    else:
        print("FULL PROCESSING MODE: Will resize images and update metadata\n")

    batches = [int(x) for x in args.batches.split(',')]
    days = [int(x) for x in args.days.split(',')]

    for batch in batches:
        for day in days:
            print(f"\n{'='*70}")
            print(f"Batch {batch}, Day {day}")
            print('='*70)
            process_batch(batch, day, args.overwrite, args.rewrite_json_only)

    print("\n✓ All done!")