# analysis/images/series_data_prep/preprocess_for_lstm.py
"""
Simple LSTM preprocessing: Just resize images and masks to target um/px and pad to square
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

from config import OUTPUT_FOLDER, RAW_IMAGE_DATA

# Target physical scale and dimensions
TARGET_UM_PER_PX = 8.0
TARGET_SIZE = 575

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def pad_to_square(image, target_size, is_mask=False):
    """
    Pad image to target_size WITHOUT scaling - ONLY adds padding
    """
    h, w = image.shape[:2]
    
    # Check if image fits
    if h > target_size or w > target_size:
        print(f"WARNING: Image {w}×{h} exceeds target {target_size}!")
        # Crop to fit
        h = min(h, target_size)
        w = min(w, target_size)
        if len(image.shape) == 2:
            image = image[:h, :w]
        else:
            image = image[:h, :w, :]
    
    # Calculate padding to center the image
    pad_h = target_size - h
    pad_w = target_size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if is_mask:
        # Black padding for masks
        if len(image.shape) == 2:
            padded = np.pad(image, 
                           ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode='constant', constant_values=0)
        else:
            padded = np.pad(image,
                           ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                           mode='constant', constant_values=0)
    else:
        # Edge padding for images
        if len(image.shape) == 2:
            padded = np.pad(image,
                           ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode='edge')
        else:
            padded = np.pad(image,
                           ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                           mode='edge')
    
    return padded

def process_entry(key, entry, base_folder, output_images_dir, output_masks_dir, stats):
    """Simple processing: load, resize, pad, save"""
    try:
        main_id = entry.get('main_id')
        if not main_id:
            stats['errors'].append(f"{key}: No main_id")
            return None
        
        # Get original um/px
        orig_um_per_px = entry.get('um_per_px')
        if orig_um_per_px is None:
            stats['errors'].append(f"{key}: No um_per_px")
            return None
        
        if isinstance(orig_um_per_px, (list, tuple)):
            orig_um_per_px = orig_um_per_px[0]
        
        # Get Best Z image path
        raw_image_path_rel = entry.get('Best Z Filename')
        if not raw_image_path_rel:
            stats['errors'].append(f"{key}: No Best Z Filename")
            return None
        
        raw_image_path = base_folder / raw_image_path_rel
        if not raw_image_path.exists():
            stats['errors'].append(f"{key}: Image not found at {raw_image_path}")
            return None
        
        # Get mask path
        mask_path = None
        if 'processed' in entry and 'mask_path' in entry['processed']:
            mask_path = Path(entry['processed']['mask_path'])
            if not mask_path.exists():
                mask_path = None
        
        # ===== PROCESS IMAGE =====
        raw_image = imread(str(raw_image_path))
        if raw_image is None or raw_image.size == 0:
            stats['errors'].append(f"{key}: Could not read image")
            return None
        
        # Convert grayscale to RGB
        if raw_image.ndim == 2:
            raw_image = np.stack([raw_image] * 3, axis=-1)
        
        # Get original dimensions
        orig_h, orig_w = raw_image.shape[:2]
        
        # Calculate scale factor to achieve target um/px
        scale_factor = orig_um_per_px / TARGET_UM_PER_PX
        scaled_h = int(orig_h * scale_factor)
        scaled_w = int(orig_w * scale_factor)
        
        # Scale image to target um/px
        scaled_image = resize(raw_image, (scaled_h, scaled_w),
                             order=1, preserve_range=True, anti_aliasing=True)
        
        if scaled_image.max() > 255:
            scaled_image = (scaled_image / 65535.0 * 255.0).astype(np.uint8)
        else:
            scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)
        
        # Pad to square (NO additional scaling!)
        image_processed = pad_to_square(scaled_image, TARGET_SIZE, is_mask=False)
        
        # ===== PROCESS MASK =====
        mask_processed = None
        if mask_path:
            mask = imread(str(mask_path))
            if mask is not None and mask.size > 0:
                # Unstretch mask to original aspect ratio
                mask_unstretched = resize(mask, (orig_h, orig_w),
                                         order=0, preserve_range=True, anti_aliasing=False)
                mask_unstretched = mask_unstretched.astype(np.uint8)
                
                # Scale mask to same dimensions as scaled image
                mask_scaled = resize(mask_unstretched, (scaled_h, scaled_w),
                                    order=0, preserve_range=True, anti_aliasing=False)
                mask_scaled = mask_scaled.astype(np.uint8)
                
                # Pad to square (NO additional scaling!)
                mask_processed = pad_to_square(mask_scaled, TARGET_SIZE, is_mask=True)
        
        # ===== DEBUG OUTPUT =====
        if stats['processed'] < 5:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {main_id}")
            print(f"{'='*60}")
            print(f"Original: {orig_w}×{orig_h} at {orig_um_per_px:.3f} µm/px")
            print(f"Scaled to: {scaled_w}×{scaled_h} at {TARGET_UM_PER_PX} µm/px")
            print(f"Final padded: {image_processed.shape}")
            if mask_processed is not None:
                print(f"Mask: {mask.shape} → {mask_unstretched.shape} → {mask_scaled.shape} → {mask_processed.shape}")
            print(f"{'='*60}\n")
        
        # ===== SAVE =====
        image_output_path = output_images_dir / f"{main_id}.png"
        imsave(str(image_output_path), image_processed, check_contrast=False)
        
        mask_output_path = None
        if mask_processed is not None:
            mask_output_path = output_masks_dir / f"{main_id}.png"
            imsave(str(mask_output_path), mask_processed, check_contrast=False)
        
        stats['processed'] += 1
        
        return {
            'image_path': str(image_output_path),
            'mask_path': str(mask_output_path) if mask_output_path else None,
            'target_um_per_px': TARGET_UM_PER_PX,
            'final_size': TARGET_SIZE,
            'scaled_width': scaled_w,
            'scaled_height': scaled_h,
        }
        
    except Exception as e:
        stats['errors'].append(f"{key}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Simple LSTM preprocessing - just resize and pad'
    )
    parser.add_argument('--target-um-per-px', type=float, default=8.0,
                       help='Target physical scale in um per pixel')
    parser.add_argument('--target-size', type=int, default=575,
                       help='Target square size in pixels')
    parser.add_argument('--base-folder', type=str, default=None,
                       help='Base folder for raw images')
    args = parser.parse_args()
    
    global TARGET_UM_PER_PX, TARGET_SIZE
    TARGET_UM_PER_PX = args.target_um_per_px
    TARGET_SIZE = args.target_size
    
    print(f"\n{'='*70}")
    print("SIMPLE LSTM PREPROCESSING")
    print(f"{'='*70}")
    print(f"Target: {TARGET_UM_PER_PX} um/px, {TARGET_SIZE}×{TARGET_SIZE} px")
    
    # Load data
    data_path = OUTPUT_FOLDER / 'complete_series_data_no_blanks.json'
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return
    
    print(f"\nLoading data from {data_path}")
    data = load_json(data_path)
    print(f"Loaded {len(data)} entries")
    
    # Base folder
    base_folder = Path(args.base_folder) if args.base_folder else RAW_IMAGE_DATA
    print(f"Raw images: {base_folder}")
    
    if not base_folder.exists():
        print(f"ERROR: Base folder does not exist: {base_folder}")
        return
    
    # Output directories
    output_base = OUTPUT_FOLDER / 'lstm_ready'
    output_images_dir = output_base / 'images'
    output_masks_dir = output_base / 'masks'
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("PROCESSING")
    print(f"{'='*70}")
    print(f"Output:")
    print(f"  Images: {output_images_dir}")
    print(f"  Masks:  {output_masks_dir}")
    
    # Process
    stats = {'processed': 0, 'errors': [], 'warnings': []}
    
    for key, entry in tqdm(data.items(), desc="Processing"):
        result = process_entry(key, entry, base_folder, output_images_dir, 
                              output_masks_dir, stats)
        if result:
            entry['lstm_processed'] = result
    
    # Save
    print(f"\nSaving updated data to {data_path}")
    save_json(data_path, data)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Processed: {stats['processed']} / {len(data)}")
    print(f"Errors: {len(stats['errors'])}")
    
    if stats['errors']:
        print(f"\nFirst 10 errors:")
        for error in stats['errors'][:10]:
            print(f"  - {error}")
    
    print(f"\n✅ Done! Images ready for LSTM training")

if __name__ == "__main__":
    main()