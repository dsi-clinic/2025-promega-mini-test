# analysis/images/series/check_sizes.py
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from skimage.io import imread
import numpy as np

from config import OUTPUT_FOLDER

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def check_dimensions(data, series_metadata, sample_size=50):
    """
    Check image and mask dimensions across the dataset
    Focus on physical scale consistency - critical for LSTM training
    """
    print(f"\n{'='*70}")
    print("IMAGE DIMENSION & PHYSICAL SCALE ANALYSIS")
    print(f"{'='*70}")
    
    # Collect metadata dimensions
    entries_analyzed = 0
    scale_data = []
    dimension_data = []
    missing_processed = 0
    
    for key, entry in tqdm(data.items(), desc="Analyzing metadata"):
        if 'processed' not in entry:
            missing_processed += 1
            continue
        
        proc = entry['processed']
        entries_analyzed += 1
        
        entry_info = {
            'key': key,
            'orig_width_px': proc.get('orig_width_px'),
            'orig_height_px': proc.get('orig_height_px'),
            'orig_um_per_px_x': proc.get('orig_um_per_px_x'),
            'orig_um_per_px_y': proc.get('orig_um_per_px_y'),
            'final_um_per_px_x': proc.get('final_um_per_px_x'),
            'final_um_per_px_y': proc.get('final_um_per_px_y'),
            'img_path': proc.get('img_path', '')
        }
        
        scale_data.append(entry_info)
        
        # Infer processed dimensions from path
        if 'resized_512x384' in entry_info['img_path']:
            dimension_data.append((512, 384))
    
    print(f"\nEntries analyzed: {entries_analyzed} / {len(data)}")
    if missing_processed > 0:
        print(f"[WARNING] {missing_processed} entries missing 'processed' field!")
    
    # CRITICAL CHECK 1: Physical scale uniformity
    print(f"\n{'='*70}")
    print("CRITICAL CHECK 1: Physical Scale Uniformity")
    print(f"{'='*70}")
    
    final_scales_x = [e['final_um_per_px_x'] for e in scale_data if e['final_um_per_px_x'] is not None]
    final_scales_y = [e['final_um_per_px_y'] for e in scale_data if e['final_um_per_px_y'] is not None]
    
    if final_scales_x:
        unique_x = len(set([round(s, 6) for s in final_scales_x]))
        print(f"\nFinal um/px (X-axis):")
        print(f"  Mean: {np.mean(final_scales_x):.6f}")
        print(f"  Std:  {np.std(final_scales_x):.6f}")
        print(f"  Range: {min(final_scales_x):.6f} - {max(final_scales_x):.6f}")
        print(f"  Unique values (rounded to 6 decimals): {unique_x}")
        
        if unique_x == 1:
            print(f"  [OK] Uniform X scale across all images")
        else:
            print(f"  [PROBLEM] Non-uniform X scale! LSTM will see inconsistent physical sizes!")
            scale_counts = Counter([round(s, 6) for s in final_scales_x])
            print(f"  Top 5 scale values:")
            for scale, count in scale_counts.most_common(5):
                print(f"    {scale:.6f} um/px: {count} images")
    
    if final_scales_y:
        unique_y = len(set([round(s, 6) for s in final_scales_y]))
        print(f"\nFinal um/px (Y-axis):")
        print(f"  Mean: {np.mean(final_scales_y):.6f}")
        print(f"  Std:  {np.std(final_scales_y):.6f}")
        print(f"  Range: {min(final_scales_y):.6f} - {max(final_scales_y):.6f}")
        print(f"  Unique values (rounded to 6 decimals): {unique_y}")
        
        if unique_y == 1:
            print(f"  [OK] Uniform Y scale across all images")
        else:
            print(f"  [PROBLEM] Non-uniform Y scale!")
    
    # CRITICAL CHECK 2: Aspect ratio preservation (X scale == Y scale?)
    print(f"\n{'='*70}")
    print("CRITICAL CHECK 2: Aspect Ratio Preservation")
    print(f"{'='*70}")
    
    if final_scales_x and final_scales_y:
        scale_ratios = [x/y for x, y in zip(final_scales_x, final_scales_y)]
        print(f"\nFinal um/px X / um/px Y ratio:")
        print(f"  Mean: {np.mean(scale_ratios):.6f}")
        print(f"  Std:  {np.std(scale_ratios):.6f}")
        print(f"  Range: {min(scale_ratios):.6f} - {max(scale_ratios):.6f}")
        
        if np.std(scale_ratios) < 0.001 and abs(np.mean(scale_ratios) - 1.0) < 0.01:
            print(f"  [OK] Aspect ratios preserved (X scale ≈ Y scale)")
        else:
            print(f"  [PROBLEM] Aspect ratios NOT preserved!")
            print(f"  Images are stretched/squashed - this distorts organoid morphology!")
    
    # CRITICAL CHECK 3: Original scale variation
    print(f"\n{'='*70}")
    print("CRITICAL CHECK 3: Original Image Scale Variation")
    print(f"{'='*70}")
    
    orig_scales_x = [e['orig_um_per_px_x'] for e in scale_data if e['orig_um_per_px_x'] is not None]
    
    if orig_scales_x:
        unique_orig = len(set([round(s, 6) for s in orig_scales_x]))
        print(f"\nOriginal um/px (X-axis):")
        print(f"  Mean: {np.mean(orig_scales_x):.6f}")
        print(f"  Std:  {np.std(orig_scales_x):.6f}")
        print(f"  Range: {min(orig_scales_x):.6f} - {max(orig_scales_x):.6f}")
        print(f"  Unique values: {unique_orig}")
        
        if unique_orig == 1:
            print(f"  [INFO] All original images had same scale - good!")
        else:
            print(f"  [INFO] Original images had {unique_orig} different scales")
            print(f"  This is OK if final scales are uniform")
    
    # Check processed dimensions
    print(f"\n{'='*70}")
    print("PROCESSED IMAGE DIMENSIONS")
    print(f"{'='*70}")
    
    if dimension_data:
        dim_counts = Counter(dimension_data)
        print(f"\nProcessed dimensions:")
        for dims, count in dim_counts.most_common():
            print(f"  {dims[0]}×{dims[1]}: {count} images")
        
        if len(dim_counts) == 1:
            width, height = list(dim_counts.keys())[0]
            print(f"  [OK] All images uniformly processed to {width}×{height}")
            proc_aspect = width / height
            print(f"  Processed aspect ratio: {proc_aspect:.4f} ({width}/{height} = {width/height:.4f})")
        else:
            print(f"  [PROBLEM] Multiple processed dimensions found!")
    
    # Sample actual files to verify
    print(f"\n{'='*70}")
    print(f"SAMPLING ACTUAL IMAGE FILES (n={sample_size}):")
    print(f"{'='*70}")
    
    sample_keys = list(data.keys())[:sample_size]
    actual_dims = {'images': [], 'masks': []}
    missing_files = {'images': 0, 'masks': 0}
    
    for key in tqdm(sample_keys, desc="Sampling files"):
        entry = data[key]
        
        if 'processed' not in entry:
            continue
        
        proc = entry['processed']
        
        # Check image
        if 'img_path' in proc:
            img_path = Path(proc['img_path'])
            if img_path.exists():
                try:
                    img = imread(str(img_path))
                    actual_dims['images'].append(img.shape)
                except Exception as e:
                    print(f"  Error reading {img_path}: {e}")
            else:
                missing_files['images'] += 1
        
        # Check mask
        if 'mask_path' in proc:
            mask_path = Path(proc['mask_path'])
            if mask_path.exists():
                try:
                    mask = imread(str(mask_path))
                    actual_dims['masks'].append(mask.shape)
                except Exception as e:
                    print(f"  Error reading {mask_path}: {e}")
            else:
                missing_files['masks'] += 1
    
    # Analyze sampled dimensions
    if actual_dims['images']:
        img_dim_counts = Counter(actual_dims['images'])
        print(f"\nActual IMAGE dimensions from {len(actual_dims['images'])} samples:")
        for dims, count in img_dim_counts.most_common():
            print(f"  {dims}: {count} images")
        
        if len(img_dim_counts) == 1:
            print(f"  [OK] All sampled images have consistent dimensions")
        else:
            print(f"  [WARNING] Multiple dimensions found in sampled images!")
    
    if actual_dims['masks']:
        mask_dim_counts = Counter(actual_dims['masks'])
        print(f"\nActual MASK dimensions from {len(actual_dims['masks'])} samples:")
        for dims, count in mask_dim_counts.most_common():
            print(f"  {dims}: {count} masks")
        
        if len(mask_dim_counts) == 1:
            print(f"  [OK] All sampled masks have consistent dimensions")
        else:
            print(f"  [WARNING] Multiple dimensions found in sampled masks!")
    
    # Check if images and masks match
    if actual_dims['images'] and actual_dims['masks']:
        print(f"\nImage-Mask dimension matching:")
        for img_dim in img_dim_counts.keys():
            if img_dim in mask_dim_counts:
                print(f"  [OK] Images and masks both have {img_dim}")
            else:
                print(f"  [WARNING] Images have {img_dim} but masks don't!")
    
    if missing_files['images'] > 0:
        print(f"\n[WARNING] {missing_files['images']} image files not found in sample")
    if missing_files['masks'] > 0:
        print(f"[WARNING] {missing_files['masks']} mask files not found in sample")
    
    # Final recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR LSTM TRAINING:")
    print(f"{'='*70}")
    
    problems = []
    
    # Check if scales are uniform
    if final_scales_x:
        if len(set([round(s, 6) for s in final_scales_x])) > 1:
            problems.append("Non-uniform X scale")
    
    if final_scales_y:
        if len(set([round(s, 6) for s in final_scales_y])) > 1:
            problems.append("Non-uniform Y scale")
    
    # Check if aspect ratio preserved
    if final_scales_x and final_scales_y:
        scale_ratios = [x/y for x, y in zip(final_scales_x, final_scales_y)]
        if abs(np.mean(scale_ratios) - 1.0) > 0.01:
            problems.append("Aspect ratios not preserved (images stretched)")
    
    if not problems:
        print("\n[OK] Images are properly processed for LSTM training!")
        print("     - Uniform physical scale across all images")
        print("     - Aspect ratios preserved")
        print("     - Consistent dimensions")
        print("\nYou can proceed to LSTM training.")
    else:
        print("\n[ACTION REQUIRED] Issues detected:")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")
        
        print("\nRECOMMENDATION:")
        print("  Reprocess images with the following strategy:")
        print("  1. Determine target physical scale (e.g., 4.0 um/px)")
        print("  2. Resize all images to that scale (maintaining aspect ratio)")
        print("  3. Pad to uniform dimensions (e.g., 512×512 with black padding)")
        print("  4. This ensures:")
        print("     - Uniform physical scale (um/px) across all images")
        print("     - No distortion (aspect ratio preserved)")
        print("     - Uniform input size for LSTM")

def main():
    print("Loading complete series data...")
    data_path = OUTPUT_FOLDER / 'complete_series_data_no_blanks.json'
    metadata_path = OUTPUT_FOLDER / 'complete_series_metadata_no_blanks.json'
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run filter_complete_series.py first.")
        return
    
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Run filter_complete_series.py first.")
        return
    
    data = load_json(data_path)
    series_metadata = load_json(metadata_path)
    
    print(f"Loaded {len(data)} entries")
    print(f"Loaded {len(series_metadata)} complete series")
    
    check_dimensions(data, series_metadata, sample_size=100)

if __name__ == "__main__":
    main()