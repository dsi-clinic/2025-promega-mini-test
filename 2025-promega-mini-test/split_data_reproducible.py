#!/usr/bin/env python3 -u
"""
Reproducible train/val split for image and metabolite models.

CRITICAL: Splits by ORGANOID, not by individual samples!
This ensures the same organoid across all timepoints stays together in train or val.
Prevents data leakage when training on early days to predict Dy30 outcomes.

Base mode: Only BA1+BA2 with both image and complete metabolite data
Switch 1: Image gets additional BA1/BA2 image-only samples
Switch 2: Include BA3/BA4 intersection (both image+metabolite)
Switch 3: Image gets ALL data from all 4 batches
"""
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import re

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ============================================================
# CONFIGURATION
# ============================================================
ALL_DATA_JSON = 'all_data.json'
RANDOM_SEED = 42  # Fixed seed for reproducibility
TEST_SIZE = 0.2  # 20% test set (held out)
VAL_SIZE = 0.1   # 10% validation set (within the 80% training set)
# Final ratios: 72% train / 8% val / 20% test (80/20 training/testing, 90/10 train/val within training)

# Good metabolites (based on IDOR/Promega restrictions)
# Always included:
# - GlucoseGlo ✓
# - GlutamateGlo ✓
# - LactateGlo ✓
# - PyruvateGlo ✓
#
# Conditionally included:
# - MalateGlo: included for days >10, excluded for days ≤10 (inclusive)
#
# Excluded metabolites:
# - BCAAGlo: completely excluded (do not use at all)
REQUIRED_METABOLITES = ['GlucoseGlo', 'GlutamateGlo', 'LactateGlo', 'PyruvateGlo']
MALATE_EXCLUSION_THRESHOLD_DAY = 10  # Don't use MalateGlo for days ≤10

# Target day for survey labels (labels come from Dy30)
LABEL_DAY = 'Dy30'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_majority_label(evaluations, min_votes=4):
    """Compute majority label from survey evaluations."""
    if not evaluations or len(evaluations) != 5:
        return None
    
    votes = {}
    for eval_data in evaluations:
        evaluation = eval_data.get('evaluation', '')
        if evaluation:
            votes[evaluation] = votes.get(evaluation, 0) + 1
    
    acceptable = votes.get('Acceptable', 0)
    not_acceptable = votes.get('Not Acceptable', 0)
    
    if acceptable >= min_votes:
        return 'Acceptable'
    elif not_acceptable >= min_votes:
        return 'Not Acceptable'
    else:
        return None

def extract_organoid_id(key):
    """
    Extract organoid ID without day from key.
    'BA1 96_1 Dy30 A1' -> 'BA1 96_1 A1'
    """
    match = re.match(r'^(.*)\s+Dy\d+\s+(.*)$', key)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return key

def extract_day_number(day_id):
    """
    Extract numeric day from dayID string.
    'Dy03' -> 3, 'Dy30' -> 30, returns None if invalid.
    """
    if not day_id:
        return None
    match = re.match(r'^Dy(\d+)$', day_id)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def has_complete_metabolites(metabolites):
    """Check if sample has all required metabolites with valid data."""
    if not metabolites:
        return False
    
    for met_name in REQUIRED_METABOLITES:
        if met_name not in metabolites:
            return False
        if 'concentration_uM' not in metabolites[met_name]:
            return False
        if metabolites[met_name]['concentration_uM'] is None:
            return False
    
    return True

def get_batch_prefix(ba_string):
    """Extract batch prefix (BA1, BA2, etc.) from full batch string."""
    if not ba_string:
        return None
    return ba_string.split()[0] if ' ' in ba_string else ba_string

def has_valid_image_data(record):
    """Check if record has valid processed image data."""
    return ('processed' in record and 
            record['processed'] and 
            'img_path' in record['processed'] and
            'mask_path' in record['processed'])

# ============================================================
# DATA COLLECTION FUNCTIONS
# ============================================================

def collect_organoid_data(all_data, batches=['BA1', 'BA2'], require_metabolites=True):
    """
    Collect all timepoints for organoids, grouped by organoid ID.
    Only include organoids that have Dy30 labels.
    
    Returns:
    - organoid_dict: {organoid_id: {'label': ..., 'timepoints': {...}}}
    """
    # First pass: get Dy30 labels for each organoid
    organoid_labels = {}
    
    for key, value in all_data.items():
        # Check if this is Dy30 with survey label
        if value.get('dayID') != LABEL_DAY:
            continue
        
        # Check batch
        batch = get_batch_prefix(value.get('BA'))
        if batch not in batches:
            continue
        
        # Get label from survey
        if 'survey' not in value:
            continue
        
        evaluations = value['survey'].get('evaluations', [])
        label = compute_majority_label(evaluations, min_votes=4)
        if label is None:
            continue
        
        # Extract organoid ID
        organoid_id = extract_organoid_id(key)
        organoid_labels[organoid_id] = label
    
    print(f"  Found {len(organoid_labels)} organoids with Dy30 labels in {batches}")
    
    # Second pass: collect all timepoints for labeled organoids
    organoid_data = {}
    
    for key, value in all_data.items():
        # Extract organoid ID and check if it has a label
        organoid_id = extract_organoid_id(key)
        if organoid_id not in organoid_labels:
            continue
        
        # Check batch
        batch = get_batch_prefix(value.get('BA'))
        if batch not in batches:
            continue
        
        # Check if has valid image data
        if not has_valid_image_data(value):
            continue
        
        # If metabolites required, check completeness
        has_metabolites = has_complete_metabolites(value.get('metabolites'))
        if require_metabolites and not has_metabolites:
            continue
        
        # Initialize organoid entry if needed
        if organoid_id not in organoid_data:
            organoid_data[organoid_id] = {
                'label': organoid_labels[organoid_id],
                'batch': batch,
                'timepoints': {}
            }
        
        # Add this timepoint
        day = value.get('dayID')
        # Merge Dy20 and Dy21 into Dy20_5 (they represent the same timepoint)
        if day in ['Dy20', 'Dy21']:
            day = 'Dy20_5'
        timepoint_data = {
            'img_path': value['processed']['img_path'],
            'mask_path': value['processed']['mask_path'],
            'day': day
        }
        
        # Add metabolites if present
        # Note: For image-only modes (switch1/switch3), this only runs if metabolites exist.
        # Image-only samples without metabolites will have no 'metabolites' field.
        if has_metabolites:
            metabolites_dict = {}
            
            # Extract both concentration_uM and initial_concentration for each required metabolite
            for met in REQUIRED_METABOLITES:
                met_data = value['metabolites'][met]
                # Store with full feature names matching the expected format
                metabolites_dict[f'{met}_concentration_uM'] = met_data.get('concentration_uM')
                metabolites_dict[f'{met}_initial_concentration'] = met_data.get('initial_concentration')
            
            # Conditionally include MalateGlo for days >10
            # (Only applies when metabolites are included - not relevant for image-only samples)
            day_num = extract_day_number(day)
            if day_num is not None and day_num > MALATE_EXCLUSION_THRESHOLD_DAY:
                if 'MalateGlo' in value.get('metabolites', {}):
                    malate_data = value['metabolites']['MalateGlo']
                    if 'concentration_uM' in malate_data and malate_data['concentration_uM'] is not None:
                        metabolites_dict['MalateGlo_concentration_uM'] = malate_data['concentration_uM']
                    if 'initial_concentration' in malate_data and malate_data['initial_concentration'] is not None:
                        metabolites_dict['MalateGlo_initial_concentration'] = malate_data['initial_concentration']
            
            timepoint_data['metabolites'] = metabolites_dict
        
        organoid_data[organoid_id]['timepoints'][day] = timepoint_data
    
    return organoid_data

# ============================================================
# SPLIT FUNCTIONS
# ============================================================

def split_by_organoid(organoid_data, random_seed=RANDOM_SEED, test_size=TEST_SIZE, val_size=VAL_SIZE):
    """
    Split organoids into train/val/test sets with stratification by label.
    
    Structure:
    1. First split: 80% training / 20% test (held out)
    2. Within 80% training: split into train/val (90% train, 10% val of training set)
    
    Returns train_data, val_data, test_data (all in same format as organoid_data)
    """
    if not organoid_data:
        return {}, {}, {}
    
    # Extract organoid IDs and labels
    organoid_ids = list(organoid_data.keys())
    labels = [organoid_data[oid]['label'] for oid in organoid_ids]
    
    # First split: 80% training / 20% test (held out)
    train_test_ids, test_ids = train_test_split(
        organoid_ids,
        test_size=test_size,
        stratify=labels,
        random_state=random_seed
    )
    
    # Extract labels for the training set
    train_test_labels = [organoid_data[oid]['label'] for oid in train_test_ids]
    
    # Second split: Within training set, split into train/val
    # val_size is relative to the training set (e.g., 0.1 = 10% of training set goes to val)
    train_ids, val_ids = train_test_split(
        train_test_ids,
        test_size=val_size,
        stratify=train_test_labels,
        random_state=random_seed
    )
    
    # Create train, val, and test dictionaries
    train_data = {oid: organoid_data[oid] for oid in train_ids}
    val_data = {oid: organoid_data[oid] for oid in val_ids}
    test_data = {oid: organoid_data[oid] for oid in test_ids}
    
    return train_data, val_data, test_data

# ============================================================
# OUTPUT FUNCTIONS
# ============================================================

def save_splits(train_data, val_data, test_data, output_prefix, mode_name):
    """Save train/val/test splits to JSON files."""
    output_dir = Path('data_splits')
    output_dir.mkdir(exist_ok=True)
    
    train_file = output_dir / f'{output_prefix}_train_{mode_name}.json'
    val_file = output_dir / f'{output_prefix}_val_{mode_name}.json'
    test_file = output_dir / f'{output_prefix}_test_{mode_name}.json'
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return train_file, val_file, test_file

def print_statistics(data_dict, name):
    """Print statistics about a dataset."""
    if not data_dict:
        print(f"  {name}: 0 organoids")
        return
    
    # Count organoids by label
    labels = [v['label'] for v in data_dict.values()]
    acceptable = labels.count('Acceptable')
    not_acceptable = labels.count('Not Acceptable')
    
    # Count total samples (all timepoints)
    total_samples = sum(len(v['timepoints']) for v in data_dict.values())
    
    # Count timepoints per day
    day_counts = {}
    for org_data in data_dict.values():
        for day in org_data['timepoints'].keys():
            day_counts[day] = day_counts.get(day, 0) + 1
    
    print(f"  {name}:")
    print(f"    - {len(data_dict)} organoids ({acceptable} Acceptable, {not_acceptable} Not Acceptable)")
    print(f"    - {total_samples} total samples across all timepoints")
    print(f"    - Days available: {sorted(day_counts.keys())}")
    for day in sorted(day_counts.keys()):
        print(f"      {day}: {day_counts[day]} samples")

# ============================================================
# MAIN MODES
# ============================================================

def run_base_mode(all_data):
    """
    Base mode: Only BA1+BA2 organoids with both image and complete metabolite data.
    Both models train on the exact same organoids.
    """
    print("\n" + "="*60)
    print("BASE MODE: BA1+BA2 Intersection Only")
    print("="*60)
    
    # Collect organoid data
    organoid_data = collect_organoid_data(
        all_data, 
        batches=['BA1', 'BA2'], 
        require_metabolites=True
    )
    
    print(f"\nCollected data for {len(organoid_data)} organoids with complete data")
    
    # Split by organoid: 80% training / 20% test, then split training into train/val
    train_data, val_data, test_data = split_by_organoid(organoid_data, random_seed=RANDOM_SEED)
    
    print("\nTrain/Val/Test Split:")
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation (within training)")
    print_statistics(test_data, "Test (held out)")
    
    # Save
    train_file, val_file, test_file = save_splits(train_data, val_data, test_data, 'both', 'base')
    print(f"\n✓ Saved: {train_file}")
    print(f"✓ Saved: {val_file}")
    print(f"✓ Saved: {test_file}")
    
    return organoid_data, train_data, val_data, test_data

def run_switch1_mode(all_data, intersection_organoids):
    """
    Switch 1: Image gets additional BA1/BA2 organoids (image-only).
    Metabolite still uses only intersection.
    """
    print("\n" + "="*60)
    print("SWITCH 1: Image Gets Additional BA1+BA2 Samples")
    print("="*60)
    
    # Collect all BA1/BA2 organoids with image data (metabolites optional)
    all_image_organoids = collect_organoid_data(
        all_data, 
        batches=['BA1', 'BA2'], 
        require_metabolites=False
    )
    
    print(f"Total BA1+BA2 organoids with image: {len(all_image_organoids)}")
    print(f"Additional image-only organoids: {len(all_image_organoids) - len(intersection_organoids)}")
    
    # Split by organoid: 80% training / 20% test, then split training into train/val
    train_data, val_data, test_data = split_by_organoid(all_image_organoids, random_seed=RANDOM_SEED)
    
    print("\nImage Train/Val/Test Split:")
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation (within training)")
    print_statistics(test_data, "Test (held out)")
    
    # Save
    train_file, val_file, test_file = save_splits(train_data, val_data, test_data, 'image', 'switch1')
    print(f"\n✓ Saved: {train_file}")
    print(f"✓ Saved: {val_file}")
    print(f"✓ Saved: {test_file}")
    
    return train_data, val_data, test_data

def run_switch2_mode(all_data):
    """
    Switch 2: Include BA3+BA4 organoids (intersection with both image+metabolite).
    Both models use this extended intersection.
    """
    print("\n" + "="*60)
    print("SWITCH 2: Include BA3+BA4 Intersection")
    print("="*60)
    
    # Collect from all batches
    organoid_data = collect_organoid_data(
        all_data, 
        batches=['BA1', 'BA2', 'BA3', 'BA4'], 
        require_metabolites=True
    )
    
    print(f"\nCollected data for {len(organoid_data)} organoids (all batches)")
    
    # Split by organoid: 80% training / 20% test, then split training into train/val
    train_data, val_data, test_data = split_by_organoid(organoid_data, random_seed=RANDOM_SEED)
    
    print("\nTrain/Val/Test Split:")
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation (within training)")
    print_statistics(test_data, "Test (held out)")
    
    # Save
    train_file, val_file, test_file = save_splits(train_data, val_data, test_data, 'both', 'switch2')
    print(f"\n✓ Saved: {train_file}")
    print(f"✓ Saved: {val_file}")
    print(f"✓ Saved: {test_file}")
    
    return train_data, val_data, test_data

def run_switch3_mode(all_data):
    """
    Switch 3: Image gets ALL organoids from all 4 batches.
    Metabolite still uses BA1+BA2 intersection only.
    """
    print("\n" + "="*60)
    print("SWITCH 3: Image Gets All Available Data")
    print("="*60)
    
    # Collect all organoids from all batches with image data
    organoid_data = collect_organoid_data(
        all_data, 
        batches=['BA1', 'BA2', 'BA3', 'BA4'], 
        require_metabolites=False
    )
    
    print(f"\nCollected data for {len(organoid_data)} organoids (all batches)")
    
    # Split by organoid: 80% training / 20% test, then split training into train/val
    train_data, val_data, test_data = split_by_organoid(organoid_data, random_seed=RANDOM_SEED)
    
    print("\nImage Train/Val/Test Split:")
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation (within training)")
    print_statistics(test_data, "Test (held out)")
    
    # Save
    train_file, val_file, test_file = save_splits(train_data, val_data, test_data, 'image', 'switch3')
    print(f"\n✓ Saved: {train_file}")
    print(f"✓ Saved: {val_file}")
    print(f"✓ Saved: {test_file}")
    
    return train_data, val_data, test_data

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reproducible train/val split for image and metabolite models (by organoid)'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='base',
        choices=['base', 'switch1', 'switch2', 'switch3', 'all'],
        help='Split mode: base, switch1, switch2, switch3, or all'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading {ALL_DATA_JSON}...", flush=True)
    with open(ALL_DATA_JSON) as f:
        all_data = json.load(f)
    print(f"✓ Loaded {len(all_data)} records", flush=True)
    
    print(f"\nIMPORTANT: Splitting by ORGANOID, not by individual samples!")
    print(f"   This prevents data leakage when training across timepoints.")
    print(f"\nUsing fixed random seed: {RANDOM_SEED}")
    print(f"Split structure: 80% Training / 20% Test (held out)")
    print(f"Within Training: {int((1-VAL_SIZE)*100)}% Train / {int(VAL_SIZE*100)}% Val")
    print(f"Final ratios: ~{int((1-TEST_SIZE)*(1-VAL_SIZE)*100)}% Train / ~{int((1-TEST_SIZE)*VAL_SIZE*100)}% Val / {int(TEST_SIZE*100)}% Test")
    print(f"Labels from: {LABEL_DAY}")
    
    # Run requested mode(s)
    organoid_data = None
    if args.mode == 'base' or args.mode == 'all':
        organoid_data, train_base, val_base, test_base = run_base_mode(all_data)
    
    if args.mode == 'switch1' or args.mode == 'all':
        if args.mode == 'switch1' and organoid_data is None:
            organoid_data, _, _, _ = run_base_mode(all_data)
        if organoid_data is not None:
            run_switch1_mode(all_data, organoid_data)
    
    if args.mode == 'switch2' or args.mode == 'all':
        run_switch2_mode(all_data)
    
    if args.mode == 'switch3' or args.mode == 'all':
        run_switch3_mode(all_data)
    
    print("\n" + "="*60)
    print("✓ Split complete! All files saved to data_splits/")
    print("="*60)
    print("\nData format: Each organoid has all its timepoints together.")
    print("Use this to train on early days and predict Dy30 outcomes!")
    print("="*60)

if __name__ == '__main__':
    main()