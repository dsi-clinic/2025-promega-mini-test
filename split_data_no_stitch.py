#!/usr/bin/env python3 -u
"""
Reproducible train/val/test split - BASE MODE ONLY
WITH EXCLUSION OF STITCHED/PRESPLIT SAMPLES

Based on split_data_reproducible.py but filters out:
- stitched images
- presplit images

Only keeps: nosplit_nostitch images

Output files:
- both_train_base_no_stitch.json
- both_val_base_no_stitch.json
- both_test_base_no_stitch.json
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

LABEL_DAY = 'Dy30'

REQUIRED_METABOLITES = ['GlucoseGlo', 'GlutamateGlo', 'LactateGlo', 'PyruvateGlo']
MALATE_EXCLUSION_THRESHOLD_DAY = 10

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
    """Extract organoid ID without day from key."""
    match = re.match(r'^(.*)\s+Dy\d+\s+(.*)$', key)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return key

def extract_day_number(day_id):
    """Extract numeric day from dayID string."""
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

def is_stitch_or_split(common_key, img_path=None):
    """Check if sample is stitched or presplit - EXCLUDE THESE."""
    common_key_lower = str(common_key).lower()
    
    # Check common_key
    # Exclude stitched (but nosplit_stitched is still stitched!)
    if 'stitched' in common_key_lower:
        return True
    
    # Exclude presplit
    if 'presplit' in common_key_lower:
        return True
    
    # Also check img_path in case common_key doesn't have it
    if img_path:
        img_path_lower = str(img_path).lower()
        if 'stitched' in img_path_lower:
            return True
        if 'presplit' in img_path_lower:
            return True
    
    return False

# ============================================================
# DATA COLLECTION FUNCTIONS
# ============================================================

def collect_organoid_data(all_data, batches=['BA1', 'BA2'], require_metabolites=True):
    """
    Collect all timepoints for organoids, grouped by organoid ID.
    EXCLUDING stitched and presplit samples.
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
    excluded_stitched = 0
    
    for key, value in all_data.items():
        # Extract organoid ID and check if it has a label
        organoid_id = extract_organoid_id(key)
        if organoid_id not in organoid_labels:
            continue
        
        # Check batch
        batch = get_batch_prefix(value.get('BA'))
        if batch not in batches:
            continue
        
        # *** CRITICAL: EXCLUDE STITCHED/PRESPLIT ***
        common_key = value.get('common_key', '')
        img_path = value.get('processed', {}).get('img_path', '')
        if is_stitch_or_split(common_key, img_path):
            excluded_stitched += 1
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
        # Merge Dy20 and Dy21 into Dy20_5
        if day in ['Dy20', 'Dy21']:
            day = 'Dy20_5'
        timepoint_data = {
            'img_path': value['processed']['img_path'],
            'mask_path': value['processed']['mask_path'],
            'day': day
        }
        
        # Add metabolites if present
        if has_metabolites:
            metabolites_dict = {}
            
            for met in REQUIRED_METABOLITES:
                met_data = value['metabolites'][met]
                metabolites_dict[f'{met}_concentration_uM'] = met_data.get('concentration_uM')
                metabolites_dict[f'{met}_initial_concentration'] = met_data.get('initial_concentration')
            
            # Conditionally include MalateGlo for days >10
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
    
    print(f"  Excluded {excluded_stitched} stitched/presplit samples")
    print(f"  Kept {len(organoid_data)} organoids with timepoint data")
    
    return organoid_data

# ============================================================
# SPLIT FUNCTIONS
# ============================================================

def split_by_organoid(organoid_data, random_seed=RANDOM_SEED, test_size=TEST_SIZE, val_size=VAL_SIZE):
    """
    Split organoids into train/val/test sets with stratification by label.
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
    
    with train_file.open('w') as f:
        json.dump(train_data, f, indent=2)
    print(f"✓ Saved: {train_file}")
    
    with val_file.open('w') as f:
        json.dump(val_data, f, indent=2)
    print(f"✓ Saved: {val_file}")
    
    with test_file.open('w') as f:
        json.dump(test_data, f, indent=2)
    print(f"✓ Saved: {test_file}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("SPLIT DATA (NO STITCH/PRESPLIT) - BASE MODE ONLY")
    print("=" * 100)
    
    # Load all_data
    print("\n1. Loading all_data.json...")
    with open(ALL_DATA_JSON) as f:
        all_data = json.load(f)
    print(f"   Loaded {len(all_data)} records")
    
    # Collect organoid data (BASE MODE: BA1+BA2, image + metabolite)
    print("\n2. Collecting organoid data (BASE MODE)...")
    print("   Criteria: BA1+BA2 batches, both image and complete metabolite data")
    print("   EXCLUDING: Stitched and presplit samples")
    organoid_data = collect_organoid_data(
        all_data, 
        batches=['BA1', 'BA2'], 
        require_metabolites=True
    )
    
    # Split data
    print("\n3. Splitting organoids into train/val/test...")
    train_data, val_data, test_data = split_by_organoid(organoid_data)
    
    print(f"   Train: {len(train_data)} organoids")
    print(f"   Val:   {len(val_data)} organoids")
    print(f"   Test:  {len(test_data)} organoids")
    
    # Save splits
    print("\n4. Saving splits to data_splits directory...")
    save_splits(train_data, val_data, test_data, 'both', 'base_no_stitch')
    
    print("\n" + "=" * 100)
    print("✅ COMPLETED!")
    print("=" * 100)
    print(f"""
Output files created:
  - both_train_base_no_stitch.json
  - both_val_base_no_stitch.json
  - both_test_base_no_stitch.json

These files contain ONLY nosplit_nostitch images (stitched/presplit excluded).

To use these in training:
  --train-split data_splits/both_train_base_no_stitch.json
  --val-split data_splits/both_val_base_no_stitch.json
  --test-split data_splits/both_test_base_no_stitch.json
""")

if __name__ == "__main__":
    main()

