#!/usr/bin/env python3 -u
"""
Reproducible train/val/test split - BASE MODE
EXCLUDE ONLY SPLIT ORGANOIDS (Keep stitched samples)

This variant excludes ONLY split/presplit organoids to isolate the effect of splitting.
Stitched samples are KEPT in this dataset.

Output files:
- both_train_exclude_split_only.json
- both_val_exclude_split_only.json
- both_test_exclude_split_only.json
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
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

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
    """
    Extract organoid ID without day from key.
    'BA1 96_1 Dy30 A1' -> 'BA1 96_1 A1'
    """
    import re
    match = re.match(r'^(.*)\s+Dy\d+\s+(.*)$', key)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return key

def get_batch_prefix(ba_string):
    """Extract batch prefix (BA1, BA2, etc.) from full batch string."""
    if not ba_string:
        return None
    return ba_string.split()[0] if ' ' in ba_string else ba_string

def is_split_only(common_key, img_path=None):
    """
    Check if sample is SPLIT/PRESPLIT.
    EXCLUDE: split-only images AND split+stitched images
    KEEP: stitched-only images and nosplit_nostitch images
    """
    common_key_lower = str(common_key).lower()
    
    # Exclude presplit samples
    if 'presplit' in common_key_lower:
        return True
    
    # Check if has split (but not nosplit)
    has_split = ('split' in common_key_lower and 'nosplit' not in common_key_lower)
    
    # Check if has stitched (but not nostitch/no_stitch/no-stitch)
    has_stitched = False
    if 'stitched' in common_key_lower:
        if 'nostitch' not in common_key_lower and 'no_stitch' not in common_key_lower and 'no-stitch' not in common_key_lower:
            has_stitched = True
    
    # Check img_path
    if img_path:
        img_path_lower = str(img_path).lower()
        if 'presplit' in img_path_lower:
            return True
        if 'split' in img_path_lower and 'nosplit' not in img_path_lower:
            has_split = True
        if 'stitched' in img_path_lower:
            if 'nostitch' not in img_path_lower and 'no_stitch' not in img_path_lower and 'no-stitch' not in img_path_lower:
                has_stitched = True
    
    # Exclude if has split (whether or not it also has stitched)
    # This excludes: split-only AND split+stitched
    # This keeps: stitched-only (no split) and nosplit_nostitch
    return has_split

def has_valid_image_data(record):
    """Check if record has valid processed image data."""
    return ('processed' in record and 
            record['processed'] and 
            'img_path' in record['processed'] and
            'mask_path' in record['processed'])

def has_complete_metabolites(metabolites_dict, day_int):
    """Check if all required metabolites are present."""
    if not metabolites_dict:
        return False
    
    for met in REQUIRED_METABOLITES:
        if met not in metabolites_dict:
            return False
        if 'concentration_uM' not in metabolites_dict[met]:
            return False
        if metabolites_dict[met]['concentration_uM'] is None:
            return False
    
    # Check MalateGlo requirement
    if day_int is not None and day_int > MALATE_EXCLUSION_THRESHOLD_DAY:
        if 'MalateGlo' not in metabolites_dict:
            return False
    
    return True

def extract_day_int(day_str):
    """Extract integer from day string."""
    match = re.search(r'(\d+)', str(day_str))
    if match:
        return int(match.group(1))
    return None

def collect_organoid_data(all_data, batches=['BA1', 'BA2'], require_metabolites=True):
    """
    Collect all timepoints for organoids, grouped by organoid ID.
    EXCLUDING ONLY split/presplit samples (keeping stitched samples).
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
        organoid_labels[organoid_id] = {'label': label, 'batch': batch}
    
    # Second pass: collect ALL timepoints for labeled organoids (before filtering)
    organoid_data_raw = {}
    
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
        day = value.get('dayID')
        day_int = extract_day_int(day)
        has_metabolites = has_complete_metabolites(value.get('metabolites', {}), day_int)
        if require_metabolites and not has_metabolites:
            continue
        
        # Initialize organoid entry if needed
        if organoid_id not in organoid_data_raw:
            organoid_data_raw[organoid_id] = {
                'label': organoid_labels[organoid_id]['label'],
                'batch': organoid_labels[organoid_id]['batch'],
                'timepoints': {}
            }
        
        # Merge Dy20 and Dy21 into Dy20_5
        if day in ['Dy20', 'Dy21']:
            day = 'Dy20_5'
        
        common_key = value.get('common_key', '')
        img_path = value.get('processed', {}).get('img_path', '')
        
        # Add timepoint (collect ALL timepoints first, flag for later filtering)
        timepoint_data = {
            'img_path': value['processed']['img_path'],
            'mask_path': value['processed']['mask_path'],
            'day': day,
            'is_split': is_split_only(common_key, img_path)  # Flag for later filtering
        }
        
        # Add metabolites if present
        if has_metabolites:
            metabolites_dict = {}
            for met in REQUIRED_METABOLITES:
                met_data = value['metabolites'][met]
                metabolites_dict[f'{met}_concentration_uM'] = met_data.get('concentration_uM')
                metabolites_dict[f'{met}_initial_concentration'] = met_data.get('initial_concentration')
            
            # Conditionally include MalateGlo for days >10
            if day_int is not None and day_int > MALATE_EXCLUSION_THRESHOLD_DAY:
                if 'MalateGlo' in value.get('metabolites', {}):
                    malate_data = value['metabolites']['MalateGlo']
                    if 'concentration_uM' in malate_data and malate_data['concentration_uM'] is not None:
                        metabolites_dict['MalateGlo_concentration_uM'] = malate_data['concentration_uM']
                    if 'initial_concentration' in malate_data and malate_data['initial_concentration'] is not None:
                        metabolites_dict['MalateGlo_initial_concentration'] = malate_data['initial_concentration']
            
            timepoint_data['metabolites'] = metabolites_dict
        
        organoid_data_raw[organoid_id]['timepoints'][day] = timepoint_data
    
    # Third pass: EXCLUDE ENTIRE ORGANOIDS if ANY day has split/presplit
    # This ensures pure organoids only - if one day is bad, exclude all days
    organoid_data = {}
    excluded_organoids = 0
    excluded_split = 0
    
    for organoid_id, org_info in organoid_data_raw.items():
        # Check if ANY timepoint has split/presplit
        has_bad_timepoint = False
        for day, day_data in org_info['timepoints'].items():
            if day_data.get('is_split', False):
                has_bad_timepoint = True
                excluded_split += 1
                break  # Found one bad day, exclude entire organoid
        
        if has_bad_timepoint:
            excluded_organoids += 1
            continue  # Skip this entire organoid
        
        # Organoid is clean - keep all its timepoints
        organoid_data[organoid_id] = {
            'label': org_info['label'],
            'batch': org_info['batch'],
            'timepoints': {}
        }
        
        # Copy all timepoints (removing the flag)
        for day, day_data in org_info['timepoints'].items():
            clean_timepoint = {k: v for k, v in day_data.items() if k != 'is_split'}
            organoid_data[organoid_id]['timepoints'][day] = clean_timepoint
    
    print(f"  Excluded {excluded_organoids} entire organoids (had split/presplit timepoints)")
    print(f"  Excluded {excluded_split} split/presplit timepoints from excluded organoids")
    print(f"  Kept {len(organoid_data)} pure organoids (no split/presplit, but may have stitched)")
    
    return organoid_data

# ============================================================
# MAIN SPLIT FUNCTION
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-data', default=ALL_DATA_JSON, help='Path to all_data.json')
    parser.add_argument('--output-dir', default='data_splits', help='Output directory')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    args = parser.parse_args()
    
    print("="*80)
    print("EXCLUDE SPLIT ONLY - Keep Stitched Samples")
    print("="*80)
    print(f"Random seed: {args.seed}")
    print(f"Test size: {TEST_SIZE} ({TEST_SIZE*100:.0f}%)")
    print(f"Val size: {VAL_SIZE} ({VAL_SIZE*100:.0f}%)")
    print()
    
    # Load data
    print("1. Loading all_data.json...")
    with open(args.all_data, 'r') as f:
        all_data = json.load(f)
    print(f"  Loaded {len(all_data)} entries")
    
    # Collect organoid data
    print("\n2. Collecting organoid data (BASE MODE)...")
    print("   Criteria: BA1+BA2 batches, both image and complete metabolite data")
    print("   EXCLUDING: Split/presplit samples ONLY (keeping stitched samples)")
    organoid_data = collect_organoid_data(all_data, batches=['BA1', 'BA2'])
    
    if not organoid_data:
        print("ERROR: No organoids found!")
        return
    
    # Count labels
    label_counts = {}
    for org_id, org_info in organoid_data.items():
        label = org_info['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n3. Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Split by organoid ID
    print(f"\n4. Splitting by organoid (seed={args.seed})...")
    organoid_ids = list(organoid_data.keys())
    labels_for_split = [organoid_data[oid]['label'] for oid in organoid_ids]
    
    # Train+Val / Test split
    train_val_ids, test_ids = train_test_split(
        organoid_ids,
        test_size=TEST_SIZE,
        random_state=args.seed,
        stratify=labels_for_split
    )
    
    # Train / Val split
    train_val_labels = [organoid_data[oid]['label'] for oid in train_val_ids]
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=args.seed,
        stratify=train_val_labels
    )
    
    print(f"  Train: {len(train_ids)} organoids")
    print(f"  Val: {len(val_ids)} organoids")
    print(f"  Test: {len(test_ids)} organoids")
    
    # Create split dictionaries
    train_split = {oid: organoid_data[oid] for oid in train_ids}
    val_split = {oid: organoid_data[oid] for oid in val_ids}
    test_split = {oid: organoid_data[oid] for oid in test_ids}
    
    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / 'both_train_exclude_split_only.json'
    val_file = output_dir / 'both_val_exclude_split_only.json'
    test_file = output_dir / 'both_test_exclude_split_only.json'
    
    print(f"\n5. Saving splits to {output_dir}/...")
    with open(train_file, 'w') as f:
        json.dump(train_split, f, indent=2)
    print(f"  ✓ {train_file}")
    
    with open(val_file, 'w') as f:
        json.dump(val_split, f, indent=2)
    print(f"  ✓ {val_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_split, f, indent=2)
    print(f"  ✓ {test_file}")
    
    print("\n" + "="*80)
    print("✓ SPLIT COMPLETE")
    print("="*80)
    print("\nThese splits EXCLUDE split/presplit organoids but KEEP stitched samples.")
    print("This isolates the effect of splitting on model performance.")

if __name__ == '__main__':
    main()


