"""
Load split data from JSON files (from minitest/data_splits) and convert to format
expected by training scripts.

The split JSON files have structure:
{
  "organoid_id": {
    "label": "Acceptable",
    "batch": "BA1",
    "timepoints": {
      "Dy03": {
        "img_path": "...",
        "mask_path": "...",
        "day": "Dy03",
        "metabolites": {...}
      }
    }
  }
}

Training scripts expect:
- train_ids, val_ids, test_ids: lists of organoid IDs
- series_metadata: {organoid_id: {entry_keys: [...], days: [...], label: ...}}
- data: {entry_key: {lstm_processed: {image_path, mask_path, clipped_image_path}}}
"""

import json
import re
from pathlib import Path


def _derive_overlay_path(mask_path):
    """Derive overlay path from mask path: .../image_mask_overlays/<stem>_overlay.png"""
    if not mask_path:
        return ""
    mp = Path(mask_path)
    if not mp.exists():
        return ""
    mask_dir = mp.parent
    day_dir = mask_dir.parent
    overlays_dir = day_dir / "image_mask_overlays"
    stem = mp.stem
    out_stem = re.sub(r"_predmask$", "", stem, flags=re.IGNORECASE) + "_overlay"
    out = overlays_dir / f"{out_stem}.png"
    return str(out) if out.exists() else ""


def day_str_to_float(day_str):
    """Convert day string (Dy03, Dy20_5) to float (3.0, 20.5)"""
    if day_str.startswith('Dy'):
        day_str = day_str[2:]
    if '_' in day_str:
        parts = day_str.split('_')
        return float(parts[0]) + float(parts[1]) / 10
    return float(day_str)


def construct_entry_key(organoid_id, day):
    """
    Construct entry_key from organoid_id and day.
    Based on filename pattern: BA2_96_1_Dy03_H7_nosplit_nostitch
    """
    # Extract components from organoid_id like "BA2 96_1 H7"
    parts = organoid_id.split()
    if len(parts) >= 3:
        batch = parts[0]  # BA2
        plate_well = parts[1]  # 96_1
        well_pos = parts[2]  # H7
        # Format: BA2_96_1_Dy03_H7_nosplit_nostitch
        entry_key = f"{batch}_{plate_well}_{day}_{well_pos}_nosplit_nostitch"
    else:
        # Fallback: simpler format
        entry_key = f"{organoid_id}_{day}_nosplit_nostitch"
    return entry_key


def load_split_data(train_split_path, val_split_path, test_split_path):
    """
    Load split JSON files and convert to format expected by training scripts.
    
    Args:
        train_split_path: Path to train split JSON file
        val_split_path: Path to validation split JSON file
        test_split_path: Path to test split JSON file
    
    Returns:
        train_ids, val_ids, test_ids, series_metadata, data
    """
    print("="*80)
    print("LOADING SPLIT DATA FROM JSON FILES")
    print("="*80)
    
    # Load split files
    print(f"\nLoading train split: {train_split_path}")
    with open(train_split_path) as f:
        train_split = json.load(f)
    print(f"  Loaded {len(train_split)} organoids")
    
    print(f"\nLoading val split: {val_split_path}")
    with open(val_split_path) as f:
        val_split = json.load(f)
    print(f"  Loaded {len(val_split)} organoids")
    
    print(f"\nLoading test split: {test_split_path}")
    with open(test_split_path) as f:
        test_split = json.load(f)
    print(f"  Loaded {len(test_split)} organoids")
    
    # Extract organoid IDs
    train_ids = list(train_split.keys())
    val_ids = list(val_split.keys())
    test_ids = list(test_split.keys())
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ids)} organoids")
    print(f"  Val:   {len(val_ids)} organoids")
    print(f"  Test:  {len(test_ids)} organoids")
    
    # Build series_metadata and data dicts
    series_metadata = {}
    data = {}
    
    # Combine all splits to build complete metadata
    all_splits = {**train_split, **val_split, **test_split}
    
    print(f"\nBuilding series_metadata and data dicts from {len(all_splits)} organoids...")
    
    for organoid_id, org_data in all_splits.items():
        timepoints = org_data.get('timepoints', {})
        
        # Sort timepoints by day
        sorted_days = sorted(timepoints.keys(), key=lambda d: day_str_to_float(d))
        
        entry_keys = []
        days = []
        
        for day_str in sorted_days:
            tp_data = timepoints[day_str]
            
            # Construct entry_key
            entry_key = construct_entry_key(organoid_id, day_str)
            entry_keys.append(entry_key)
            
            # Convert day string to float
            day_float = day_str_to_float(day_str)
            days.append(day_float)
            
            # Build data entry
            img_path = tp_data.get('img_path', '')
            mask_path = tp_data.get('mask_path', '')
            overlay_path = tp_data.get('overlay_path') or _derive_overlay_path(mask_path)
            
            # Include both 'processed' (for EfficientNet) and 'lstm_processed' (for CNN-LSTM)
            data[entry_key] = {
                'processed': {
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'overlay_path': overlay_path or ''
                },
                'lstm_processed': {
                    'image_path': img_path,
                    'clipped_image_path': img_path,
                    'mask_path': mask_path,
                    'overlay_path': overlay_path or ''
                }
            }
        
        # Build series_metadata entry
        label = org_data.get('label', '')
        series_metadata[organoid_id] = {
            'organoid_id': organoid_id,
            'label': label,
            'entry_keys': entry_keys,
            'days': days,
            'n_timepoints': len(entry_keys),
            'batch': org_data.get('batch', '')
        }
    
    print(f"  Built {len(series_metadata)} series_metadata entries")
    print(f"  Built {len(data)} data entries")
    
    # Verify no overlap between splits
    assert set(train_ids).isdisjoint(set(val_ids)), "Train and val overlap!"
    assert set(train_ids).isdisjoint(set(test_ids)), "Train and test overlap!"
    assert set(val_ids).isdisjoint(set(test_ids)), "Val and test overlap!"
    print("\n✓ No overlap between splits verified")
    
    # Print statistics
    def get_label_stats(ids):
        labels = [series_metadata[oid].get('label', '').lower() for oid in ids if oid in series_metadata]
        acceptable = sum(1 for l in labels if l in ('acceptable', 'good', 'accepted'))
        not_acceptable = sum(1 for l in labels if l in ('not acceptable', 'bad', 'rejected', 'not_good'))
        return acceptable, not_acceptable, len(ids)
    
    tr_acc, tr_not, tr_tot = get_label_stats(train_ids)
    val_acc, val_not, val_tot = get_label_stats(val_ids)
    te_acc, te_not, te_tot = get_label_stats(test_ids)
    
    print(f"\nLabel distribution:")
    print(f"  Train: {tr_acc} Acceptable, {tr_not} Not Acceptable ({tr_tot} total)")
    print(f"  Val:   {val_acc} Acceptable, {val_not} Not Acceptable ({val_tot} total)")
    print(f"  Test:  {te_acc} Acceptable, {te_not} Not Acceptable ({te_tot} total)")
    
    return train_ids, val_ids, test_ids, series_metadata, data


if __name__ == "__main__":
    # Test loading — REPLACE your_name and image_classifier_ts with your workspace path
    base_dir = Path("/home/your_name/image_classifier_ts/data_splits")
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        base_dir / "both_train_amanda_style.json",
        base_dir / "both_val_amanda_style.json",
        base_dir / "both_test_amanda_style.json"
    )
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"\nSample organoid ID: {train_ids[0]}")
    sample_meta = series_metadata[train_ids[0]]
    print(f"  Label: {sample_meta['label']}")
    print(f"  Entry keys: {sample_meta['entry_keys'][:3]}... ({len(sample_meta['entry_keys'])} total)")
    print(f"  Days: {sample_meta['days'][:3]}... ({len(sample_meta['days'])} total)")
    
    sample_entry_key = sample_meta['entry_keys'][0]
    sample_entry = data[sample_entry_key]
    print(f"\nSample entry key: {sample_entry_key}")
    print(f"  image_path: {sample_entry['lstm_processed']['image_path']}")
    print(f"  mask_path: {sample_entry['lstm_processed']['mask_path']}")
