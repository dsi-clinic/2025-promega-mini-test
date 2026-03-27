#!/usr/bin/env python3 -u
"""
Reproducible train/val/test split for the time series (LSTM) model.

Reads data/all_data.json and outputs splits where each entry is a complete
time series for one organoid (all 11 expected timepoints).

Key design decisions:
- Splits by BASE WELL, not by individual organoid or timepoint.
  Split daughters (split1 + split2) from the same well share presplit
  timepoints, so they must land in the same partition to avoid data leakage.
- Requires ALL 11 timepoints to be present (no partial series).
- Labels come from the pre-computed label.value field (new schema).
- Handles organoid genealogy: nosplit, presplit+split1, presplit+split2.

Expected timepoints (mdl_day values):
  Dy03->3.0, Dy06->6.0, Dy08->8.0, Dy10->10.0, Dy13->13.0, Dy15->15.0,
  Dy17->17.0, Dy20/Dy21->20.5, Dy24->24.0, Dy28->28.0, Dy30->30.0

Output (data_splits/):
  series_train.json  - {organoid_id: {label, base_well_id, series: [{key, mdl_day, ...}]}}
  series_val.json
  series_test.json
  series_summary.json - counts and statistics
"""

import json
import re
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ============================================================
# CONFIGURATION
# ============================================================
ALL_DATA_JSON = 'data/all_data.json'
OUTPUT_DIR = Path('data_splits')
RANDOM_SEED = 42
TEST_SIZE = 0.2   # 20% test (held out)
VAL_SIZE = 0.1    # 10% val (within 80% training pool)

# All expected timepoints as mdl_day floats
# Dy20 and Dy21 both map to 20.5 (same physical timepoint, different naming)
EXPECTED_DAYS = [3.0, 6.0, 8.0, 10.0, 13.0, 15.0, 17.0, 20.5, 24.0, 28.0, 30.0]

LABEL_DAY = 30.0  # Survey labels come from Dy30

# ============================================================
# HELPERS
# ============================================================

def extract_mdl_day(value: dict) -> float | None:
    """
    NEW SCHEMA: day number is pre-computed as a float at value["day"]["number"].
    Falls back to parsing value["day"]["id"] string for safety.
    Dy20/Dy21 -> 20.5 is handled upstream in the new schema.
    """
    day = value.get('day', {})
    number = day.get('number')
    if number is not None:
        return float(number)
    # Fallback: parse day.id string (e.g. "Dy3", "Dy30")
    day_id = day.get('id', '')
    if not day_id:
        return None
    m = re.match(r'^Dy(\d+(?:\.\d+)?)$', day_id)
    if not m:
        return None
    n = float(m.group(1))
    if n in [20, 21]:
        return 20.5
    return n

def parse_split_type(main_id: str) -> str:
    """
    Determine split type from main_id string.
    e.g. 'BA2_96_1_Dy03_C1_presplit_nostitch' -> 'presplit'
         'BA4_96_1_Dy17_C12_split1_nostitch'  -> 'split1'
    """
    if not main_id:
        return 'nosplit'
    s = main_id.lower()
    if 'presplit' in s:
        return 'presplit'
    if 'split2' in s:
        return 'split2'
    if 'split1' in s:
        return 'split1'
    return 'nosplit'

def get_base_well(value: dict) -> str:
    """
    NEW SCHEMA: batch and well are under value["plate"].
    Canonical base well ID: 'BA1_96_1_A1'.
    """
    plate = value.get('plate', {})
    batch = plate.get('batch', '').replace(' ', '_')
    well = plate.get('well', '')
    return f"{batch}_{well}"

def is_blank(value: dict) -> bool:
    """True if this entry is a blank well. NEW SCHEMA: nested under metadata.verification."""
    return value.get('metadata', {}).get('verification', {}).get('blank', False) is True

def get_label(value: dict) -> str | None:
    """
    NEW SCHEMA: label is pre-computed as an object at value["label"].
    Returns the string value ('Acceptable' / 'Not Acceptable') or None.
    """
    label = value.get('label')
    if not label:
        return None
    return label.get('value')

def has_image(value: dict) -> bool:
    """
    True if the record has a processed image path.
    NEW SCHEMA: key is 'images' (was 'processed').
    """
    return (
        'images' in value
        and isinstance(value['images'], dict)
        and 'img_path' in value['images']
        and value['images']['img_path']
    )

# ============================================================
# STEP 1: BUILD GENEALOGY FROM all_data.json
# ============================================================

def build_genealogy(all_data: dict) -> dict:
    """
    Group all_data records by base_well, then by split_type.

    Returns:
        genealogy[base_well][split_type] = [
            {'key': ..., 'mdl_day': ..., 'value': ...}, ...
        ]
    """
    genealogy = defaultdict(lambda: defaultdict(list))
    skipped = {'no_image': 0, 'no_day': 0, 'no_main_id': 0, 'blank': 0}

    for key, value in all_data.items():
        # Skip blanks
        if is_blank(value):
            skipped['blank'] += 1
            continue

        # Need a processed image
        if not has_image(value):
            skipped['no_image'] += 1
            continue

        # Need a valid timepoint
        # NEW SCHEMA: day number is at value["day"]["number"]
        mdl_day = extract_mdl_day(value)
        if mdl_day is None:
            skipped['no_day'] += 1
            continue

        # Need a main_id to determine split type
        # NEW SCHEMA: main_id is at value["images"]["main_id"] or metadata.verification.main_id
        main_id = (
            value.get('images', {}).get('main_id')
            or value.get('metadata', {}).get('verification', {}).get('main_id', '')
        )
        if not main_id:
            skipped['no_main_id'] += 1
            continue

        # NEW SCHEMA: batch/well are under value["plate"]
        base_well = get_base_well(value)
        split_type = parse_split_type(main_id)

        genealogy[base_well][split_type].append({
            'key': key,
            'mdl_day': mdl_day,
            'value': value,
            'main_id': main_id,
        })

    print(f"  Skipped: {skipped}")
    print(f"  Unique base wells: {len(genealogy)}")
    return genealogy

# ============================================================
# STEP 2: BUILD COMPLETE SERIES (ALL 11 TIMEPOINTS)
# ============================================================

def build_complete_series(genealogy: dict) -> tuple[list, list]:
    """
    For each base_well, construct complete organoid series (must have all
    EXPECTED_DAYS).  Handles:
      - nosplit: single series directly
      - presplit + split1/split2: combined series per daughter

    Returns:
        complete   - list of series dicts (no blanks, all days present)
        incomplete - list of series dicts that failed completeness check
    """
    complete = []
    incomplete = []

    for base_well, splits in genealogy.items():
        nosplit = sorted(splits.get('nosplit', []), key=lambda x: x['mdl_day'])
        presplit = sorted(splits.get('presplit', []), key=lambda x: x['mdl_day'])
        split1 = sorted(splits.get('split1', []), key=lambda x: x['mdl_day'])
        split2 = sorted(splits.get('split2', []), key=lambda x: x['mdl_day'])

        # ---- Case A: no split ----
        if nosplit and not presplit and not split1 and not split2:
            _add_series(
                organoid_id=f"{base_well}_nosplit",
                base_well=base_well,
                genealogy_type='nosplit',
                items=nosplit,
                complete=complete,
                incomplete=incomplete,
            )

        # ---- Case B: presplit + daughters ----
        elif presplit and (split1 or split2):
            for daughter_name, daughter_items in [('split1', split1), ('split2', split2)]:
                if not daughter_items:
                    continue
                combined = presplit + daughter_items
                _add_series(
                    organoid_id=f"{base_well}_{daughter_name}",
                    base_well=base_well,
                    genealogy_type=f"presplit+{daughter_name}",
                    items=combined,
                    complete=complete,
                    incomplete=incomplete,
                )

        # ---- Case C: presplit only (daughters missing) -> always incomplete ----
        elif presplit and not split1 and not split2:
            days = sorted({i['mdl_day'] for i in presplit})
            incomplete.append({
                'organoid_id': f"{base_well}_presplit_only",
                'base_well': base_well,
                'genealogy_type': 'presplit_only',
                'days_present': days,
                'missing_days': sorted(set(EXPECTED_DAYS) - set(days)),
                'reason': 'presplit with no daughters',
            })

        # ---- Case D: daughters only (presplit missing) -> always incomplete ----
        else:
            for daughter_name, daughter_items in [('split1', split1), ('split2', split2)]:
                if not daughter_items:
                    continue
                days = sorted({i['mdl_day'] for i in daughter_items})
                incomplete.append({
                    'organoid_id': f"{base_well}_{daughter_name}_no_presplit",
                    'base_well': base_well,
                    'genealogy_type': f'{daughter_name}_no_presplit',
                    'days_present': days,
                    'missing_days': sorted(set(EXPECTED_DAYS) - set(days)),
                    'reason': 'daughter with no presplit data',
                })

    return complete, incomplete

def _add_series(organoid_id, base_well, genealogy_type, items, complete, incomplete):
    days_present = sorted({i['mdl_day'] for i in items})
    missing_days = sorted(set(EXPECTED_DAYS) - set(days_present))

    entry = {
        'organoid_id': organoid_id,
        'base_well': base_well,
        'genealogy_type': genealogy_type,
        'days_present': days_present,
        'missing_days': missing_days,
        'items': items,  # sorted list of timepoints
    }

    if missing_days:
        entry['reason'] = f"missing days: {missing_days}"
        incomplete.append(entry)
    else:
        complete.append(entry)

# ============================================================
# STEP 3: GET Dy30 LABELS
# ============================================================

def attach_labels(complete_series: list) -> tuple[list, int]:
    """
    For each complete series, find the Dy30 entry and read the pre-computed label.
    NEW SCHEMA: label is at value["label"]["value"], survey.evaluations is gone.
    Returns the labeled series and a count of how many were dropped for missing labels.
    """
    labeled = []
    dropped = 0

    for series in complete_series:
        dy30_items = [i for i in series['items'] if i['mdl_day'] == LABEL_DAY]
        if not dy30_items:
            dropped += 1
            continue

        # NEW SCHEMA: read pre-computed label directly
        dy30_value = dy30_items[0]['value']
        label = get_label(dy30_value)

        if label is None:
            dropped += 1
            continue

        series['label'] = label
        labeled.append(series)

    return labeled, dropped

# ============================================================
# STEP 4: SPLIT BY BASE WELL
# ============================================================

def split_by_base_well(
    labeled_series: list,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> tuple[list, list, list]:
    """
    Split at the base_well level (not organoid_id) to prevent leakage from
    shared presplit timepoints.

    Returns train_series, val_series, test_series (each a list of series dicts).
    """
    # Group organoids by base_well
    well_to_series: dict[str, list] = defaultdict(list)
    for s in labeled_series:
        well_to_series[s['base_well']].append(s)

    # One label per base_well: if daughters disagree, use most common
    wells = list(well_to_series.keys())
    well_labels = []
    for well in wells:
        all_labels = [s['label'] for s in well_to_series[well]]
        # Majority among daughters (usually identical for same well)
        label = max(set(all_labels), key=all_labels.count)
        well_labels.append(label)

    # First split: 80% train pool / 20% test
    train_wells, test_wells = train_test_split(
        wells,
        test_size=test_size,
        stratify=well_labels,
        random_state=random_seed,
    )

    # Second split: within train pool, 10% -> val
    train_well_labels = [well_labels[wells.index(w)] for w in train_wells]
    train_wells_final, val_wells = train_test_split(
        train_wells,
        test_size=val_size,
        stratify=train_well_labels,
        random_state=random_seed,
    )

    # Expand wells back to individual organoid series
    def expand(well_list):
        out = []
        for w in well_list:
            out.extend(well_to_series[w])
        return out

    return expand(train_wells_final), expand(val_wells), expand(test_wells)

# ============================================================
# STEP 5: SERIALIZE AND SAVE
# ============================================================

def series_to_output(series: dict) -> dict:
    """Convert internal series dict to clean output format."""
    timepoints = []
    # De-duplicate on mdl_day (presplit entries appear in both split1 and split2)
    seen_days = set()
    for item in sorted(series['items'], key=lambda x: x['mdl_day']):
        day = item['mdl_day']
        if day in seen_days:
            continue
        seen_days.add(day)
        value = item['value']
        # NEW SCHEMA: image paths are under value["images"]
        images = value.get('images', {})
        timepoints.append({
            'key': item['key'],
            'mdl_day': day,
            'dayID': value.get('day', {}).get('id'),
            'img_path': images.get('img_path'),
            'mask_path': images.get('mask_path'),
            'main_id': item['main_id'],
            'split_type': parse_split_type(item['main_id']),
        })

    return {
        'organoid_id': series['organoid_id'],
        'base_well': series['base_well'],
        'genealogy_type': series['genealogy_type'],
        'label': series['label'],
        'n_timepoints': len(timepoints),
        'timepoints': timepoints,
    }

def save_split(series_list: list, path: Path) -> None:
    output = {s['organoid_id']: series_to_output(s) for s in series_list}
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

def print_split_stats(series_list: list, name: str) -> None:
    labels = [s['label'] for s in series_list]
    acc = labels.count('Acceptable')
    na = labels.count('Not Acceptable')
    wells = {s['base_well'] for s in series_list}
    print(f"  {name}: {len(series_list)} organoids ({len(wells)} wells) "
          f"| {acc} Acceptable, {na} Not Acceptable")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reproducible train/val/test split for time series (LSTM) model'
    )
    parser.add_argument('--all-data', type=Path, default=ALL_DATA_JSON,
                        help=f'Path to all_data.json (default: {ALL_DATA_JSON})')
    parser.add_argument('--out-dir', type=Path, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    args = parser.parse_args()

    print(f"\nLoading {args.all_data}...")
    with open(args.all_data) as f:
        all_data = json.load(f)
    print(f"✓ Loaded {len(all_data)} records")

    print(f"\nExpected timepoints: {EXPECTED_DAYS}")
    print(f"Label day: Dy30 (mdl_day={LABEL_DAY})")
    print(f"Random seed: {args.seed}")
    print(f"Split: {int((1-TEST_SIZE)*(1-VAL_SIZE)*100)}% train / "
          f"{int((1-TEST_SIZE)*VAL_SIZE*100)}% val / {int(TEST_SIZE*100)}% test")

    # Step 1: genealogy
    print("\n[1/5] Building genealogy...")
    genealogy = build_genealogy(all_data)

    # Step 2: complete series
    print("\n[2/5] Filtering to complete series (all 11 timepoints)...")
    complete, incomplete = build_complete_series(genealogy)
    print(f"  Complete series: {len(complete)}")
    print(f"  Incomplete series (dropped): {len(incomplete)}")

    # Step 3: labels
    print("\n[3/5] Attaching Dy30 survey labels...")
    labeled, dropped = attach_labels(complete)
    print(f"  Labeled series: {len(labeled)}")
    print(f"  Dropped (no/ambiguous label): {dropped}")
    label_counts = defaultdict(int)
    for s in labeled:
        label_counts[s['label']] += 1
    for lbl, cnt in label_counts.items():
        print(f"    {lbl}: {cnt}")

    # Step 4: split
    print("\n[4/5] Splitting by base well (preserving genealogy integrity)...")
    train, val, test = split_by_base_well(labeled, random_seed=args.seed)
    print_split_stats(train, "Train")
    print_split_stats(val,   "Val  ")
    print_split_stats(test,  "Test ")

    # Step 5: save
    print("\n[5/5] Saving splits...")
    args.out_dir.mkdir(exist_ok=True)
    train_path = args.out_dir / 'series_train.json'
    val_path   = args.out_dir / 'series_val.json'
    test_path  = args.out_dir / 'series_test.json'

    save_split(train, train_path)
    save_split(val,   val_path)
    save_split(test,  test_path)

    # Summary
    summary = {
        'random_seed': args.seed,
        'expected_days': EXPECTED_DAYS,
        'label_day': LABEL_DAY,
        'total_records_loaded': len(all_data),
        'complete_series': len(complete),
        'incomplete_dropped': len(incomplete),
        'no_label_dropped': dropped,
        'total_labeled': len(labeled),
        'train': {
            'organoids': len(train),
            'wells': len({s['base_well'] for s in train}),
            'label_counts': {l: [s['label'] for s in train].count(l) for l in set(s['label'] for s in train)},
        },
        'val': {
            'organoids': len(val),
            'wells': len({s['base_well'] for s in val}),
            'label_counts': {l: [s['label'] for s in val].count(l) for l in set(s['label'] for s in val)},
        },
        'test': {
            'organoids': len(test),
            'wells': len({s['base_well'] for s in test}),
            'label_counts': {l: [s['label'] for s in test].count(l) for l in set(s['label'] for s in test)},
        },
    }
    summary_path = args.out_dir / 'series_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ {train_path}")
    print(f"✓ {val_path}")
    print(f"✓ {test_path}")
    print(f"✓ {summary_path}")
    print("\nOutput format per organoid:")
    print("  {organoid_id: {label, base_well, genealogy_type, n_timepoints,")
    print("                 timepoints: [{key, mdl_day, dayID, img_path, mask_path, ...}]}}")
    print("\nReady for LSTM/time-series model training!")

if __name__ == '__main__':
    main()
