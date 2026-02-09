# analysis/images/series/filter_complete_series.py
from __future__ import annotations
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from file_utils.merge.merge_all_data import extract_mdl_day

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

# Define expected timepoints using mdl_day values
EXPECTED_DAYS = [3.0, 6.0, 8.0, 10.0, 13.0, 15.0, 17.0, 20.5, 24.0, 28.0, 30.0]

def create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-mapping-json', type=Path, required=True, help='Path to image mapping JSON file (resized)')
    parser.add_argument('--out-dir', type=Path, required=True, help='Path to output directory')
    parser.add_argument('--show-examples', action='store_true',help='Show detailed examples of incomplete series')
    args = parser.parse_args()
    return args

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def is_blank(entry):
    """Check if an entry is a blank well"""
    if 'verification' in entry:
        blank = entry['verification'].get('blank', False)
        if blank is True or blank == 'true':
            return True
    return False

def parse_split_from_main_id(main_id):
    """
    Extract split status from main_id
    Example: "BA4_96_1_Dy17_C12_split2_nostitch" -> "split2"
    Example: "BA4_96_1_Dy17_C2_nosplit_nostitch" -> "nosplit"
    Example: "BA4_96_1_Dy10_H9_presplit_nostitch" -> "presplit"
    """
    if not main_id:
        return 'nosplit'

    main_id_lower = main_id.lower()

    if 'presplit' in main_id_lower:
        return 'presplit'
    elif 'split2' in main_id_lower:
        return 'split2'
    elif 'split1' in main_id_lower:
        return 'split1'
    elif 'nosplit' in main_id_lower:
        return 'nosplit'
    else:
        # Fallback - shouldn't happen
        return 'nosplit'

def parse_organoid_info(entry):
    """Extract BA, well, and split status from entry using main_id"""
    ba = entry['BA'].replace(' ', '_')
    well = entry['wellID']

    # Get split type from main_id (ground truth!)
    main_id = entry.get('verification', {}).get('main_id', '')
    split_type = parse_split_from_main_id(main_id)

    return ba, well, split_type

def get_base_well_id(ba, well):
    """Get the base identifier for a well (without split info)"""
    return f"{ba}_{well}"

def organize_by_genealogy(entries):
    """
    Organize data by well genealogy, tracking presplit and daughter organoids
    Returns: dict mapping base_well_id -> {presplit: [...], split1: [...], split2: [...], nosplit: [...]}
    """
    genealogy = defaultdict(lambda: {'presplit': [], 'split1': [], 'split2': [], 'nosplit': []})

    skipped_no_mdl_day = 0
    skipped_no_main_id = 0

    for key, entry in tqdm(entries.items(), desc="Organizing by genealogy"):
        # Skip if no mdl_day
        if 'mdl_day' not in entry or entry['mdl_day'] is None:
            skipped_no_mdl_day += 1
            continue

        # Skip if no main_id
        if 'main_id' not in entry.get('verification', {}) or not entry['verification']['main_id']:
            skipped_no_main_id += 1
            continue

        ba, well, split_type = parse_organoid_info(entry)
        base_well_id = get_base_well_id(ba, well)
        mdl_day = entry['mdl_day']

        genealogy[base_well_id][split_type].append({
            'key': key,
            'entry': entry,
            'mdl_day': mdl_day,
            'split_type': split_type,
            'main_id': entry['verification']['main_id'],
            'is_blank': is_blank(entry)
        })

    if skipped_no_mdl_day > 0:
        logging.info("Skipped %d entries with no mdl_day", skipped_no_mdl_day)
    if skipped_no_main_id > 0:
        logging.info("Skipped %d entries with no main_id", skipped_no_main_id)

    # Sort each list by mdl_day
    for base_well_id in genealogy:
        for split_type in genealogy[base_well_id]:
            genealogy[base_well_id][split_type].sort(key=lambda x: x['mdl_day'])

    return genealogy

def build_complete_series(genealogy):
    """
    Build complete time series for each organoid, handling splits
    Returns list of complete series with their genealogy info
    """
    complete_series = []
    complete_with_blanks = []
    incomplete_series = []

    for base_well_id, splits in tqdm(genealogy.items(), desc="Building series"):
        presplit = splits['presplit']
        split1 = splits['split1']
        split2 = splits['split2']
        nosplit = splits['nosplit']

        # Case 1: No split - simple case
        if nosplit and not presplit and not split1 and not split2:
            days_present = set(item['mdl_day'] for item in nosplit)
            missing_days = set(EXPECTED_DAYS) - days_present
            is_complete = len(missing_days) == 0

            # Check for blanks
            has_blanks = any(item['is_blank'] for item in nosplit)
            blank_days = [item['mdl_day'] for item in nosplit if item['is_blank']]

            series_info = {
                'organoid_id': f"{base_well_id}_nosplit",
                'base_well_id': base_well_id,
                'split_genealogy': 'nosplit',
                'days_present': sorted(list(days_present)),
                'missing_days': sorted(list(missing_days)),
                'n_days': len(days_present),
                'is_complete': is_complete,
                'has_blanks': has_blanks,
                'blank_days': blank_days,
                'series': nosplit
            }

            if is_complete and not has_blanks:
                complete_series.append(series_info)
            elif is_complete and has_blanks:
                complete_with_blanks.append(series_info)
            else:
                incomplete_series.append(series_info)

        # Case 2: Split occurred - need to combine presplit + daughter
        elif presplit and (split1 or split2):
            # For each daughter
            for daughter_name, daughter_data in [('split1', split1), ('split2', split2)]:
                if not daughter_data:
                    continue

                # Combine presplit + daughter
                combined_series = presplit + daughter_data
                days_present = set(item['mdl_day'] for item in combined_series)
                missing_days = set(EXPECTED_DAYS) - days_present
                is_complete = len(missing_days) == 0

                # Check for blanks
                has_blanks = any(item['is_blank'] for item in combined_series)
                blank_days = [item['mdl_day'] for item in combined_series if item['is_blank']]

                series_info = {
                    'organoid_id': f"{base_well_id}_{daughter_name}",
                    'base_well_id': base_well_id,
                    'split_genealogy': f"presplit+{daughter_name}",
                    'days_present': sorted(list(days_present)),
                    'missing_days': sorted(list(missing_days)),
                    'n_days': len(days_present),
                    'is_complete': is_complete,
                    'has_blanks': has_blanks,
                    'blank_days': blank_days,
                    'series': combined_series,
                    'n_presplit': len(presplit),
                    'n_daughter': len(daughter_data)
                }

                if is_complete and not has_blanks:
                    complete_series.append(series_info)
                elif is_complete and has_blanks:
                    complete_with_blanks.append(series_info)
                else:
                    incomplete_series.append(series_info)

        # Case 3: Only presplit (no daughters found) - incomplete by definition
        elif presplit and not split1 and not split2:
            days_present = set(item['mdl_day'] for item in presplit)
            has_blanks = any(item['is_blank'] for item in presplit)
            incomplete_series.append({
                'organoid_id': f"{base_well_id}_presplit_only",
                'base_well_id': base_well_id,
                'split_genealogy': 'presplit_only',
                'days_present': sorted(list(days_present)),
                'missing_days': sorted(list(set(EXPECTED_DAYS) - days_present)),
                'n_days': len(days_present),
                'is_complete': False,
                'has_blanks': has_blanks,
                'series': presplit
            })

        # Case 4: Only daughters (no presplit) - need presplit for completeness
        elif (split1 or split2) and not presplit:
            for daughter_name, daughter_data in [('split1', split1), ('split2', split2)]:
                if not daughter_data:
                    continue
                days_present = set(item['mdl_day'] for item in daughter_data)
                has_blanks = any(item['is_blank'] for item in daughter_data)
                incomplete_series.append({
                    'organoid_id': f"{base_well_id}_{daughter_name}_no_presplit",
                    'base_well_id': base_well_id,
                    'split_genealogy': f"{daughter_name}_only",
                    'days_present': sorted(list(days_present)),
                    'missing_days': sorted(list(set(EXPECTED_DAYS) - days_present)),
                    'n_days': len(days_present),
                    'is_complete': False,
                    'has_blanks': has_blanks,
                    'series': daughter_data
                })

    return complete_series, complete_with_blanks, incomplete_series

def sanity_check_splits(complete_series):
    """
    Verify that split daughters share the same presplit parent data
    """
    logging.info("%s", '='*70)
    logging.info("SANITY CHECK: Split Genealogy")
    logging.info("%s", '='*70)

    # Find wells that have both split1 and split2
    split_wells = defaultdict(lambda: {'split1': None, 'split2': None})

    for series in complete_series:
        if 'presplit+split1' in series['split_genealogy']:
            split_wells[series['base_well_id']]['split1'] = series
        elif 'presplit+split2' in series['split_genealogy']:
            split_wells[series['base_well_id']]['split2'] = series

    # Filter to only wells with BOTH daughters
    both_daughters = {k: v for k, v in split_wells.items()
                      if v['split1'] is not None and v['split2'] is not None}

    logging.info("Found %d wells with both split daughters", len(both_daughters))

    if not both_daughters:
        logging.info("No wells with both daughters to check")
        return True

    # Check first few examples
    all_good = True
    examples_to_check = min(5, len(both_daughters))

    logging.info("Checking %d examples:", examples_to_check)
    for i, (base_well_id, daughters) in enumerate(list(both_daughters.items())[:examples_to_check]):
        split1_series = daughters['split1']
        split2_series = daughters['split2']

        logging.info("%d. %s:", i+1, base_well_id)

        # Get presplit entries for each daughter
        split1_presplit = [item for item in split1_series['series'] if item['split_type'] == 'presplit']
        split2_presplit = [item for item in split2_series['series'] if item['split_type'] == 'presplit']

        # Get daughter-specific entries
        split1_daughter = [item for item in split1_series['series'] if item['split_type'] == 'split1']
        split2_daughter = [item for item in split2_series['series'] if item['split_type'] == 'split2']

        logging.info("   Split1: %d presplit + %d daughter days", split1_series['n_presplit'], len(split1_daughter))
        logging.info("   Split2: %d presplit + %d daughter days", split2_series['n_presplit'], len(split2_daughter))

        # Check 1: Same number of presplit entries
        if len(split1_presplit) != len(split2_presplit):
            logging.info("   [ERROR] Different presplit counts!")
            all_good = False
            continue

        # Check 2: Same presplit days
        split1_presplit_days = sorted([item['mdl_day'] for item in split1_presplit])
        split2_presplit_days = sorted([item['mdl_day'] for item in split2_presplit])

        if split1_presplit_days != split2_presplit_days:
            logging.info("   [ERROR] Different presplit days!")
            logging.info("      Split1 presplit: %s", split1_presplit_days)
            logging.info("      Split2 presplit: %s", split2_presplit_days)
            all_good = False
            continue

        # Check 3: Actually the same entry keys (same images)
        split1_presplit_keys = sorted([item['key'] for item in split1_presplit])
        split2_presplit_keys = sorted([item['key'] for item in split2_presplit])

        if split1_presplit_keys != split2_presplit_keys:
            logging.info("   [ERROR] Different presplit entry keys!")
            all_good = False
            continue

        # Check 4: Daughters have different data
        split1_daughter_keys = sorted([item['key'] for item in split1_daughter])
        split2_daughter_keys = sorted([item['key'] for item in split2_daughter])

        overlap = set(split1_daughter_keys) & set(split2_daughter_keys)
        if overlap:
            logging.info("   [ERROR] Daughters share %d entries (should be 0)!", len(overlap))
            all_good = False
            continue

        # All checks passed!
        logging.info("   [OK] Presplit days: %s", split1_presplit_days)
        split1_daughter_days = sorted([item['mdl_day'] for item in split1_daughter])
        split2_daughter_days = sorted([item['mdl_day'] for item in split2_daughter])
        logging.info("   [OK] Split1 daughter days: %s", split1_daughter_days)
        logging.info("   [OK] Split2 daughter days: %s", split2_daughter_days)
        logging.info("   [OK] PASSED all checks")

    logging.info("%s", '='*70)
    if all_good:
        logging.info("All sanity checks PASSED!")
    else:
        logging.info("Some sanity checks FAILED - review genealogy logic!")
    logging.info("%s", '='*70)

    return all_good

def main():
    args = create_args()
    for key, value in vars(args).items():
        logging.info("  %s: %s", key, value)

    logging.info("Loading data from %s", args.image_mapping_json)
    mapping = load_json(args.image_mapping_json)
    logging.info("Total entries: %d", len(mapping.get('entries', {})))

    # Add mdl_day to each entry if it doesn't already exist
    for key, entry in mapping.get('entries', {}).items():
        if 'mdl_day' not in entry and entry.get('dayID'):
            entry['mdl_day'] = extract_mdl_day(entry.get('dayID'))

    # Check if mdl_day exists
    has_mdl_day = sum(1 for entry in mapping.get('entries', {}).values() if entry.get('mdl_day') is not None)
    logging.info("Entries with mdl_day: %d / %d", has_mdl_day, has_mdl_day)

    # Check if main_id exists
    has_main_id = sum(1 for entry in mapping.get('entries', {}).values() if entry.get('verification', {}).get('main_id'))
    logging.info("Entries with main_id: %d / %d", has_main_id, has_main_id)

    if has_mdl_day == 0:
        logging.error("No entries have mdl_day! Please regenerate %s with mdl_day.", args.image_mapping_json)
        return

    if has_main_id == 0:
        logging.error("No entries have main_id! Cannot determine split status from %s.", args.image_mapping_json)
        return

    logging.info("Organizing by genealogy...")
    genealogy = organize_by_genealogy(mapping.get('entries', {}))

    logging.info("Building complete series...")
    complete_series, complete_with_blanks, incomplete_series = build_complete_series(genealogy)

    # Sanity check splits
    sanity_check_splits(complete_series)

    logging.info("%s", '='*70)
    logging.info("SERIES COMPLETENESS ANALYSIS")
    logging.info("%s", '='*70)
    logging.info("Total unique wells: %d", len(genealogy))
    logging.info("Complete series (all %d days, NO BLANKS): %d", len(EXPECTED_DAYS), len(complete_series))
    logging.info("Complete series WITH BLANKS: %d", len(complete_with_blanks))
    logging.info("Incomplete series: %d", len(incomplete_series))

    # Analyze split patterns in complete series (no blanks)
    split_types = defaultdict(int)
    for series in complete_series:
        split_types[series['split_genealogy']] += 1

    logging.info("Complete series (no blanks) by type:")
    for split_type in sorted(split_types.keys()):
        logging.info("  %s: %d", split_type, split_types[split_type])

    # Analyze complete with blanks
    if complete_with_blanks:
        logging.info("Complete series WITH BLANKS by type:")
        split_types_blanks = defaultdict(int)
        for series in complete_with_blanks:
            split_types_blanks[series['split_genealogy']] += 1
        for split_type in sorted(split_types_blanks.keys()):
            logging.info("  %s: %d", split_type, split_types_blanks[split_type])

        # Show examples
        logging.info("Example series with blanks (first 5):")
        for series in complete_with_blanks[:5]:
            logging.info("  %s: blank on days %s", series['organoid_id'], series['blank_days'])

    # Show distribution of incomplete series
    if incomplete_series:
        logging.info("Incomplete series distribution:")
        day_counts = defaultdict(int)
        genealogy_counts = defaultdict(int)
        for series in incomplete_series:
            day_counts[series['n_days']] += 1
            genealogy_counts[series['split_genealogy']] += 1

        logging.info("  By number of days:")
        for n_days in sorted(day_counts.keys(), reverse=True):
            logging.info("    %d days: %d organoids", n_days, day_counts[n_days])

        logging.info("  By genealogy type:")
        for gen_type in sorted(genealogy_counts.keys()):
            logging.info("    %s: %d", gen_type, genealogy_counts[gen_type])

        # Show examples if requested
        if args.show_examples:
            logging.info("Example incomplete series (first 10):")
            for series in incomplete_series[:10]:
                blank_str = f", has blanks on {series.get('blank_days', [])}" if series.get('has_blanks') else ""
                logging.info("  %s: %d days, missing %s%s", series['organoid_id'], series['n_days'], series['missing_days'], blank_str)
                if 'n_presplit' in series:
                    logging.info("    (presplit: %d, daughter: %d)", series['n_presplit'], series['n_daughter'])

    # Create filtered data with only complete series (NO BLANKS)
    logging.info("Creating filtered dataset (NO BLANKS)...")
    filtered_data = {}
    series_metadata = {}

    for series in complete_series:
        for item in series['series']:
            filtered_data[item['key']] = item['entry']

        # Store mapping from organoid_id to entry keys (in temporal order)
        series_metadata[series['organoid_id']] = {
            'organoid_id': series['organoid_id'],
            'base_well_id': series['base_well_id'],
            'split_genealogy': series['split_genealogy'],
            'days': series['days_present'],
            'entry_keys': [item['key'] for item in series['series']],
            'n_timepoints': len(series['series'])
        }

    logging.info("Filtered entries (no blanks): %d (from %d)", len(filtered_data), len(mapping.get('entries', {})))
    logging.info("Complete organoid series (no blanks): %d", len(series_metadata))
    logging.info("Retention rate: %.1f%%", 100*len(filtered_data)/len(mapping.get('entries', {})))

    # Save filtered data (NO BLANKS)
    output_path = args.out_dir / 'complete_series_data_no_blanks.json'
    save_json(output_path, filtered_data)
    logging.info("Filtered data (NO BLANKS) saved to: %s", output_path)

    # Save series metadata (NO BLANKS)
    metadata_path = args.out_dir / 'complete_series_metadata_no_blanks.json'
    save_json(metadata_path, series_metadata)
    logging.info("Series metadata (NO BLANKS) saved to: %s", metadata_path)
    logging.info("Maps organoid_id to temporal sequence of entries")

    # Save analysis summary
    summary = {
        'total_wells': len(genealogy),
        'complete_series_no_blanks': len(complete_series),
        'complete_series_with_blanks': len(complete_with_blanks),
        'incomplete_series': len(incomplete_series),
        'complete_series_by_type': dict(split_types),
        'expected_days': EXPECTED_DAYS,
        'total_entries_original': len(mapping.get('entries', {})),
        'total_entries_filtered_no_blanks': len(filtered_data),
        'retention_rate_no_blanks': len(filtered_data)/len(mapping.get('entries', {})) if len(mapping.get('entries', {})) > 0 else 0
    }
    summary_path = args.out_dir / 'series_completeness_summary.json'
    save_json(summary_path, summary)
    logging.info("Summary saved to: %s", summary_path)

if __name__ == "__main__":
    main()