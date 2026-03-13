#!/usr/bin/env python3 -u
"""
Unified reproducible train/val/test split for image and metabolite models.

CRITICAL: Splits by ORGANOID, not by individual samples!
This ensures the same organoid across all timepoints stays together in train/val/test.
Prevents data leakage when training on early days to predict Dy30 outcomes.

HIGH QUALITY DATA ONLY:
- BA1+BA2 batches only (high quality batches)
- 4/5 vote consensus required for labels
- Complete metabolite data required (all 4 metabolites)
- Valid processed images required (img_path + mask_path)

4 SWITCHES (Image Filtering Only):
1. exclude_stitched_only: Exclude stitched images only (keep split)
2. exclude_split_only: Exclude split/presplit images only (keep stitched)
3. exclude_both: Exclude both stitched AND split/presplit images
4. exclude_nothing: Include all images (no filtering)

ORGANOID-LEVEL EXCLUSION:
If ANY day has a problematic image type (per switch), the ENTIRE organoid
is excluded from all days (including metabolite data).
"""

import json
import argparse
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import re

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(
    sys.stdout, "reconfigure"
) else None

# ============================================================
# CONFIGURATION
# ============================================================
ALL_DATA_JSON = "all_data.json"
RANDOM_SEED = 42  # Fixed seed for reproducibility
TEST_SIZE = 0.2  # 20% test set (held out)
VAL_SIZE = 0.1  # 10% validation set (within the 80% training set)
# Final ratios: 72% train / 8% val / 20% test (80/20 training/testing, 90/10 train/val within training)

# Good metabolites (based on IDOR/Promega restrictions)
# Always included:
# - GlucoseGlo [OK]
# - GlutamateGlo [OK]
# - LactateGlo [OK]
# - PyruvateGlo [OK]
#
# Conditionally included:
# - MalateGlo: included for days >10, excluded for days ≤10 (inclusive)
#
# Excluded metabolites:
# - BCAAGlo: completely excluded (do not use at all)
REQUIRED_METABOLITES = ["GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo"]
MALATE_EXCLUSION_THRESHOLD_DAY = 10  # Don't use MalateGlo for days ≤10

# Target day for survey labels (labels come from Dy30)
LABEL_DAY = "Dy30"

# High quality batches only
HIGH_QUALITY_BATCHES = ["BA1", "BA2"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def compute_majority_label(evaluations, min_votes=4):
    """Compute majority label from survey evaluations. Requires 4/5 votes for high quality."""
    if not evaluations or len(evaluations) != 5:
        return None

    votes = {}
    for eval_data in evaluations:
        evaluation = eval_data.get("evaluation", "")
        if evaluation:
            votes[evaluation] = votes.get(evaluation, 0) + 1

    acceptable = votes.get("Acceptable", 0)
    not_acceptable = votes.get("Not Acceptable", 0)

    if acceptable >= min_votes:
        return "Acceptable"
    elif not_acceptable >= min_votes:
        return "Not Acceptable"
    else:
        return None


def extract_organoid_id(key):
    """Extract organoid ID without day from key.

    Args:
        key: Organoid key string (e.g., 'BA1 96_1 Dy30 A1').

    Returns:
        Organoid ID string without day (e.g., 'BA1 96_1 A1').
        Returns original key if pattern not found.
    """
    match = re.match(r"^(.*)\s+Dy\d+\s+(.*)$", key)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return key


def extract_day_number(day_id):
    """Extract numeric day from dayID string.

    Args:
        day_id: Day ID string (e.g., 'Dy03', 'Dy30').

    Returns:
        Day number as integer (e.g., 3, 30), or None if invalid.
    """
    if not day_id:
        return None
    match = re.match(r"^Dy(\d+)$", day_id)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def has_complete_metabolites(metabolites):
    """Check if sample has all required metabolites with valid data.

    Args:
        metabolites: Dictionary of metabolite data.

    Returns:
        True if all required metabolites have valid concentration_uM values, False otherwise.
    """
    if not metabolites:
        return False

    for met_name in REQUIRED_METABOLITES:
        if met_name not in metabolites:
            return False
        if "concentration_uM" not in metabolites[met_name]:
            return False
        if metabolites[met_name]["concentration_uM"] is None:
            return False

    return True


def get_batch_prefix(ba_string):
    """Extract batch prefix from full batch string.

    Args:
        ba_string: Full batch string (e.g., 'BA1 96_1' or 'BA2').

    Returns:
        Batch prefix string (e.g., 'BA1', 'BA2'), or None if ba_string is empty.
    """
    if not ba_string:
        return None
    return ba_string.split()[0] if " " in ba_string else ba_string


def has_valid_image_data(record):
    """Check if record has valid processed image data.

    Args:
        record: Data record dictionary.

    Returns:
        True if record has processed image data with img_path and mask_path, False otherwise.
    """
    return (
        "processed" in record
        and record["processed"]
        and "img_path" in record["processed"]
        and "mask_path" in record["processed"]
    )


# ============================================================
# IMAGE TYPE FILTERING FUNCTIONS
# ============================================================


def is_stitched(common_key, img_path=None):
    """Check if sample is stitched.

    Args:
        common_key: Common key string to check.
        img_path: Optional image path string to check.

    Returns:
        True if sample is stitched, False otherwise.
        Explicitly returns False for "nostitch" patterns.
    """
    common_key_lower = str(common_key).lower() if common_key else ""
    img_path_lower = str(img_path).lower() if img_path else ""

    # Check for stitched, but allow "nostitch" patterns
    if "stitched" in common_key_lower:
        if (
            "nostitch" in common_key_lower
            or "no_stitch" in common_key_lower
            or "no-stitch" in common_key_lower
        ):
            return False  # Explicitly not stitched
        return True

    if img_path and "stitched" in img_path_lower:
        if (
            "nostitch" in img_path_lower
            or "no_stitch" in img_path_lower
            or "no-stitch" in img_path_lower
        ):
            return False  # Explicitly not stitched
        return True

    return False


def is_split_or_presplit(common_key, img_path=None):
    """Check if sample is split or presplit.

    Args:
        common_key: Common key string to check.
        img_path: Optional image path string to check.

    Returns:
        True if sample is split or presplit, False otherwise.
        Explicitly excludes "nosplit" patterns.
    """
    common_key_lower = str(common_key).lower() if common_key else ""
    img_path_lower = str(img_path).lower() if img_path else ""

    # Check for presplit
    if "presplit" in common_key_lower or (img_path and "presplit" in img_path_lower):
        return True

    # Check for split (but NOT nosplit)
    if "split" in common_key_lower and "nosplit" not in common_key_lower:
        return True

    if img_path and "split" in img_path_lower and "nosplit" not in img_path_lower:
        return True

    return False


def should_exclude_image(common_key, img_path, switch_mode):
    """
    Determine if an image should be excluded based on switch mode.

    Args:
        common_key: Common key from all_data.json
        img_path: Image path string
        switch_mode: One of 'exclude_stitched_only', 'exclude_split_only', 'exclude_both', 'exclude_nothing'

    Returns:
        True if image should be excluded, False otherwise
    """
    if switch_mode == "exclude_nothing":
        return False

    is_stitched_flag = is_stitched(common_key, img_path)
    is_split_flag = is_split_or_presplit(common_key, img_path)

    if switch_mode == "exclude_stitched_only":
        return is_stitched_flag
    elif switch_mode == "exclude_split_only":
        return is_split_flag
    elif switch_mode == "exclude_both":
        return is_stitched_flag or is_split_flag

    return False


# ============================================================
# DATA COLLECTION FUNCTIONS
# ============================================================


def collect_organoid_data(all_data, switch_mode="exclude_nothing"):
    """
    Collect all timepoints for organoids, grouped by organoid ID.

    HIGH QUALITY FILTERS APPLIED:
    - Only BA1+BA2 batches
    - Only organoids with Dy30 labels (4/5 vote consensus)
    - Only organoids with complete metabolite data
    - Only organoids with valid processed images

    ORGANOID-LEVEL EXCLUSION:
    - If ANY day has problematic image type (per switch_mode), exclude ENTIRE organoid
    - All timepoints and metabolite data are excluded for that organoid

    Args:
        all_data: Full all_data.json dictionary
        switch_mode: Image filtering mode ('exclude_stitched_only', 'exclude_split_only',
                    'exclude_both', 'exclude_nothing')

    Returns:
        organoid_dict: {organoid_id: {'label': ..., 'batch': ..., 'timepoints': {...}}}
    """
    # First pass: get Dy30 labels for each organoid (with high quality filters)
    organoid_labels = {}

    for key, value in all_data.items():
        # Check if this is Dy30 with survey label
        if value.get("dayID") != LABEL_DAY:
            continue

        # Check batch (HIGH QUALITY: BA1+BA2 only)
        batch = get_batch_prefix(value.get("BA"))
        if batch not in HIGH_QUALITY_BATCHES:
            continue

        # Get label from survey (HIGH QUALITY: 4/5 votes required)
        if "survey" not in value:
            continue

        evaluations = value["survey"].get("evaluations", [])
        label = compute_majority_label(evaluations, min_votes=4)
        if label is None:
            continue

        # Extract organoid ID
        organoid_id = extract_organoid_id(key)
        organoid_labels[organoid_id] = label

    print(
        f"  Found {len(organoid_labels)} organoids with Dy30 labels in {HIGH_QUALITY_BATCHES}"
    )

    # Second pass: collect all timepoints for labeled organoids
    # First, collect ALL timepoints for each organoid
    organoid_all_timepoints = {}

    for key, value in all_data.items():
        # Extract organoid ID and check if it has a label
        organoid_id = extract_organoid_id(key)
        if organoid_id not in organoid_labels:
            continue

        # Check batch
        batch = get_batch_prefix(value.get("BA"))
        if batch not in HIGH_QUALITY_BATCHES:
            continue

        # Check if has valid image data
        if not has_valid_image_data(value):
            continue

        # Check metabolites (HIGH QUALITY: complete metabolites required)
        has_metabolites = has_complete_metabolites(value.get("metabolites"))
        if not has_metabolites:
            continue  # Skip if missing metabolites (high quality requirement)

        # Initialize organoid entry if needed
        if organoid_id not in organoid_all_timepoints:
            organoid_all_timepoints[organoid_id] = {
                "label": organoid_labels[organoid_id],
                "batch": batch,
                "timepoints": {},
            }

        # Add this timepoint
        day = value.get("dayID")
        # Merge Dy20 and Dy21 into Dy20_5 (they represent the same timepoint)
        if day in ["Dy20", "Dy21"]:
            day = "Dy20_5"

        # Store timepoint with metadata for filtering
        common_key = key
        img_path = value["processed"]["img_path"] if "processed" in value else None

        organoid_all_timepoints[organoid_id]["timepoints"][day] = {
            "common_key": common_key,
            "img_path": img_path,
            "value": value,  # Store full value for later processing
        }

    # Third pass: Apply organoid-level exclusion based on switch_mode
    # If ANY day has problematic image, exclude ENTIRE organoid
    organoid_data = {}

    for organoid_id, org_info in organoid_all_timepoints.items():
        # Check if ANY timepoint has problematic image type
        should_exclude_organoid = False

        for day, timepoint_info in org_info["timepoints"].items():
            common_key = timepoint_info["common_key"]
            img_path = timepoint_info["img_path"]

            if should_exclude_image(common_key, img_path, switch_mode):
                should_exclude_organoid = True
                break  # Found problematic image, exclude entire organoid

        # Skip this organoid if it should be excluded
        if should_exclude_organoid:
            continue

        # Organoid passed all filters - include it with all timepoints
        organoid_data[organoid_id] = {
            "label": org_info["label"],
            "batch": org_info["batch"],
            "timepoints": {},
        }

        # Process timepoints and add metabolites
        for day, timepoint_info in org_info["timepoints"].items():
            value = timepoint_info["value"]

            timepoint_data = {
                "img_path": value["processed"]["img_path"],
                "mask_path": value["processed"]["mask_path"],
                "day": day,
            }

            # Add metabolites (all required metabolites are present by this point)
            metabolites_dict = {}
            for met in REQUIRED_METABOLITES:
                met_data = value["metabolites"][met]
                metabolites_dict[f"{met}_concentration_uM"] = met_data.get(
                    "concentration_uM"
                )
                metabolites_dict[f"{met}_initial_concentration"] = met_data.get(
                    "initial_concentration"
                )

            # Conditionally include MalateGlo for days >10
            day_num = extract_day_number(day)
            if day_num is not None and day_num > MALATE_EXCLUSION_THRESHOLD_DAY:
                if "MalateGlo" in value.get("metabolites", {}):
                    malate_data = value["metabolites"]["MalateGlo"]
                    if (
                        "concentration_uM" in malate_data
                        and malate_data["concentration_uM"] is not None
                    ):
                        metabolites_dict["MalateGlo_concentration_uM"] = malate_data[
                            "concentration_uM"
                        ]
                    if (
                        "initial_concentration" in malate_data
                        and malate_data["initial_concentration"] is not None
                    ):
                        metabolites_dict[
                            "MalateGlo_initial_concentration"
                        ] = malate_data["initial_concentration"]

            timepoint_data["metabolites"] = metabolites_dict
            organoid_data[organoid_id]["timepoints"][day] = timepoint_data

    return organoid_data


# ============================================================
# SPLIT FUNCTIONS
# ============================================================


def split_by_organoid(
    organoid_data, random_seed=RANDOM_SEED, test_size=TEST_SIZE, val_size=VAL_SIZE
):
    """Split organoids into train/val/test sets with stratification by label.

    Structure:
    1. First split: 80% training / 20% test (held out)
    2. Within 80% training: split into train/val (90% train, 10% val of training set)

    Args:
        organoid_data: Dictionary mapping organoid IDs to organoid data dictionaries.
        random_seed: Random seed for reproducibility.
        test_size: Proportion of data to use for test set (default: 0.2).
        val_size: Proportion of training data to use for validation (default: 0.1).

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries in same format as organoid_data.
    """
    if not organoid_data:
        return {}, {}, {}

    # Extract organoid IDs and labels
    organoid_ids = list(organoid_data.keys())
    labels = [organoid_data[oid]["label"] for oid in organoid_ids]

    # First split: 80% training / 20% test (held out)
    train_test_ids, test_ids = train_test_split(
        organoid_ids, test_size=test_size, stratify=labels, random_state=random_seed
    )

    # Extract labels for the training set
    train_test_labels = [organoid_data[oid]["label"] for oid in train_test_ids]

    # Second split: Within training set, split into train/val
    # val_size is relative to the training set (e.g., 0.1 = 10% of training set goes to val)
    train_ids, val_ids = train_test_split(
        train_test_ids,
        test_size=val_size,
        stratify=train_test_labels,
        random_state=random_seed,
    )

    # Create train, val, and test dictionaries
    train_data = {oid: organoid_data[oid] for oid in train_ids}
    val_data = {oid: organoid_data[oid] for oid in val_ids}
    test_data = {oid: organoid_data[oid] for oid in test_ids}

    return train_data, val_data, test_data


# ============================================================
# OUTPUT FUNCTIONS
# ============================================================


def get_output_filename_suffix(switch_mode):
    """Get output filename suffix based on switch mode.

    Naming convention matches metabolite classifier expectations:
    - 'exclude_nothing' → 'base' → generates both_train_base.json, both_val_base.json, both_test_base.json
      (This is the default and matches what metabolite classifier uses)
    - Other modes generate suffixes that are appended to 'both_train_', 'both_val_', 'both_test_'

    This ensures compatibility with both image and metabolite classifiers.

    Args:
        switch_mode: One of 'exclude_stitched_only', 'exclude_split_only', 'exclude_both', 'exclude_nothing'.

    Returns:
        Filename suffix string (e.g., 'base', 'exclude_stitch_only').
    """
    mode_map = {
        "exclude_stitched_only": "exclude_stitch_only",
        "exclude_split_only": "exclude_split_only",
        "exclude_both": "base_no_stitch",
        "exclude_nothing": "base",  # Default: matches metabolite classifier naming (both_train_base.json)
    }
    return mode_map.get(switch_mode, switch_mode)


def save_splits(train_data, val_data, test_data, switch_mode):
    """Save train/val/test splits to JSON files with appropriate naming.

    Naming convention: both_train_<suffix>.json, both_val_<suffix>.json, both_test_<suffix>.json
    This matches the metabolite classifier convention for compatibility.
    Default (exclude_nothing) generates: both_train_base.json, both_val_base.json, both_test_base.json

    Args:
        train_data: Training split data dictionary.
        val_data: Validation split data dictionary.
        test_data: Test split data dictionary.
        switch_mode: Switch mode for determining filename suffix.

    Returns:
        Tuple of (train_file, val_file, test_file) Path objects.
    """
    output_dir = Path("data_splits")
    output_dir.mkdir(exist_ok=True)

    suffix = get_output_filename_suffix(switch_mode)

    # Naming convention: both_train_<suffix>.json (matches metabolite classifier)
    train_file = output_dir / f"both_train_{suffix}.json"
    val_file = output_dir / f"both_val_{suffix}.json"
    test_file = output_dir / f"both_test_{suffix}.json"

    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=2)

    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)

    return train_file, val_file, test_file


def print_statistics(data_dict, name):
    """Print statistics about a dataset.

    Args:
        data_dict: Dictionary mapping organoid IDs to organoid data.
        name: Name to display for the dataset in output.
    """
    if not data_dict:
        print(f"  {name}: 0 organoids")
        return

    # Count organoids by label
    labels = [v["label"] for v in data_dict.values()]
    acceptable = labels.count("Acceptable")
    not_acceptable = labels.count("Not Acceptable")

    # Count total samples (all timepoints)
    total_samples = sum(len(v["timepoints"]) for v in data_dict.values())

    # Count timepoints per day
    day_counts = {}
    for org_data in data_dict.values():
        for day in org_data["timepoints"].keys():
            day_counts[day] = day_counts.get(day, 0) + 1

    print(f"  {name}:")
    print(
        f"    - {len(data_dict)} organoids ({acceptable} Acceptable, {not_acceptable} Not Acceptable)"
    )
    print(f"    - {total_samples} total samples across all timepoints")
    print(f"    - Days available: {sorted(day_counts.keys())}")
    for day in sorted(day_counts.keys()):
        print(f"      {day}: {day_counts[day]} samples")


# ============================================================
# MAIN MODES
# ============================================================


def run_split_mode(all_data, switch_mode):
    """Run split with specified switch mode.

    Collects organoid data, applies filtering based on switch mode, splits data,
    and saves train/val/test splits to JSON files.

    Args:
        all_data: Full all_data.json dictionary.
        switch_mode: One of 'exclude_stitched_only', 'exclude_split_only',
                    'exclude_both', 'exclude_nothing'.

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries.
    """
    mode_names = {
        "exclude_stitched_only": "Exclude Stitched Only",
        "exclude_split_only": "Exclude Split/Presplit Only",
        "exclude_both": "Exclude Both Stitched and Split",
        "exclude_nothing": "Exclude Nothing (Include All)",
    }

    print("\n" + "=" * 60)
    print(f"SWITCH MODE: {mode_names.get(switch_mode, switch_mode)}")
    print("=" * 60)
    print("High Quality Filters: BA1+BA2, 4/5 votes, complete metabolites")
    print(f"Image Filtering: {mode_names.get(switch_mode, switch_mode)}")
    print(
        "Organoid-Level Exclusion: If ANY day has problematic image, exclude entire organoid"
    )

    # Collect organoid data with filtering
    organoid_data = collect_organoid_data(all_data, switch_mode=switch_mode)

    print(f"\nCollected data for {len(organoid_data)} organoids")

    # Split by organoid: 80% training / 20% test, then split training into train/val
    train_data, val_data, test_data = split_by_organoid(
        organoid_data, random_seed=RANDOM_SEED
    )

    print("\nTrain/Val/Test Split:")
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation (within training)")
    print_statistics(test_data, "Test (held out)")

    # Save
    train_file, val_file, test_file = save_splits(
        train_data, val_data, test_data, switch_mode
    )
    print(f"\n[OK] Saved: {train_file}")
    print(f"[OK] Saved: {val_file}")
    print(f"[OK] Saved: {test_file}")

    return train_data, val_data, test_data


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified reproducible train/val/test split for high quality data (BA1+BA2 only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Switch Modes (Image Filtering Only):
  1. exclude_stitched_only: Exclude stitched images only (keep split)
  2. exclude_split_only: Exclude split/presplit images only (keep stitched)
  3. exclude_both: Exclude both stitched AND split/presplit images
  4. exclude_nothing: Include all images (no filtering)

High Quality Filters (Always Applied):
  - BA1+BA2 batches only
  - 4/5 vote consensus required for labels
  - Complete metabolite data required (all 4 metabolites)
  - Valid processed images required

Organoid-Level Exclusion:
  If ANY day has a problematic image type (per switch), the ENTIRE organoid
  is excluded from all days (including metabolite data).

Output files saved to data_splits/ with names matching old patterns.
        """,
    )
    parser.add_argument(
        "--switch",
        type=str,
        default="exclude_nothing",
        choices=[
            "exclude_stitched_only",
            "exclude_split_only",
            "exclude_both",
            "exclude_nothing",
        ],
        help="Switch mode for image filtering",
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate splits for all 4 switch modes"
    )

    args = parser.parse_args()

    # Load data
    print(f"\nLoading {ALL_DATA_JSON}...", flush=True)
    with open(ALL_DATA_JSON) as f:
        all_data = json.load(f)
    print(f"[OK] Loaded {len(all_data)} records", flush=True)

    print("\n[WARNING] IMPORTANT: Splitting by ORGANOID, not by individual samples!")
    print("   This prevents data leakage when training across timepoints.")
    print(f"\nUsing fixed random seed: {RANDOM_SEED}")
    print("Split structure: 80% Training / 20% Test (held out)")
    print(
        f"Within Training: {int((1 - VAL_SIZE) * 100)}% Train / {int(VAL_SIZE * 100)}% Val"
    )
    print(
        f"Final ratios: ~{int((1 - TEST_SIZE) * (1 - VAL_SIZE) * 100)}% Train / ~{int((1 - TEST_SIZE) * VAL_SIZE * 100)}% Val / {int(TEST_SIZE * 100)}% Test"
    )
    print(f"Labels from: {LABEL_DAY}")
    print("\nHIGH QUALITY DATA ONLY:")
    print(f"  - Batches: {', '.join(HIGH_QUALITY_BATCHES)} only")
    print("  - Labels: 4/5 vote consensus required")
    print("  - Metabolites: Complete data required (all 4 metabolites)")

    # Run requested switch mode(s)
    if args.all:
        switch_modes = [
            "exclude_stitched_only",
            "exclude_split_only",
            "exclude_both",
            "exclude_nothing",
        ]
        for switch_mode in switch_modes:
            run_split_mode(all_data, switch_mode)
    else:
        run_split_mode(all_data, args.switch)

    print("\n" + "=" * 60)
    print("[OK] Split complete! All files saved to data_splits/")
    print("=" * 60)
    print("\nData format: Each organoid has all its timepoints together.")
    print("Use this to train on early days and predict Dy30 outcomes!")
    print("=" * 60)


if __name__ == "__main__":
    main()
