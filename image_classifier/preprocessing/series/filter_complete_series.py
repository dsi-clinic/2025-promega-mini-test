# analysis/images/series/filter_complete_series.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from config import ALL_DATA_JSON, OUTPUT_FOLDER

# Define expected timepoints using mdl_day values
EXPECTED_DAYS = [3.0, 6.0, 8.0, 10.0, 13.0, 15.0, 17.0, 20.5, 24.0, 28.0, 30.0]


def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def is_blank(entry):
    """Check if an entry is a blank well"""
    if "verification" in entry:
        blank = entry["verification"].get("blank", False)
        if blank is True or blank == "true":
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
        return "nosplit"

    main_id_lower = main_id.lower()

    if "presplit" in main_id_lower:
        return "presplit"
    elif "split2" in main_id_lower:
        return "split2"
    elif "split1" in main_id_lower:
        return "split1"
    elif "nosplit" in main_id_lower:
        return "nosplit"
    else:
        # Fallback - shouldn't happen
        return "nosplit"


def parse_organoid_info(entry):
    """Extract BA, well, and split status from entry using main_id"""
    ba = entry["BA"].replace(" ", "_")
    well = entry["wellID"]

    # Get split type from main_id (ground truth!)
    main_id = entry.get("main_id", "")
    split_type = parse_split_from_main_id(main_id)

    return ba, well, split_type


def get_base_well_id(ba, well):
    """Get the base identifier for a well (without split info)"""
    return f"{ba}_{well}"


def organize_by_genealogy(data):
    """
    Organize data by well genealogy, tracking presplit and daughter organoids
    Returns: dict mapping base_well_id -> {presplit: [...], split1: [...], split2: [...], nosplit: [...]}
    """
    genealogy = defaultdict(
        lambda: {"presplit": [], "split1": [], "split2": [], "nosplit": []}
    )

    skipped_no_mdl_day = 0
    skipped_no_main_id = 0

    for key, entry in tqdm(data.items(), desc="Organizing by genealogy"):
        # Skip if no mdl_day
        if "mdl_day" not in entry or entry["mdl_day"] is None:
            skipped_no_mdl_day += 1
            continue

        # Skip if no main_id
        if "main_id" not in entry or not entry["main_id"]:
            skipped_no_main_id += 1
            continue

        ba, well, split_type = parse_organoid_info(entry)
        base_well_id = get_base_well_id(ba, well)
        mdl_day = entry["mdl_day"]

        genealogy[base_well_id][split_type].append(
            {
                "key": key,
                "entry": entry,
                "mdl_day": mdl_day,
                "split_type": split_type,
                "main_id": entry["main_id"],
                "is_blank": is_blank(entry),
            }
        )

    if skipped_no_mdl_day > 0:
        print(f"  Skipped {skipped_no_mdl_day} entries with no mdl_day")
    if skipped_no_main_id > 0:
        print(f"  Skipped {skipped_no_main_id} entries with no main_id")

    # Sort each list by mdl_day
    for base_well_id in genealogy:
        for split_type in genealogy[base_well_id]:
            genealogy[base_well_id][split_type].sort(key=lambda x: x["mdl_day"])

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
        presplit = splits["presplit"]
        split1 = splits["split1"]
        split2 = splits["split2"]
        nosplit = splits["nosplit"]

        # Case 1: No split - simple case
        if nosplit and not presplit and not split1 and not split2:
            days_present = set(item["mdl_day"] for item in nosplit)
            missing_days = set(EXPECTED_DAYS) - days_present
            is_complete = len(missing_days) == 0

            # Check for blanks
            has_blanks = any(item["is_blank"] for item in nosplit)
            blank_days = [item["mdl_day"] for item in nosplit if item["is_blank"]]

            series_info = {
                "organoid_id": f"{base_well_id}_nosplit",
                "base_well_id": base_well_id,
                "split_genealogy": "nosplit",
                "days_present": sorted(list(days_present)),
                "missing_days": sorted(list(missing_days)),
                "n_days": len(days_present),
                "is_complete": is_complete,
                "has_blanks": has_blanks,
                "blank_days": blank_days,
                "series": nosplit,
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
            for daughter_name, daughter_data in [
                ("split1", split1),
                ("split2", split2),
            ]:
                if not daughter_data:
                    continue

                # Combine presplit + daughter
                combined_series = presplit + daughter_data
                days_present = set(item["mdl_day"] for item in combined_series)
                missing_days = set(EXPECTED_DAYS) - days_present
                is_complete = len(missing_days) == 0

                # Check for blanks
                has_blanks = any(item["is_blank"] for item in combined_series)
                blank_days = [
                    item["mdl_day"] for item in combined_series if item["is_blank"]
                ]

                series_info = {
                    "organoid_id": f"{base_well_id}_{daughter_name}",
                    "base_well_id": base_well_id,
                    "split_genealogy": f"presplit+{daughter_name}",
                    "days_present": sorted(list(days_present)),
                    "missing_days": sorted(list(missing_days)),
                    "n_days": len(days_present),
                    "is_complete": is_complete,
                    "has_blanks": has_blanks,
                    "blank_days": blank_days,
                    "series": combined_series,
                    "n_presplit": len(presplit),
                    "n_daughter": len(daughter_data),
                }

                if is_complete and not has_blanks:
                    complete_series.append(series_info)
                elif is_complete and has_blanks:
                    complete_with_blanks.append(series_info)
                else:
                    incomplete_series.append(series_info)

        # Case 3: Only presplit (no daughters found) - incomplete by definition
        elif presplit and not split1 and not split2:
            days_present = set(item["mdl_day"] for item in presplit)
            has_blanks = any(item["is_blank"] for item in presplit)
            incomplete_series.append(
                {
                    "organoid_id": f"{base_well_id}_presplit_only",
                    "base_well_id": base_well_id,
                    "split_genealogy": "presplit_only",
                    "days_present": sorted(list(days_present)),
                    "missing_days": sorted(list(set(EXPECTED_DAYS) - days_present)),
                    "n_days": len(days_present),
                    "is_complete": False,
                    "has_blanks": has_blanks,
                    "series": presplit,
                }
            )

        # Case 4: Only daughters (no presplit) - need presplit for completeness
        elif (split1 or split2) and not presplit:
            for daughter_name, daughter_data in [
                ("split1", split1),
                ("split2", split2),
            ]:
                if not daughter_data:
                    continue
                days_present = set(item["mdl_day"] for item in daughter_data)
                has_blanks = any(item["is_blank"] for item in daughter_data)
                incomplete_series.append(
                    {
                        "organoid_id": f"{base_well_id}_{daughter_name}_no_presplit",
                        "base_well_id": base_well_id,
                        "split_genealogy": f"{daughter_name}_only",
                        "days_present": sorted(list(days_present)),
                        "missing_days": sorted(list(set(EXPECTED_DAYS) - days_present)),
                        "n_days": len(days_present),
                        "is_complete": False,
                        "has_blanks": has_blanks,
                        "series": daughter_data,
                    }
                )

    return complete_series, complete_with_blanks, incomplete_series


def sanity_check_splits(complete_series, data):
    """
    Verify that split daughters share the same presplit parent data
    """
    print(f"\n{'=' * 70}")
    print("SANITY CHECK: Split Genealogy")
    print(f"{'=' * 70}")

    # Find wells that have both split1 and split2
    split_wells = defaultdict(lambda: {"split1": None, "split2": None})

    for series in complete_series:
        if "presplit+split1" in series["split_genealogy"]:
            split_wells[series["base_well_id"]]["split1"] = series
        elif "presplit+split2" in series["split_genealogy"]:
            split_wells[series["base_well_id"]]["split2"] = series

    # Filter to only wells with BOTH daughters
    both_daughters = {
        k: v
        for k, v in split_wells.items()
        if v["split1"] is not None and v["split2"] is not None
    }

    print(f"Found {len(both_daughters)} wells with both split daughters")

    if not both_daughters:
        print("No wells with both daughters to check!")
        return True

    # Check first few examples
    all_good = True
    examples_to_check = min(5, len(both_daughters))

    print(f"\nChecking {examples_to_check} examples:")
    for i, (base_well_id, daughters) in enumerate(
        list(both_daughters.items())[:examples_to_check]
    ):
        split1_series = daughters["split1"]
        split2_series = daughters["split2"]

        print(f"\n{i + 1}. {base_well_id}:")

        # Get presplit entries for each daughter
        split1_presplit = [
            item for item in split1_series["series"] if item["split_type"] == "presplit"
        ]
        split2_presplit = [
            item for item in split2_series["series"] if item["split_type"] == "presplit"
        ]

        # Get daughter-specific entries
        split1_daughter = [
            item for item in split1_series["series"] if item["split_type"] == "split1"
        ]
        split2_daughter = [
            item for item in split2_series["series"] if item["split_type"] == "split2"
        ]

        print(
            f"   Split1: {split1_series['n_presplit']} presplit + {len(split1_daughter)} daughter days"
        )
        print(
            f"   Split2: {split2_series['n_presplit']} presplit + {len(split2_daughter)} daughter days"
        )

        # Check 1: Same number of presplit entries
        if len(split1_presplit) != len(split2_presplit):
            print(f"   [ERROR] Different presplit counts!")
            all_good = False
            continue

        # Check 2: Same presplit days
        split1_presplit_days = sorted([item["mdl_day"] for item in split1_presplit])
        split2_presplit_days = sorted([item["mdl_day"] for item in split2_presplit])

        if split1_presplit_days != split2_presplit_days:
            print(f"   [ERROR] Different presplit days!")
            print(f"      Split1 presplit: {split1_presplit_days}")
            print(f"      Split2 presplit: {split2_presplit_days}")
            all_good = False
            continue

        # Check 3: Actually the same entry keys (same images)
        split1_presplit_keys = sorted([item["key"] for item in split1_presplit])
        split2_presplit_keys = sorted([item["key"] for item in split2_presplit])

        if split1_presplit_keys != split2_presplit_keys:
            print(f"   [ERROR] Different presplit entry keys!")
            all_good = False
            continue

        # Check 4: Daughters have different data
        split1_daughter_keys = sorted([item["key"] for item in split1_daughter])
        split2_daughter_keys = sorted([item["key"] for item in split2_daughter])

        overlap = set(split1_daughter_keys) & set(split2_daughter_keys)
        if overlap:
            print(f"   [ERROR] Daughters share {len(overlap)} entries (should be 0)!")
            all_good = False
            continue

        # All checks passed!
        print(f"   [OK] Presplit days: {split1_presplit_days}")
        split1_daughter_days = sorted([item["mdl_day"] for item in split1_daughter])
        split2_daughter_days = sorted([item["mdl_day"] for item in split2_daughter])
        print(f"   [OK] Split1 daughter days: {split1_daughter_days}")
        print(f"   [OK] Split2 daughter days: {split2_daughter_days}")
        print(f"   [OK] PASSED all checks")

    print(f"\n{'=' * 70}")
    if all_good:
        print("[PASSED] All sanity checks PASSED!")
    else:
        print("[FAILED] Some sanity checks FAILED - review genealogy logic!")
    print(f"{'=' * 70}\n")

    return all_good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show detailed examples of incomplete series",
    )
    args = parser.parse_args()

    print(f"Loading data from {ALL_DATA_JSON}")
    data = load_json(ALL_DATA_JSON)
    print(f"Total entries: {len(data)}")

    # Check if mdl_day exists
    has_mdl_day = sum(1 for entry in data.values() if entry.get("mdl_day") is not None)
    print(f"Entries with mdl_day: {has_mdl_day} / {len(data)}")

    # Check if main_id exists
    has_main_id = sum(1 for entry in data.values() if entry.get("main_id"))
    print(f"Entries with main_id: {has_main_id} / {len(data)}")

    if has_mdl_day == 0:
        print(
            "\nERROR: No entries have mdl_day! Please regenerate all_data.json with mdl_day."
        )
        return

    if has_main_id == 0:
        print("\nERROR: No entries have main_id! Cannot determine split status.")
        return

    print("\nOrganizing by genealogy...")
    genealogy = organize_by_genealogy(data)

    print(f"\nBuilding complete series...")
    complete_series, complete_with_blanks, incomplete_series = build_complete_series(
        genealogy
    )

    # Sanity check splits
    sanity_check_splits(complete_series, data)

    print(f"\n{'=' * 70}")
    print("SERIES COMPLETENESS ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Total unique wells: {len(genealogy)}")
    print(
        f"Complete series (all {len(EXPECTED_DAYS)} days, NO BLANKS): {len(complete_series)}"
    )
    print(f"Complete series WITH BLANKS: {len(complete_with_blanks)}")
    print(f"Incomplete series: {len(incomplete_series)}")

    # Analyze split patterns in complete series (no blanks)
    split_types = defaultdict(int)
    for series in complete_series:
        split_types[series["split_genealogy"]] += 1

    print("\nComplete series (no blanks) by type:")
    for split_type in sorted(split_types.keys()):
        print(f"  {split_type}: {split_types[split_type]}")

    # Analyze complete with blanks
    if complete_with_blanks:
        print(f"\nComplete series WITH BLANKS by type:")
        split_types_blanks = defaultdict(int)
        for series in complete_with_blanks:
            split_types_blanks[series["split_genealogy"]] += 1
        for split_type in sorted(split_types_blanks.keys()):
            print(f"  {split_type}: {split_types_blanks[split_type]}")

        # Show examples
        print("\nExample series with blanks (first 5):")
        for series in complete_with_blanks[:5]:
            print(f"  {series['organoid_id']}: blank on days {series['blank_days']}")

    # Show distribution of incomplete series
    if incomplete_series:
        print("\nIncomplete series distribution:")
        day_counts = defaultdict(int)
        genealogy_counts = defaultdict(int)
        for series in incomplete_series:
            day_counts[series["n_days"]] += 1
            genealogy_counts[series["split_genealogy"]] += 1

        print("  By number of days:")
        for n_days in sorted(day_counts.keys(), reverse=True):
            print(f"    {n_days} days: {day_counts[n_days]} organoids")

        print("  By genealogy type:")
        for gen_type in sorted(genealogy_counts.keys()):
            print(f"    {gen_type}: {genealogy_counts[gen_type]}")

        # Show examples if requested
        if args.show_examples:
            print("\nExample incomplete series (first 10):")
            for series in incomplete_series[:10]:
                blank_str = (
                    f", has blanks on {series.get('blank_days', [])}"
                    if series.get("has_blanks")
                    else ""
                )
                print(
                    f"  {series['organoid_id']}: {series['n_days']} days, missing {series['missing_days']}{blank_str}"
                )
                if "n_presplit" in series:
                    print(
                        f"    (presplit: {series['n_presplit']}, daughter: {series['n_daughter']})"
                    )

    # Create filtered data with only complete series (NO BLANKS)
    print(f"\nCreating filtered dataset (NO BLANKS)...")
    filtered_data = {}
    series_metadata = {}

    for series in complete_series:
        for item in series["series"]:
            filtered_data[item["key"]] = item["entry"]

        # Store mapping from organoid_id to entry keys (in temporal order)
        series_metadata[series["organoid_id"]] = {
            "organoid_id": series["organoid_id"],
            "base_well_id": series["base_well_id"],
            "split_genealogy": series["split_genealogy"],
            "days": series["days_present"],
            "entry_keys": [item["key"] for item in series["series"]],
            "n_timepoints": len(series["series"]),
        }

    print(f"Filtered entries (no blanks): {len(filtered_data)} (from {len(data)})")
    print(f"Complete organoid series (no blanks): {len(series_metadata)}")
    print(f"Retention rate: {100 * len(filtered_data) / len(data):.1f}%")

    # Save filtered data (NO BLANKS)
    output_path = OUTPUT_FOLDER / "complete_series_data_no_blanks.json"
    save_json(output_path, filtered_data)
    print(f"\nFiltered data (NO BLANKS) saved to: {output_path}")

    # Save series metadata (NO BLANKS)
    metadata_path = OUTPUT_FOLDER / "complete_series_metadata_no_blanks.json"
    save_json(metadata_path, series_metadata)
    print(f"Series metadata (NO BLANKS) saved to: {metadata_path}")
    print("Maps organoid_id to temporal sequence of entries")

    # Save analysis summary
    summary = {
        "total_wells": len(genealogy),
        "complete_series_no_blanks": len(complete_series),
        "complete_series_with_blanks": len(complete_with_blanks),
        "incomplete_series": len(incomplete_series),
        "complete_series_by_type": dict(split_types),
        "expected_days": EXPECTED_DAYS,
        "total_entries_original": len(data),
        "total_entries_filtered_no_blanks": len(filtered_data),
        "retention_rate_no_blanks": len(filtered_data) / len(data)
        if len(data) > 0
        else 0,
    }
    summary_path = OUTPUT_FOLDER / "series_completeness_summary.json"
    save_json(summary_path, summary)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
