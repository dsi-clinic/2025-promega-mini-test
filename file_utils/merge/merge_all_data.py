#!/usr/bin/env python3
import json
import re
from pathlib import Path
from tqdm import tqdm
from file_utils.common.organoid_patterns import OrganoidNormalizer

from config import (
    ORIGINAL_MAPPING,
    INFER_RESIZED_DIR,
    METABOLITE_MAP_JSON,
    SURVEY_AGGREGATED_JSON,
    MANUAL_THRESHOLD_MAPPING,
    ALL_DATA_JSON,
)

OUTPUT_PATH = str(ALL_DATA_JSON)


# ---------- helpers ----------
def load_json(path: Path | str):
    """Load JSON data from file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data as dictionary or list.
    """
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def sanitize_for_json(obj):
    """Recursively sanitize data to be JSON-safe.

    Converts NaN and infinity values to None, handles pandas objects,
    and ensures all data types are JSON-serializable.

    Args:
        obj: Object to sanitize (can be dict, list, or primitive type).

    Returns:
        Sanitized object safe for JSON serialization.
    """
    import math

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif obj is None or isinstance(obj, (str, int, bool)):
        return obj
    else:
        try:
            if hasattr(obj, "isna") and obj.isna():
                return None
        except (TypeError, ValueError):
            pass
        return str(obj)


def normalized_parent_key(id_like: str) -> str:
    """Normalize organoid ID to consistent format.

    Uses OrganoidNormalizer to get consistent BA# 96_# Dy## A# format (no suffixes).

    Args:
        id_like: Organoid ID string to normalize.

    Returns:
        Normalized organoid ID string.
    """
    try:
        return OrganoidNormalizer.normalize_key(id_like)
    except ValueError:
        return OrganoidNormalizer.clean_string(id_like).upper()


def extract_mdl_day(day_id: str) -> float:
    """Extract numerical day from dayID string.

    Args:
        day_id: Day ID string (e.g., 'Dy17', 'Dy30').

    Returns:
        Day number as float (e.g., 17.0, 30.0). Returns 20.5 for both 'Dy20' and 'Dy21'.
        Returns None if day_id is empty or invalid.
    """
    if not day_id:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", day_id)
    if match:
        day_num = float(match.group(1))
        if day_num in [20.0, 21.0]:
            return 20.5
        return day_num
    return None


# ---------- merge functions ----------
def build_survey_map(survey_json):
    """Build survey map indexed by (main_id, split_index).

    Args:
        survey_json: Survey JSON data dictionary.

    Returns:
        Dictionary mapping (main_id, split_index) tuples to survey data.
    """
    survey_map = {}
    for row in survey_json.values():
        for category in ["evaluations", "quality_scores"]:
            if row.get(category):
                for item in row[category]:
                    main_id = item.get("main_id")
                    split_index = item.get("split_index")
                    if not main_id:
                        continue
                    main_id_norm = main_id.replace(" ", "_").upper()
                    key = (main_id_norm, split_index)
                    if key not in survey_map:
                        survey_map[key] = {"evaluations": [], "quality_scores": []}
                    survey_map[key][category].append(item)
    return survey_map


def normalize_manual_masks(manual_mask_map):
    """Normalize manual mask mapping keys.

    Args:
        manual_mask_map: Dictionary with raw keys to normalize.

    Returns:
        Dictionary with normalized keys.
    """
    manual_mask_normalized = {}
    for raw_key, manual_data in manual_mask_map.items():
        try:
            norm_key = OrganoidNormalizer.normalize_key(raw_key)
        except ValueError:
            norm_key = OrganoidNormalizer.clean_string(raw_key).upper()
        manual_mask_normalized[norm_key] = manual_data
    return manual_mask_normalized


def load_processed_image_mappings(infer_resized_dir):
    """Load processed image mappings from JSON files.

    Args:
        infer_resized_dir: Directory to search for processed image mapping files.

    Returns:
        Dictionary mapping keys to processed image data.
    """
    processed_map = {}
    found_files = list(Path(infer_resized_dir).rglob("image_mapping*_processed.json"))
    for p in found_files:
        raw = load_json(p)
        processed_map.update(raw)
    return processed_map


def merge_all_data(
    base_map, processed_map, survey_map, metab_map, manual_mask_normalized
):
    """Merge all data sources into combined dictionary.

    Args:
        base_map: Base mapping data dictionary.
        processed_map: Processed image mappings dictionary.
        survey_map: Survey data map indexed by (main_id, split_index).
        metab_map: Metabolite mapping dictionary.
        manual_mask_normalized: Normalized manual mask mapping dictionary.

    Returns:
        Tuple of (combined dictionary, statistics dictionary with counts).
    """
    combined = {}
    manual_mask_count = 0
    survey_matched_count = 0
    survey_not_matched_count = 0

    for raw_k, payload in tqdm(base_map.items(), desc="Merging"):
        entry = dict(payload)

        # Extract mdl_day
        if "dayID" in entry:
            entry["mdl_day"] = extract_mdl_day(entry["dayID"])

        # Match processed info
        processed = processed_map.get(raw_k) or processed_map.get(
            normalized_parent_key(raw_k)
        )
        if processed:
            entry["processed"] = processed
            entry["main_id"] = processed.get("main_id")

        norm_key_parent = normalized_parent_key(raw_k)

        # Match survey data
        main_id = entry.get("main_id", "")
        split_index = entry.get("split_index", payload.get("split_index"))
        if main_id:
            main_id_norm = main_id.replace(" ", "_").upper()
            key = (main_id_norm, split_index)
            if key in survey_map:
                entry["survey"] = survey_map[key]
                survey_matched_count += 1
            else:
                survey_not_matched_count += 1

        # Add metabolites
        if norm_key_parent in metab_map:
            entry["metabolites"] = metab_map[norm_key_parent]

        # Add manual mask path
        if norm_key_parent in manual_mask_normalized:
            manual_data = manual_mask_normalized[norm_key_parent]
            entry["manual_mask_path"] = manual_data.get("MT Mask Path")
            manual_mask_count += 1

        combined[raw_k] = entry

    stats = {
        "manual_mask_count": manual_mask_count,
        "survey_matched_count": survey_matched_count,
        "survey_not_matched_count": survey_not_matched_count,
    }

    return combined, stats


def main():
    """Main entry point for merging all data sources.

    Loads base mapping, metabolite data, survey data, manual masks, and processed images,
    then merges them into a single all_data.json file.
    """
    # ---------- load sources ----------
    print(f"Loading base mapping: {ORIGINAL_MAPPING}")
    base_json = load_json(ORIGINAL_MAPPING)
    base_map = base_json.get("entries", {})

    print(f"Loading metabolite map: {METABOLITE_MAP_JSON}")
    metab_map = load_json(METABOLITE_MAP_JSON)

    print(f"Loading survey data: {SURVEY_AGGREGATED_JSON}")
    survey_json = load_json(SURVEY_AGGREGATED_JSON)

    print(f"Loading manual threshold mapping: {MANUAL_THRESHOLD_MAPPING}")
    manual_mask_map = load_json(MANUAL_THRESHOLD_MAPPING)

    # ---------- build survey map ----------
    print("Building survey map by (main_id, split_index)...")
    survey_map = build_survey_map(survey_json)
    print(
        f"Built survey map with {len(survey_map)} unique (main_id, split_index) pairs"
    )

    # ---------- manual mask normalization ----------
    manual_mask_normalized = normalize_manual_masks(manual_mask_map)

    # ---------- load processed image mappings ----------
    processed_map = load_processed_image_mappings(INFER_RESIZED_DIR)

    # ---------- merge ----------
    combined, stats = merge_all_data(
        base_map, processed_map, survey_map, metab_map, manual_mask_normalized
    )

    # ---------- sanitize + write output ----------
    print("\nSanitizing data for JSON...")
    combined_clean = sanitize_for_json(combined)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(combined_clean, f, indent=2)

    print(f"\nWrote {len(combined_clean):,} merged records → {OUTPUT_PATH}")
    print(f"Survey matches: {stats['survey_matched_count']:,}")
    print(f"Survey not matched: {stats['survey_not_matched_count']:,}")
    print(f"Found {stats['manual_mask_count']:,} manual masks")


if __name__ == "__main__":
    main()
