#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import re
from tqdm import tqdm
from typing import NamedTuple

from file_utils.common.organoid_patterns import OrganoidNormalizer

from config import (
    ORIGINAL_MAPPING_JSON,
    INFER_RESIZED_DIR,
    METABOLITE_MAP_JSON,
    SURVEY_AGGREGATED_JSON,
    MANUAL_THRESHOLD_MAPPING_JSON,
    ALL_DATA_JSON,
)

# ---------- helpers ----------
class DataSources(NamedTuple):
    """Class to capture input data sources."""
    base_map: dict
    metab_map: dict
    survey_json: dict
    manual_mask_map: dict
    found_files: list

def get_args():
    arg_parser = create_args()
    args = arg_parser.parse_args()
    in_dir = args.indir
    out_dir = args.outdir
    for key,val in vars(args).items():
        print(f"{key}: {val}")
    return in_dir, out_dir

def create_args():
    """Create and return argparser with arguments."""
    arg_parser = argparse.ArgumentParser(description="Create a main file to aggregate all data sources")
    arg_parser.add_argument("-i",
                            "--indir",
                            type=Path,
                            help="Path to input directory on the file system")
    arg_parser.add_argument("-o",
                            "--outdir",
                            type=Path,
                            help="Path to input directory on the file system")
    return arg_parser

def load_data_sources(in_dir):
    """Load data sources and return NamedTuple with source data in memory."""
    original_mapping = in_dir.joinpath("json", ORIGINAL_MAPPING_JSON)
    print(f"Loading base mapping: {original_mapping}")
    base_json = load_json(original_mapping)
    base_map = base_json.get("entries", {})

    metabolite_map = in_dir.joinpath("json", METABOLITE_MAP_JSON)
    print(f"Loading metabolite map: {metabolite_map}")
    metab_map = load_json(metabolite_map)

    survey_aggregated = in_dir.joinpath("json", SURVEY_AGGREGATED_JSON)
    print(f"Loading survey data: {survey_aggregated}")
    survey_json = load_json(survey_aggregated)

    manual_threshold = in_dir.joinpath("json", MANUAL_THRESHOLD_MAPPING_JSON)
    print(f"Loading manual threshold mapping: {manual_threshold}")
    manual_mask_map = load_json(manual_threshold)

    infer_resized_dir = in_dir.joinpath("images", INFER_RESIZED_DIR)
    found_files = list(infer_resized_dir.rglob("image_mapping*_processed.json"))
    print(f"Located {len(found_files)} files in {infer_resized_dir}")

    return DataSources(
        base_map=base_map,
        metab_map=metab_map,
        survey_json=survey_json,
        manual_mask_map=manual_mask_map,
        found_files=found_files,
    )

def load_json(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Required JSON file does not exist: {path}")
    with path.open("r") as f:
        return json.load(f)

def build_survey_map(survey_json):
    """Build and return dictionary of survey data."""
    survey_map = {}
    for row in survey_json.values():
        ids = []
        if row.get("evaluations"):
            ids += [ev["image_id"] for ev in row["evaluations"] if "image_id" in ev]
        if row.get("quality_scores"):
            ids += [qs["image_id"] for qs in row["quality_scores"] if "image_id" in qs]

        for iid in ids:
            try:
                norm_key = OrganoidNormalizer.normalize_key(iid)
            except ValueError:
                norm_key = OrganoidNormalizer.clean_string(iid).upper()
            survey_map[norm_key] = row
    return survey_map

def normalize_manual_mask_map(manual_mask_map):
    """Normalize keys for storage of manual mask data."""
    manual_mask_normalized = {}
    for raw_key, manual_data in manual_mask_map.items():
        try:
            norm_key = OrganoidNormalizer.normalize_key(raw_key)
        except ValueError:
            norm_key = OrganoidNormalizer.clean_string(raw_key).upper()
        manual_mask_normalized[norm_key] = manual_data
    return manual_mask_normalized

def build_processed_files_map(found_files):
    """Build and return a dictionary of processed file JSON data."""
    processed_map = {}
    for p in found_files:
        raw = load_json(p)
        processed_map.update(raw)
    return processed_map

def merge_data_sources(base_map, survey_map, metab_map, manual_mask_normalized,
                       processed_map):
    """Merge and return dictionary of all data sources plus number of masks."""
    combined = {}
    manual_mask_count = 0

    for raw_k, payload in tqdm(base_map.items(), desc="Merging"):
        entry = dict(payload)

        # Add common_key for key consistency (from old structure)
        entry['common_key'] = raw_k

        # Extract day_num and mdl_day from dayID (from old structure)
        if 'dayID' in entry:
            day_id = entry['dayID']
            # Extract day_num (integer day)
            day_match = re.search(r'(\d+)', day_id)
            if day_match:
                entry['day_num'] = int(day_match.group(1))
            else:
                entry['day_num'] = None
            # Extract mdl_day (float day with special handling)
            entry['mdl_day'] = extract_mdl_day(day_id)

        # Debug: Check if fields are being added (only for first entry)
        if raw_k == list(base_map.keys())[0]:
            print(f"DEBUG: First entry fields after adding: {sorted(entry.keys())}")

        processed = processed_map.get(raw_k) or processed_map.get(normalized_parent_key(raw_k))
        if processed:
            entry["processed"] = processed
            entry["main_id"] = processed.get("main_id")

        norm_key_parent = normalized_parent_key(raw_k)

        if norm_key_parent in survey_map:
            entry["survey"] = survey_map[norm_key_parent]

        if norm_key_parent in metab_map:
            entry["metabolites"] = metab_map[norm_key_parent]

        # Add manual mask path if available
        if norm_key_parent in manual_mask_normalized:
            manual_data = manual_mask_normalized[norm_key_parent]
            entry["manual_mask_path"] = manual_data.get("MT Mask Path")
            manual_mask_count += 1

        combined[raw_k] = entry

        # Debug: Check final entry for first record
        if raw_k == list(base_map.keys())[0]:
            print(f"DEBUG: Final entry fields before sanitization: {sorted(entry.keys())}")

    return combined, manual_mask_count

def normalized_parent_key(id_like: str) -> str:
    """Use OrganoidNormalizer to get consistent BA# 96_# Dy## A# format (no suffixes)."""
    try:
        return OrganoidNormalizer.normalize_key(id_like)
    except ValueError:
        # fallback: return a stripped clean version if parsing fails
        return OrganoidNormalizer.clean_string(id_like).upper()

def extract_mdl_day(day_id: str) -> float:
    """Extract numerical day from dayID (e.g., 'Dy17' -> 17.0, 'Dy20' or 'Dy21' -> 20.5)"""
    if not day_id:
        return None
    # Extract numbers from dayID
    match = re.search(r'(\d+(?:\.\d+)?)', day_id)
    if match:
        day_num = float(match.group(1))
        # Handle day 20/21 -> 20.5 for consistency
        if day_num in [20.0, 21.0]:
            return 20.5
        return day_num
    return None

def sanitize_for_json(obj):
    """
    Recursively sanitize data to be JSON-safe.
    - Converts NaN, inf, -inf to None
    - Handles nested dicts and lists
    """
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
        # Handle pandas NA, numpy nan, etc.
        try:
            if hasattr(obj, 'isna') and obj.isna():
                return None
        except (TypeError, ValueError):
            pass
        # Try to convert to string as fallback
        return str(obj)

def main():
    # ---------- command line arguments ----------
    in_dir, out_dir = get_args()

    # ---------- load sources ----------
    sources = load_data_sources(in_dir)

    # Build survey map keyed by image_id or parent
    print("Building survey map...")
    survey_map = build_survey_map(sources.survey_json)

    # Build manual mask map with normalized keys
    print("Normalizing keys for manual mask map...")
    manual_mask_normalized = normalize_manual_mask_map(sources.manual_mask_map)

    # Load processed JSONs
    print("Loading processed files JSON data...")
    processed_map = build_processed_files_map(sources.found_files)

    # ---------- merge ----------
    print("Merging data sources...")
    combined, manual_mask_count = merge_data_sources(
        sources.base_map, survey_map, sources.metab_map, manual_mask_normalized,
        processed_map
    )

    # ---------- sanitize and write output ----------
    print("\nSanitizing data for JSON...")
    combined_clean = sanitize_for_json(combined)

    # Debug: Check first entry after sanitization
    first_key = list(combined_clean.keys())[0]
    print(f"DEBUG: First entry fields after sanitization: {sorted(combined_clean[first_key].keys())}")

    out_file = out_dir.joinpath("json", ALL_DATA_JSON)
    with open(out_file, "w") as f:
        json.dump(combined_clean, f, indent=2)

    print(f"\nWrote {len(combined_clean):,} merged records → {out_file}")
    print(f"Found {manual_mask_count:,} manual masks")

if __name__ == "__main__":
    main()