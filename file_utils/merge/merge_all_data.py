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

def normalize_manual_mask_map(manual_mask_map, in_dir):
    """Normalize keys for storage of manual mask data and update path to data files."""
    manual_mask_normalized = {}
    for raw_key, manual_data in manual_mask_map.items():
        try:
            norm_key = OrganoidNormalizer.normalize_key(raw_key)
        except ValueError:
            norm_key = OrganoidNormalizer.clean_string(raw_key).upper()
        manual_mask_normalized[norm_key] = manual_data

        best_z = ("images", "raw_images") + Path(manual_data["Best Z Filename"]).parts[6:]
        manual_data["Best Z Filename"] = in_dir.joinpath(*best_z)
        check_existence(manual_data["Best Z Filename"])

        mt_mask = ("masks",) + Path(manual_data["MT Mask Path"]).parts[6:]
        manual_data["MT Mask Path"] = in_dir.joinpath(*mt_mask)
        check_existence(manual_data["Best Z Filename"])

    return manual_mask_normalized

def check_existence(file_path):
    """Check existence of file and raise an error if it does not exist."""
    if not file_path.exists():
        raise RuntimeError(f"Required file does not exist: {file_path}")

def build_processed_files_map(found_files, in_dir):
    """Build and return a dictionary of processed file JSON data.

    Also update hardcoded paths to point to input files on the file system.
    """
    processed_map = {}
    for p in found_files:
        raw = load_json(p)
        for batch_data in raw.values():
            img_path = ("images", INFER_RESIZED_DIR) + Path(batch_data["img_path"]).parts[7:]
            batch_data["img_path"] = in_dir.joinpath(*img_path)
            check_existence(batch_data["img_path"])

            mask_path = ("predictions",) + Path(batch_data["mask_path"]).parts[6:]
            batch_data["mask_path"] = in_dir.joinpath(*mask_path)
            check_existence(batch_data["mask_path"])

            overlay_path = ("predictions",) + Path(batch_data["overlay_path"]).parts[6:]
            batch_data["overlay_path"] = in_dir.joinpath(*overlay_path)
            check_existence(batch_data["overlay_path"])

        processed_map.update(raw)

    return processed_map

def merge_data_sources(base_map, survey_map, metab_map, manual_mask_normalized,
                       processed_map):
    """Merge and return dictionary of all data sources plus number of masks."""
    combined = {}
    manual_mask_count = 0
    survey_matched_count = 0
    survey_not_matched_count = 0

    for raw_k, payload in tqdm(base_map.items(), desc="Merging"):
        entry = dict(payload)

        # Extract mdl_day
        if 'dayID' in entry:
            entry['mdl_day'] = extract_mdl_day(entry['dayID'])

        # Match processed info
        processed = processed_map.get(raw_k) or processed_map.get(normalized_parent_key(raw_k))
        if processed:
            entry["processed"] = processed
            entry["main_id"] = processed.get("main_id")

        norm_key_parent = normalized_parent_key(raw_k)

        # ----- FIXED SURVEY MERGE LOGIC -----
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
        # ------------------------------------

        # Add metabolites
        if norm_key_parent in metab_map:
            entry["metabolites"] = metab_map[norm_key_parent]

        # Add manual mask path
        if norm_key_parent in manual_mask_normalized:
            manual_data = manual_mask_normalized[norm_key_parent]
            entry["manual_mask_path"] = manual_data.get("MT Mask Path")
            manual_mask_count += 1

        combined[raw_k] = entry

    return combined, survey_matched_count, survey_not_matched_count, manual_mask_count

def normalized_parent_key(id_like: str) -> str:
    """Use OrganoidNormalizer to get consistent BA# 96_# Dy## A# format (no suffixes)."""
    try:
        return OrganoidNormalizer.normalize_key(id_like)
    except ValueError:
        return OrganoidNormalizer.clean_string(id_like).upper()

def extract_mdl_day(day_id: str) -> float:
    """Extract numerical day from dayID (e.g., 'Dy17' -> 17.0, 'Dy20' or 'Dy21' -> 20.5)"""
    if not day_id:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)', day_id)
    if match:
        day_num = float(match.group(1))
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
        try:
            if hasattr(obj, 'isna') and obj.isna():
                return None
        except (TypeError, ValueError):
            pass
        return str(obj)

def main():
    # ---------- command line arguments ----------
    in_dir, out_dir = get_args()

    # ---------- load sources ----------
    sources = load_data_sources(in_dir)

    # Build survey map keyed by image_id or parent
    print("Building survey map by (main_id, split_index)...")
    survey_map = build_survey_map(sources.survey_json)
    print(f"Built survey map with {len(survey_map)} unique (main_id, split_index) pairs")

    # Build manual mask map with normalized keys
    print("Normalizing keys for manual mask map...")
    manual_mask_normalized = normalize_manual_mask_map(sources.manual_mask_map, in_dir)

    # Load processed JSONs
    print("Loading processed files JSON data...")
    processed_map = build_processed_files_map(sources.found_files, in_dir)

    # ---------- merge ----------
    print("Merging data sources...")
    combined, survey_matched_count, survey_not_matched_count, manual_mask_count = merge_data_sources(
        sources.base_map, survey_map, sources.metab_map, manual_mask_normalized,
        processed_map
    )

    # ---------- sanitize and write output ----------
    print("\nSanitizing data for JSON...")
    combined_clean = sanitize_for_json(combined)

    out_file = out_dir.joinpath("json", ALL_DATA_JSON)
    with open(out_file, "w") as f:
        json.dump(combined_clean, f, indent=2)

    print(f"\nWrote {len(combined_clean):,} merged records → {out_file}")
    print(f"Survey matches: {survey_matched_count:,}")
    print(f"Survey not matched: {survey_not_matched_count:,}")
    print(f"Found {manual_mask_count:,} manual masks")

if __name__ == "__main__":
    main()