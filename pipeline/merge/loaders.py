#!/usr/bin/env python3
"""Source-data loaders + path verification for the merge step (step 16)."""

import json
import logging
import typing
from pathlib import Path

from pipeline.merge.cli import Config


class DataSources(typing.NamedTuple):
    identifiers_map: dict
    image_entries: dict
    image_meta: dict
    metab_map: dict
    survey_map: dict
    manual_mask_map: dict


def load_json(path: Path | str):
    """Load JSON file and return the parsed object."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Required JSON file does not exist: {path}")
    with path.open("r") as f:
        return json.load(f)


def check_existence(file_path: Path) -> None:
    """Raise if a required file does not exist."""
    if not file_path.exists():
        raise RuntimeError(f"Required file does not exist: {file_path}")


def load_identifiers_map(cfg: Config) -> dict:
    logging.info(f"Loading identifiers map: {cfg.identifiers_map_json}")
    return load_json(cfg.identifiers_map_json)


def load_metabolite_map(cfg: Config) -> dict:
    metabolite_file = cfg.data_dir.joinpath(cfg.METABOLITE_MAP_JSON)
    logging.info(f"Loading metabolite map: {metabolite_file}")
    return load_json(metabolite_file)


def load_survey_map(cfg: Config) -> dict:
    """Load survey data and build survey map keyed by (main_id, split_index)."""
    survey_file = cfg.data_dir.joinpath(cfg.SURVEY_AGGREGATED_JSON)
    logging.info(f"Loading survey data: {survey_file} and building survey map...")
    return load_json(survey_file)


def load_image_map(cfg: Config) -> tuple[dict, dict]:
    image_file = cfg.image_mapping_json
    logging.info(f"Loading image map: {image_file}")
    image_map = load_json(image_file)
    verify_image_paths(image_map)

    entries = image_map.get("entries", {})
    for entry in entries.values():
        if "clipped_meanfill" not in entry and "clipped_meanfill_auto" in entry:
            entry["clipped_meanfill"] = entry["clipped_meanfill_auto"]

    meta = {
        "aspect_ratio": image_map.get("aspect_ratio", {}),
        "clipped_meanfill": image_map.get("clipped_meanfill", {}) or image_map.get("clipped_meanfill_auto", {}),
    }
    return entries, meta


def load_manual_mask_map(cfg: Config) -> dict:
    """Load manual mask map and verify all referenced files exist."""
    manual_threshold = cfg.data_dir.joinpath(cfg.MANUAL_THRESHOLD_MAPPING_JSON)
    logging.info(f"Loading manual threshold mapping: {manual_threshold}")
    manual_mask_map = load_json(manual_threshold)
    verify_manual_mask_paths(manual_mask_map)
    return manual_mask_map


def load_data_sources(cfg: Config) -> DataSources:
    """Load all source files into a single in-memory NamedTuple."""
    image_entries, image_meta = load_image_map(cfg)
    return DataSources(
        identifiers_map=load_identifiers_map(cfg),
        image_entries=image_entries,
        image_meta=image_meta,
        metab_map=load_metabolite_map(cfg),
        survey_map=load_survey_map(cfg),
        manual_mask_map=load_manual_mask_map(cfg),
    )


def verify_image_paths(image_map: dict) -> None:
    """Raise if any processed/predicted/overlay path in the image map is missing."""
    for image_data in image_map.get("entries", {}).values():
        check_existence(Path(image_data["processed_image"]))
        check_existence(Path(image_data["predicted_mask_path"]))
        check_existence(Path(image_data["overlay_path"]))


def verify_manual_mask_paths(manual_mask_map: dict) -> None:
    """Raise if any best-Z / mask / original-mask path in the manual map is missing."""
    for manual_data in manual_mask_map.values():
        check_existence(Path(manual_data["Best Z Filename"]))
        check_existence(Path(manual_data["manual_mask_path"]))
        mask_path_orig = manual_data["manual_mask_path_original"]
        if mask_path_orig:
            check_existence(Path(mask_path_orig))
