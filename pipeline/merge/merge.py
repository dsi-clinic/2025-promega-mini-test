#!/usr/bin/env python3
"""Core merge + normalization logic for step 16."""

import json
import logging
import math
import re

from tqdm import tqdm

from pipeline.merge.cli import Config
from pipeline.merge.loaders import DataSources
from pipeline.merge.normalized_records import OrganoidRecordBuilder, RecordMetrics
from pipeline.merge.validation import validate_data, validate_json


def extract_mdl_day(day_id: str) -> float:
    """Extract numerical day from a dayID string.

    Examples: 'Dy17' → 17.0; 'Dy20' or 'Dy21' → 20.5; '' → None.
    """
    if not day_id:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", day_id)
    if match:
        day_num = float(match.group(1))
        if day_num in (20.0, 21.0):
            return 20.5
        return day_num
    return None


def merge_data_sources(sources: DataSources) -> dict:
    """Combine identifiers, images, surveys, metabolites, and manual masks into one dict."""
    combined = {}
    for key, original_day in tqdm(sources.identifiers_map.items(), desc="Merging data sources"):
        entry = dict(sources.image_entries.get(key, {}))

        if "dayID" in entry:
            entry["mdl_day"] = extract_mdl_day(entry["dayID"])
        entry["original_day"] = original_day

        label = {}
        if key in sources.survey_map:
            entry["survey"] = sources.survey_map.get(key, {})
            label = entry["survey"].pop("label", {})
        else:
            entry["survey"] = {}
        entry["label"] = label

        if key in sources.metab_map:
            entry["metabolite"] = sources.metab_map.get(key, {})

        if key in sources.manual_mask_map:
            manual_data = sources.manual_mask_map.get(key, {})
            entry["manual_mask_path"] = manual_data.get("manual_mask_path")
            entry["manual_mask_path_original"] = manual_data.get("manual_mask_path_original")

        combined[key] = entry
    return combined


def propagate_labels(records_dict: dict, organoid_dict: dict) -> dict:
    """Propagate survey-day labels to all other days for the same organoid.

    Records that already have a direct survey label (e.g. Day 28/30 records)
    are left unchanged. Only unlabeled records receive the propagated value.
    """
    stats = {"num_labels": 0, "num_no_labels": 0}
    for record_data in records_dict.values():
        if record_data.get("label"):
            stats["num_labels"] += 1
            continue
        organoid_id = record_data["organoid_id"]
        if organoid_id in organoid_dict:
            record_data["label"] = organoid_dict[organoid_id]["label"]
            stats["num_labels"] += 1
        else:
            record_data["label"] = {}
            stats["num_no_labels"] += 1
    return stats


def clear_conflicted_labels(records_dict: dict, conflicted_organoids: set) -> None:
    """Wipe labels from every record belonging to a conflict-flagged organoid.

    If two splits of the same organoid have differing survey-day labels, no
    label is propagated to any timepoint for that organoid.
    """
    for record_data in records_dict.values():
        if record_data["organoid_id"] in conflicted_organoids:
            record_data["label"] = {}


def get_num_organoids(records_dict: dict) -> int:
    """Count distinct organoids represented in the records."""
    return len({record["organoid_id"] for record in records_dict.values()})


def sanitize_for_json(obj):
    """Recursively make a value JSON-safe (NaN/inf → None; pandas NA → None)."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    try:
        if hasattr(obj, "isna") and obj.isna():
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def build_normalized_records(cfg: Config, combined: dict, image_meta: dict):
    """Run the OrganoidRecordBuilder, propagate labels, validate, and write output JSONs."""
    builder = OrganoidRecordBuilder(
        min_survey_votes=cfg.min_survey_votes,
        target_size=(cfg.target_width, cfg.target_height),
        record_metrics=RecordMetrics(),
    )

    records = {source_id: builder.build(source_id, entry) for source_id, entry in combined.items()}
    stats = builder.record_metrics.to_dict()

    logging.info("Propagating labels for day-30 organoids to previous days...")
    records_dict = {source_id: record.to_dict() for source_id, record in records.items()}
    stats.update(propagate_labels(records_dict, builder.organoid_dict))

    if builder.conflicted_organoids:
        logging.info(f"Clearing labels for {len(builder.conflicted_organoids)} organoids with split label conflicts...")
        clear_conflicted_labels(records_dict, builder.conflicted_organoids)

    stats["num_organoids"] = get_num_organoids(records_dict)

    logging.info("Sanitizing data for JSON...")
    records_clean = sanitize_for_json(records_dict)
    stats_clean = sanitize_for_json(stats)
    stats_clean["image_mapping_meta"] = sanitize_for_json(image_meta)

    if not cfg.no_validate:
        if not validate_data(stats_clean):
            raise RuntimeError("Data validation failed")
        valid, stats_validation = validate_json(records_clean)
        if not valid:
            raise RuntimeError("Schema validation failed")
        stats_clean.update(stats_validation)

    write_json(cfg.out_file, records_clean)
    write_json(cfg.summary_file, stats_clean)
    return records, stats_clean


def write_json(out_file, payload) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Wrote: {out_file}")


def print_stats(stats: dict, out_file, no_validate: bool) -> None:
    """Pretty-print a per-section summary of merge stats to the logger."""
    logging.info("Record stats:")
    logging.info(f"  Wrote {stats['num_records']:,} merged records → {out_file}")
    logging.info(f"  Organoids: {stats['num_organoids']:,}")
    logging.info(f"  Image paths: {stats['num_img_paths']:,}")
    logging.info(f"  Image splits: {stats['num_img_split']:,} | Stitched: {stats['num_img_stitched']:,}")
    logging.info(f"  Labels: {stats['num_labels']:,} | No labels: {stats['num_no_labels']:,}")
    logging.info(f"  Survey matches: {stats['num_survey']:,} | No survey: {stats['num_no_survey']:,}")
    logging.info(f"  Acceptable votes: {stats['num_acceptable_votes']:,} | "
                 f"Not acceptable: {stats['num_not_acceptable_votes']:,}")
    logging.info(f"  Majority: {stats['num_majority']:,} | No majority: {stats['num_no_majority']:,} | "
                 f"Total votes: {stats['total_votes']:,}")
    logging.info(f"  Metabolites: {stats['num_metabolites']:,} | "
                 f"No metabolite: {stats['num_no_metabolite']:,}")
    logging.info(f"  Metabolite outliers: {stats['num_metabolite_outliers']:,}")
    logging.info(f"  Manual masks: {stats['num_manual_masks']:,}")

    if no_validate:
        return
    logging.info("Validation stats:")
    logging.info(f"  Records with required fields: {stats.get('records_with_required_fields', 0):,}")
    logging.info(f"  Records with images: {stats.get('records_with_images', 0):,}")
    logging.info(f"  Records with metabolites: {stats.get('records_with_metabolites', 0):,}")
    logging.info(f"  Records with survey: {stats.get('records_with_survey', 0):,}")

    day_dist = stats.get("day_distribution", {})
    if day_dist:
        logging.info("  Day distribution (top 5):")
        for day, count in sorted(day_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            logging.info(f"    {day}: {count:,}")

    class_verif_dist = stats.get("classification_verification_distribution", {})
    if class_verif_dist:
        logging.info("  Classification verification distribution:")
        for verif, count in sorted(class_verif_dist.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"    {verif}: {count:,}")
