#!/usr/bin/env python3
# Standard
import argparse
import dataclasses
import datetime
import json
import logging
import math
import re
import typing
from pathlib import Path

from rich.logging import RichHandler
from tqdm import tqdm

# Application
from file_utils.common.organoid_patterns import OrganoidNormalizer
from file_utils.merge.normalized_records import (
    RecordMetrics,
    OrganoidRecordBuilder,
)
from file_utils.merge.validate_schema import validate_all_data_json

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO,
                    handlers=[RichHandler()])

# Constants
EXPECTED_TOTAL_RECORDS = 5168
EXPECTED_NUM_MANUAL_MASKS = 2153
EXPECTED_NUM_METABOLITES = 4154
EXPECTED_NUM_LABELS = 301

# ---------- helpers ----------
@dataclasses.dataclass
class Config:
    data_dir: Path = dataclasses.field(metadata={
        "help": "Path to data directory containing organoid data"
    })
    image_mapping_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to image mapping JSON file created with preprocessing pipeline"
    })
    identifiers_map_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to identifiers map JSON file"
    })
    min_survey_votes: int = dataclasses.field(default=4, metadata={
        "help": "Minimum number of votes required to indicate acceptable" \
                + "or not acceptable survey results"
    })
    survey_day: int = dataclasses.field(default=30, metadata={
        "help": "Day that survey was conducted"
    })
    target_width: int = dataclasses.field(default=512, metadata={
        "help": "Target input image width (pixels)"
    })
    target_height: int = dataclasses.field(default=384, metadata={
        "help": "Target input image height (pixels)"
    })
    no_validate: bool = dataclasses.field(default=False, metadata={
        "help": "Validate schema of generated all_data.json file"
    })

    # Directory structure constants (relative to data_dir)
    IDENTIFIERS_DIR: typing.ClassVar[str] = "identifiers"
    ALL_DATA_JSON: typing.ClassVar[str] = f"{IDENTIFIERS_DIR}/all_data.json"
    SUMMARY_JSON: typing.ClassVar[str] = f"{IDENTIFIERS_DIR}/summary.json"

    MASKS_DIR: typing.ClassVar[str] = f"masks"
    MANUAL_THRESHOLD_MAPPING_JSON: typing.ClassVar[str] = f"{MASKS_DIR}/image_mapping_thresholded_and_manual.json"

    METABOLITE_MAP_JSON: typing.ClassVar[str] = "metabolite/metabolite_map.json"
    SURVEY_AGGREGATED_JSON: typing.ClassVar[str] = "survey/survey_map.json"

    def __post_init__(self):
        # Basic validation / normalization
        if not self.data_dir.exists():
            raise RuntimeError(f"{self.data_dir} does not exist")
        if not self.image_mapping_json.exists():
            raise RuntimeError(f"{self.image_mapping_json} does not exist")
        # Set up
        if self.identifiers_map_json is None:
            self.identifiers_map_json = self.data_dir / self.IDENTIFIERS_DIR / "record_identifiers.json"

class DataSources(typing.NamedTuple):
    """Class to capture input data sources."""
    identifiers_map: dict
    image_map: dict
    metab_map: dict
    survey_map: dict
    manual_mask_map: dict

def get_args():
    arg_parser = create_args()
    args = arg_parser.parse_args()
    for key,val in vars(args).items():
        logging.info(f"{key}: {val}")
    cfg = Config(**vars(args))
    return cfg

def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Run image classifier on organoid images")

    for field in dataclasses.fields(Config):
        # Build argument flag and help message
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {
            "help": field.metadata.get("help", ""),
            "default": field.default
        }

        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type
        parser.add_argument(*flags, **kwargs)

    return parser

def load_identifiers_map(cfg: Config) -> dict:
    """Load identifiers map from JSON file."""
    identifiers_file = cfg.identifiers_map_json
    logging.info(f"Loading identifiers map: {identifiers_file}")
    return load_json(identifiers_file)

def load_metabolite_map(cfg: Config) -> dict:
    """Load metabolite map from JSON file."""
    metabolite_file = cfg.data_dir.joinpath(cfg.METABOLITE_MAP_JSON)
    logging.info(f"Loading metabolite map: {metabolite_file}")
    return load_json(metabolite_file)

def load_survey_map(cfg: Config) -> dict:
    """Load survey data and build survey map keyed by (main_id, split_index)."""
    survey_file = cfg.data_dir.joinpath(cfg.SURVEY_AGGREGATED_JSON)
    logging.info(f"Loading survey data: {survey_file} and building survey map...")
    return load_json(survey_file)

def load_image_map(cfg: Config) -> dict:
    """Load base image mapping from JSON file."""
    image_file = cfg.image_mapping_json
    logging.info(f"Loading image map: {image_file}")
    image_map = load_json(image_file)
    verify_image_paths(image_map)
    return image_map.get("entries", {})

def load_manual_mask_map(cfg: Config) -> dict:
    """Load manual mask map and normalize keys."""
    manual_threshold = cfg.data_dir.joinpath(cfg.MANUAL_THRESHOLD_MAPPING_JSON)
    logging.info(f"Loading manual threshold mapping: {manual_threshold}")
    manual_mask_map = load_json(manual_threshold)
    verify_manual_mask_paths(manual_mask_map)
    return manual_mask_map

def load_data_sources(cfg: Config) -> DataSources:
    """Load all data sources and return NamedTuple with source data in memory.

    Args:
        cfg: Configuration object

    Returns:
        DataSources object containing all data sources
    """
    return DataSources(
        identifiers_map=load_identifiers_map(cfg),
        image_map=load_image_map(cfg),
        metab_map=load_metabolite_map(cfg),
        survey_map=load_survey_map(cfg),
        manual_mask_map=load_manual_mask_map(cfg),
    )

def load_json(path: Path | str):
    """Load JSON file and return dictionary.
    Args:
        path: Path to JSON file

    Returns:
        Dictionary of JSON data
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Required JSON file does not exist: {path}")
    with path.open("r") as f:
        return json.load(f)

def verify_image_paths(image_map: dict) -> dict:
    """Verify image paths exist and raise an error if they do not exist.

    Args:
        image_map: Dictionary of image data
    """
    for image_data in image_map.get("entries", {}).values():
        img_path = Path(image_data["processed_image"])
        check_existence(img_path)

        mask_path = Path(image_data["predicted_mask_path"])
        check_existence(mask_path)

        overlay_path = Path(image_data["overlay_path"])
        check_existence(overlay_path)

def verify_manual_mask_paths(manual_mask_map: dict) -> dict:
    """Verify existence of manual mask data files.

    Args:
        manual_mask_map: Dictionary of manual mask data
    """
    for manual_data in manual_mask_map.values():
        best_z = Path(manual_data["Best Z Filename"])
        check_existence(best_z)

        mask_path = Path(manual_data["manual_mask_path"])
        check_existence(mask_path)

        mask_path_orig = manual_data["manual_mask_path_original"]
        if mask_path_orig:
            mask_path_orig = Path(mask_path_orig)
            check_existence(mask_path_orig)

def check_existence(file_path):
    """Check existence of file and raise an error if it does not exist."""
    if not file_path.exists():
        raise RuntimeError(f"Required file does not exist: {file_path}")

def merge_data_sources(sources: DataSources) -> tuple[dict, dict]:
    """Merge and return dictionary of all data sources plus number of masks.

    Args:
        sources: DataSources object containing all data sources

    Returns:
        tuple[dict, dict]: Combined dictionary of all data sources and stats dictionary
    """
    combined = {}
    for key, original_day in tqdm(sources.identifiers_map.items(), desc="Merging data sources"):
        # Match image mapping info
        entry = dict(sources.image_map.get(key, {}))

        # Dates
        if 'dayID' in entry:    # Extract numerical day from dayID
            entry['mdl_day'] = extract_mdl_day(entry['dayID'])
        entry['original_day'] = original_day

        # Match survey info
        label = {}
        if key in sources.survey_map:
            entry["survey"] = sources.survey_map.get(key, {})
            label = entry["survey"].pop("label", {})
        else:
            entry["survey"] = {}

        # Store label info
        entry["label"] = label

        # Add metabolites
        if key in sources.metab_map:
            entry["metabolite"] = sources.metab_map.get(key, {})

        # Add manual mask path
        if key in sources.manual_mask_map:
            manual_data = sources.manual_mask_map.get(key, {})
            entry["manual_mask_path"] = manual_data.get("manual_mask_path")
            entry["manual_mask_path_original"] = manual_data.get("manual_mask_path_original")

        combined[key] = entry

    return combined

def normalize_day_in_key(key: str) -> str:
    """Normalize day identifiers in keys (Dy20/Dy21 -> Dy20.5)."""
    if not key:
        return key
    # Replace Dy20 and Dy21 with Dy20.5 in the key
    # Match "Dy20" or "Dy21" but not "Dy20.5" or "Dy20." (already normalized)
    key = re.sub(r'\bDy20\b(?!\.)', 'Dy20.5', key)
    key = re.sub(r'\bDy21\b(?!\.)', 'Dy20.5', key)
    return key

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

def build_normalized_records(cfg, combined):
    """Build normalized records and stats.

    Args:
        cfg: Configuration object
        combined: Dictionary of combined data sources

    Returns:
        tuple[dict, dict]: Dictionary of normalized records and stats
    """
    builder = OrganoidRecordBuilder(
        min_survey_votes=cfg.min_survey_votes,
        survey_day=cfg.survey_day,
        target_size=(cfg.target_width, cfg.target_height),
        record_metrics = RecordMetrics()
    )

    records = { source_id: builder.build(source_id, entry) for source_id, entry in combined.items() }
    stats = builder.record_metrics.to_dict()

    # Propograte labels for day 30 organoids to previous days organoids
    logging.info("Propogating labels for day 30 organoids to previous days organoids...")
    records_dict = { source_id: record.to_dict() for source_id, record in records.items() }
    label_stats =propogate_labels(records_dict, builder.organoid_dict)
    stats.update(label_stats)

    # Get final number of organoids
    stats["num_organoids"] = get_num_organoids(records_dict)

    logging.info("Sanitizing data for JSON...")
    records_clean = sanitize_for_json(records_dict)
    stats_clean = sanitize_for_json(stats)

    # Validate the records before writing (in-memory validation)
    if not cfg.no_validate:
        if not validate_data(stats_clean):
            raise RuntimeError("Data validation failed")
        valid, stats_validation = validate_json(records_clean)
        if not valid:
            raise RuntimeError("Schema validation failed")
        stats_clean.update(stats_validation)

    # Write the records and stats to JSON files
    out_file = cfg.data_dir.joinpath(cfg.ALL_DATA_JSON)
    write_json(out_file, records_clean)
    out_file = cfg.data_dir.joinpath(cfg.SUMMARY_JSON)
    write_json(out_file, stats_clean)

    return records, stats_clean

def propogate_labels(records_dict: dict, organoid_dict: dict) -> dict:
    """Propogate labels for day 30 organoids to previous days organoids."""
    stats = {
        "num_labels": 0,
        "num_no_labels": 0,
    }
    for record_data in records_dict.values():
        organoid_id = record_data["organoid_id"]
        if organoid_id in organoid_dict:
            record_data["label"] = organoid_dict[organoid_id]["label"]
            stats["num_labels"] += 1
        else:
            record_data["label"] = {}
            stats["num_no_labels"] += 1
    return stats

def get_num_organoids(records_dict: dict) -> int:
    """Get the number of organoids in the records dictionary."""
    organoid_dict: dict[str, int] = {}
    for record_data in records_dict.values():
        organoid_id = record_data["organoid_id"]
        if organoid_id in organoid_dict:
            continue
        else:
            organoid_dict[organoid_id] = 1
    return len(organoid_dict.keys())

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

def validate_data(stats: dict) -> bool:
    """Validate the data before writing."""
    logging.info("Validating data before writing...")
    try:
        assert stats['num_records'] == EXPECTED_TOTAL_RECORDS
        assert stats['num_img_paths'] == EXPECTED_TOTAL_RECORDS
        assert stats['num_mask_paths'] == EXPECTED_TOTAL_RECORDS
        assert stats['num_overlay_paths'] == EXPECTED_TOTAL_RECORDS
        assert stats['num_manual_masks'] == EXPECTED_NUM_MANUAL_MASKS
        assert stats['num_records'] == stats['num_labels'] + stats['num_no_labels']
        assert stats['num_records'] == stats['num_metabolites'] + stats['num_no_metabolite']
        assert stats['num_metabolites'] == EXPECTED_NUM_METABOLITES
        assert stats['num_records'] == stats['num_survey'] + stats['num_no_survey']
        assert stats['total_votes'] == stats['num_acceptable_votes'] + stats['num_not_acceptable_votes']
        assert stats['num_survey'] == stats['num_majority'] + stats['num_no_majority']

        logging.info(f"Data validation passed")
        return True
    except AssertionError as e:
        logging.exception(f"Data validation failed with exception: {e}")
        return False

def validate_json(records: dict) -> bool:
    """Validate the schema of the records before writing."""
    logging.info("Validating schema of records before writing...")
    try:
        validation_results = validate_all_data_json(data=records, strict=True)

        if validation_results['valid']:
            valid = True
            logging.info("Schema validation passed")

        else:
            valid = False
            error_count = len(validation_results['errors'])
            warning_count = len(validation_results['warnings'])
            logging.warning(f"Schema validation found {error_count} errors and {warning_count} warnings")

            # Log first few errors
            for error in validation_results['errors'][:5]:
                logging.warning(f"  - {error}")
            if error_count > 5:
                logging.warning(f"  ... and {error_count - 5} more errors")

    except Exception as e:
        valid = False
        logging.exception(f"Schema validation failed with exception: {e}")

    return valid, validation_results["stats"]

def write_json(out_file, payload):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Wrote :{out_file}")

def print_stats(stats, out_file, no_validate: bool):
    """Print statistics in a clean, organized format."""
    logging.info("Record stats:")
    logging.info(f"  Wrote {stats['num_records']:,} merged records → {out_file}")
    logging.info(f"  Organoids: {stats['num_organoids']:,}")

    # Records and images
    logging.info(f"  Image paths: {stats['num_img_paths']:,}")
    logging.info(f"  Image splits: {stats['num_img_split']:,} | Stitched: {stats['num_img_stitched']:,}")

    # Labels
    logging.info(f"  Labels: {stats['num_labels']:,} | No labels: {stats['num_no_labels']:,}")

    # Surveys
    logging.info(f"  Survey matches: {stats['num_survey']:,} | No survey: {stats['num_no_survey']:,}")
    logging.info(f"  Acceptable votes: {stats['num_acceptable_votes']:,} | Not acceptable: {stats['num_not_acceptable_votes']:,}")
    logging.info(f"  Majority: {stats['num_majority']:,} | No majority: {stats['num_no_majority']:,} | Total votes: {stats['total_votes']:,}")

    # Metabolites
    logging.info(f"  Metabolites: {stats['num_metabolites']:,} | No metabolite: {stats['num_no_metabolite']:,}")
    logging.info(f"  Metabolite outliers: {stats['num_metabolite_outliers']:,}")

    # Manual masks
    logging.info(f"  Manual masks: {stats['num_manual_masks']:,}")

    # Validation stats (only if validation was run)
    if not no_validate:
        logging.info("Validation stats:")
        logging.info(f"  Records with required fields: {stats.get('records_with_required_fields', 0):,}")
        logging.info(f"  Records with images: {stats.get('records_with_images', 0):,}")
        logging.info(f"  Records with metabolites: {stats.get('records_with_metabolites', 0):,}")
        logging.info(f"  Records with survey: {stats.get('records_with_survey', 0):,}")

        # Day distribution (top 5)
        day_dist = stats.get('day_distribution', {})
        if day_dist:
            logging.info(f"  Day distribution (top 5):")
            sorted_days = sorted(day_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            for day, count in sorted_days:
                logging.info(f"    {day}: {count:,}")

        # Classification verification distribution
        class_verif_dist = stats.get('classification_verification_distribution', {})
        if class_verif_dist:
            logging.info(f"  Classification verification distribution:")
            for verif, count in sorted(class_verif_dist.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"    {verif}: {count:,}")

def main():
    # ---------- command line arguments ----------
    cfg = get_args()

    # ---------- load sources ----------
    sources = load_data_sources(cfg)

    # ---------- merge ----------
    logging.info("Merging data sources...")
    combined = merge_data_sources(sources)

    # ---------- normalize ----------
    logging.info("Normalizing merged records...")
    _, stats = build_normalized_records(cfg, combined)

    # ----------  print top-level data stats ----------
    print_stats(stats, cfg.ALL_DATA_JSON, cfg.no_validate)

if __name__ == "__main__":
    main()
