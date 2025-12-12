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
EXPECTED_NUM_LABELS = 301
LABEL_MAP = {"Accepted": 1, "Not Accepted": 0, "Acceptable": 1, "Not Acceptable": 0}
LABEL_MAP_VALUES = {"Accepted": "Acceptable", "Not Accepted": "Not Acceptable"}

# ---------- helpers ----------
@dataclasses.dataclass
class Config:
    data_dir: Path = dataclasses.field(metadata={
        "help": "Path to data directory containing organoid data"
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
    IDENTIFIERS_MAP_JSON: typing.ClassVar[str] = f"{IDENTIFIERS_DIR}/main_identifiers.json"
    ALL_DATA_JSON: typing.ClassVar[str] = f"{IDENTIFIERS_DIR}/all_data.json"
    SUMMARY_JSON: typing.ClassVar[str] = f"{IDENTIFIERS_DIR}/summary.json"
    IMAGE_CLASSIFIER: typing.ClassVar[str] = f"{IDENTIFIERS_DIR}/image_classifier.json"
    SURVEY_CLASSIFIER: typing.ClassVar[str] = f"{IDENTIFIERS_DIR}/survey_classifier.json"

    IMAGES_DIR: typing.ClassVar[str] = "images"
    IMAGES_RAW_DIR: typing.ClassVar[str] = f"{IMAGES_DIR}/raw_images"
    PREPROCESSED_DIR: typing.ClassVar[str] = f"{IMAGES_DIR}/preprocessed_json"
    ORIGINAL_MAPPING_JSON: typing.ClassVar[str] = f"{IMAGES_DIR}/image_mapping.json"
    MANUAL_THRESHOLD_MAPPING_JSON: typing.ClassVar[str] = f"{IMAGES_DIR}/image_mapping_thresholded_and_manual.json"

    MASKS_DIR: typing.ClassVar[str] = f"{IMAGES_DIR}/masks"
    MASKS_PREDICTED_DIR: typing.ClassVar[str] = f"{MASKS_DIR}/predicted"
    MASKS_MANUAL_DIR: typing.ClassVar[str] = f"{MASKS_DIR}/manual"
    MASKS_OVERLAYS_DIR: typing.ClassVar[str] = f"{MASKS_DIR}/image_overlays"

    METABOLITE_MAP_JSON: typing.ClassVar[str] = "metabolite/metabolite_map.json"
    SURVEY_AGGREGATED_JSON: typing.ClassVar[str] = "survey/survey_map.json"

    def __post_init__(self):
        # Basic validation / normalization
        if not self.data_dir.exists():
            raise RuntimeError(f"{self.data_dir} does not exist")
        # Set up
        self.infer_resized_dir = f"{self.IMAGES_DIR}/infer_resized_{self.target_width}x{self.target_height}"

class DataSources(typing.NamedTuple):
    """Class to capture input data sources."""
    identifiers_map: dict
    base_map: dict
    metab_map: dict
    survey_map: dict
    manual_mask_map: dict
    processed_map: dict
    preprocessed_map: dict

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
    identifiers_file = cfg.data_dir.joinpath(cfg.IDENTIFIERS_MAP_JSON)
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

def load_base_map(cfg: Config) -> dict:
    """Load base image mapping from JSON file."""
    original_mapping = cfg.data_dir.joinpath(cfg.ORIGINAL_MAPPING_JSON)
    logging.info(f"Loading base mapping: {original_mapping}")
    base_json = load_json(original_mapping)
    return base_json.get("entries", {})

def load_manual_mask_map(cfg: Config) -> dict:
    """Load manual mask map and normalize keys."""
    manual_threshold = cfg.data_dir.joinpath(cfg.MANUAL_THRESHOLD_MAPPING_JSON)
    logging.info(f"Loading manual threshold mapping: {manual_threshold} and normalizing keys...")
    manual_mask_map = load_json(manual_threshold)
    return normalize_manual_mask_map(manual_mask_map, cfg.data_dir, cfg.IMAGES_RAW_DIR, cfg.MASKS_MANUAL_DIR)

def load_processed_map(cfg: Config) -> dict:
    """Load processed files JSON data."""
    logging.info("Loading processed files JSON data...")
    infer_resized_dir = cfg.data_dir.joinpath(cfg.infer_resized_dir)
    found_files = list(infer_resized_dir.rglob("image_mapping*_processed.json"))
    logging.info(f"Located {len(found_files)} processed files in {infer_resized_dir}")
    return build_processed_files_map(found_files, cfg.data_dir, cfg.infer_resized_dir, cfg.MASKS_PREDICTED_DIR)

def load_preprocessed_map(cfg: Config) -> dict:
    """Load preprocessed files JSON data."""
    logging.info("Loading preprocessed files JSON data...")
    preprocessed_files_dir = cfg.data_dir.joinpath(cfg.PREPROCESSED_DIR)
    preprocessed_files = list(preprocessed_files_dir.rglob("*"))
    logging.info(f"Located {len(preprocessed_files)} preprocessed files in {preprocessed_files_dir}")
    return build_preprocessed_map(preprocessed_files, cfg.data_dir, cfg.infer_resized_dir, cfg.MASKS_PREDICTED_DIR, cfg.MASKS_OVERLAYS_DIR)

def load_data_sources(cfg: Config) -> DataSources:
    """Load all data sources and return NamedTuple with source data in memory.

    Args:
        cfg: Configuration object

    Returns:
        DataSources object containing all data sources
    """
    return DataSources(
        identifiers_map=load_identifiers_map(cfg),
        base_map=load_base_map(cfg),
        metab_map=load_metabolite_map(cfg),
        survey_map=load_survey_map(cfg),
        manual_mask_map=load_manual_mask_map(cfg),
        processed_map=load_processed_map(cfg),
        preprocessed_map=load_preprocessed_map(cfg),
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

def normalize_manual_mask_map(manual_mask_map, data_dir, images_raw_dir, masks_manual_dir):
    """Normalize keys for storage of manual mask data and update path to data files.

    Args:
        manual_mask_map: Dictionary of manual mask data
        data_dir: Base input directory
        images_raw_dir: Subdirectory for raw images (e.g., "images/raw_images")
        masks_manual_dir: Subdirectory for manual masks (e.g., "masks/manual")
    """
    manual_mask_normalized = {}
    for raw_key, manual_data in manual_mask_map.items():
        try:
            norm_key = OrganoidNormalizer.normalize_key(raw_key)
        except ValueError:
            norm_key = OrganoidNormalizer.clean_string(raw_key).upper()

        best_z = data_dir.joinpath(images_raw_dir, Path(manual_data["Best Z Filename"]).name)
        check_existence(best_z)
        manual_data["Best Z Filename"] = str(best_z)

        mask_path = data_dir.joinpath(masks_manual_dir, Path(manual_data["MT Mask Path"]).name)
        check_existence(mask_path)
        manual_data["MT Mask Path"] = str(mask_path)

        manual_mask_normalized[norm_key] = manual_data

    return manual_mask_normalized

def check_existence(file_path):
    """Check existence of file and raise an error if it does not exist."""
    if not file_path.exists():
        raise RuntimeError(f"Required file does not exist: {file_path}")

def build_processed_files_map(found_files, data_dir, infer_resized_dir, masks_predicted_dir):
    """Build and return a dictionary of processed file JSON data.

    Also update hardcoded paths to point to input files on the file system.

    Args:
        found_files: List of processed JSON files to load
        data_dir: Base input directory
        infer_resized_dir: Subdirectory for resized images (e.g., "images/infer_resized_512x384")
        masks_predicted_dir: Subdirectory for predicted masks (e.g., "masks/predicted")
    """
    processed_map = {}
    for p in found_files:
        raw = load_json(p)
        for batch_data in raw.values():
            img_path = data_dir.joinpath(infer_resized_dir, Path(batch_data["img_path"]).name)
            check_existence(img_path)
            batch_data["img_path"] = str(img_path)

            mask_path = data_dir.joinpath(masks_predicted_dir, Path(batch_data["mask_path"]).name)
            check_existence(mask_path)
            batch_data["mask_path"] = str(mask_path)

        processed_map.update(raw)

    return processed_map

def build_preprocessed_map(files, data_dir, infer_resized_dir, masks_predicted_dir, masks_overlays_dir):
    """Build and return a dictionary of preprocessed JSON data.

    Args:
        files: List of preprocessed JSON files to load
        data_dir: Base input directory
        infer_resized_dir: Subdirectory for resized images (e.g., "images/infer_resized_512x384")
        masks_predicted_dir: Subdirectory for predicted masks (e.g., "masks/predicted")
        masks_overlays_dir: Subdirectory for image overlays (e.g., "masks/image_overlays")
    """
    preprocessed_map = {}
    for file in files:
        raw = load_json(file)
        for batch_data in raw:
            img_path = data_dir.joinpath(infer_resized_dir, Path(batch_data["img_path"]).name)
            check_existence(img_path)
            batch_data["img_path"] = str(img_path)

            mask_path = data_dir.joinpath(masks_predicted_dir, Path(batch_data["mask_path"]).name)
            check_existence(mask_path)
            batch_data["mask_path"] = str(mask_path)

            overlay_path = data_dir.joinpath(masks_overlays_dir, Path(batch_data["overlay_path"]).name)
            check_existence(overlay_path)
            batch_data["overlay_path"] = str(overlay_path)

            main_id = batch_data["metadata_key"]
            preprocessed_map[main_id] = batch_data

    return preprocessed_map

def merge_data_sources(sources: DataSources) -> tuple[dict, dict]:
    """Merge and return dictionary of all data sources plus number of masks.

    Args:
        sources: DataSources object containing all data sources

    Returns:
        tuple[dict, dict]: Combined dictionary of all data sources and stats dictionary
    """
    combined = {}
    for key, original_day in tqdm(sources.identifiers_map.items(), desc="Merging data sources"):
        # Match base image info
        entry = dict(sources.base_map[key])
        main_id = entry.get("main_id", "")
        if 'dayID' in entry:    # Extract numerical day from dayID
            entry['mdl_day'] = extract_mdl_day(entry['dayID'])
        entry['original_day'] = original_day

        # Match processed info
        processed = sources.processed_map.get(key)
        if processed:
            entry["processed"] = processed
            entry["main_id"] = main_id

        # Match preprocessed info
        preprocessed = sources.preprocessed_map.get(key) or sources.preprocessed_map.get(key)
        if preprocessed:
            entry["preprocessed"] = preprocessed
        else:
            entry["preprocessed"] = {}

        # Match survey info
        label = {}
        if key in sources.survey_map:
            entry["survey"] = sources.survey_map[key]
            label = entry["survey"].get("label", {})
            entry["survey"].pop("label")
        else:
            entry["survey"] = {}

        # Store label info
        if label:
            entry["label"] = label
        elif preprocessed:
            value = preprocessed.get("label", {})
            entry["label"] = {
                "value": LABEL_MAP_VALUES.get(value),
                "acceptance_flag": LABEL_MAP.get(value),
                "source": "preprocessed.label",
            }
        else:
            entry["label"] = {
                "value": None,
                "acceptance_flag": None,
                "source": None,
            }

        # Add metabolites
        if key in sources.metab_map:
            entry["metabolites"] = sources.metab_map[key]

        # Add manual mask path
        if key in sources.manual_mask_map:
            manual_data = sources.manual_mask_map[key]
            entry["manual_mask_path"] = manual_data.get("MT Mask Path")

        combined[key] = entry

    return combined

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

    records = { source_id.replace(" ", "_"): builder.build(source_id, entry) for source_id, entry in combined.items() }
    stats = builder.record_metrics.to_dict()

    logging.info("Sanitizing data for JSON...")
    records_dict = { source_id: record.to_dict() for source_id, record in records.items() }
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

        assert stats['num_records'] == stats['num_labels'] + stats['num_no_labels']
        assert stats['num_labels'] == stats['num_preprocessed_labels'] + stats['num_survey_labels']

        assert stats['num_records'] == stats['num_metabolites'] + stats['num_no_metabolite']

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

    # Records and images
    logging.info(f"  Image paths: {stats['num_img_paths']:,}")
    logging.info(f"  Image splits: {stats['num_img_split']:,} | Stitched: {stats['num_img_stitched']:,} | No image label: {stats['num_img_no_label']:,}")

    # Labels
    logging.info(f"  Labels: {stats['num_labels']:,} | No labels: {stats['num_no_labels']:,}  | Preprocessed: {stats['num_preprocessed_labels']:,} | Survey: {stats['num_survey_labels']:,}")

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
