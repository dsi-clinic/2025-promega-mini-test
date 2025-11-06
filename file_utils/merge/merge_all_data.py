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


from tqdm import tqdm

# Application
from file_utils.common.organoid_patterns import OrganoidNormalizer
from file_utils.merge.normalized_records import (
    RecordMetrics,
    OrganoidRecordBuilder,
    ImageClassifierEmitter,
    SurveyClassifierEmitter,
    emit_views,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

# ---------- helpers ----------
@dataclasses.dataclass
class Config:
    in_dir: Path = dataclasses.field(metadata={
        "help": "Path to input directory containing organoid images"
    })
    out_dir: Path = dataclasses.field(metadata={
        "help": "Path to output directory where results will be saved"
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

    ORIGINAL_MAPPING_JSON: typing.ClassVar[str] = "image_mapping.json"
    MANUAL_THRESHOLD_MAPPING_JSON: typing.ClassVar[str] = "image_mapping_thresholded_and_manual.json"
    METABOLITE_MAP_JSON: typing.ClassVar[str] = "metabolite_map.json"
    SURVEY_AGGREGATED_JSON: typing.ClassVar[str] = "organoid_surveys_aggregated.json"
    ALL_DATA_JSON: typing.ClassVar[str] = "all_data.json"
    IMAGE_CLASSIFIER: typing.ClassVar[str] = "image_classifier.json"
    SURVEY_CLASSIFIER: typing.ClassVar[str] = "survey_classifier.json"
    SCHEMA_VERSION: typing.ClassVar[int] = 1

    def __post_init__(self):
        # Basic validation / normalization
        if not self.in_dir.exists():
            raise RuntimeError(f"{self.in_dir} does not exist")
        # Set up
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.infer_resized_dir = f"infer_resized_{self.target_width}x{self.target_height}"

class DataSources(typing.NamedTuple):
    """Class to capture input data sources."""
    base_map: dict
    metab_map: dict
    survey_json: dict
    manual_mask_map: dict
    found_files: list
    preprocessed_files: list

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

def load_data_sources(cfg):
    """Load data sources and return NamedTuple with source data in memory."""
    original_mapping = cfg.in_dir.joinpath("json", cfg.ORIGINAL_MAPPING_JSON)
    logging.info(f"Loading base mapping: {original_mapping}")
    base_json = load_json(original_mapping)
    base_map = base_json.get("entries", {})

    metabolite_map = cfg.in_dir.joinpath("json", cfg.METABOLITE_MAP_JSON)
    logging.info(f"Loading metabolite map: {metabolite_map}")
    metab_map = load_json(metabolite_map)

    survey_aggregated = cfg.in_dir.joinpath("json", cfg.SURVEY_AGGREGATED_JSON)
    logging.info(f"Loading survey data: {survey_aggregated}")
    survey_json = load_json(survey_aggregated)

    manual_threshold = cfg.in_dir.joinpath("json", cfg.MANUAL_THRESHOLD_MAPPING_JSON)
    logging.info(f"Loading manual threshold mapping: {manual_threshold}")
    manual_mask_map = load_json(manual_threshold)

    infer_resized_dir = cfg.in_dir.joinpath("images", cfg.infer_resized_dir)
    found_files = list(infer_resized_dir.rglob("image_mapping*_processed.json"))
    logging.info(f"Located {len(found_files)} processed files in {infer_resized_dir}")

    preprocessed_files_dir = cfg.in_dir.joinpath("json", "preprocessed")
    preprocessed_files = list(preprocessed_files_dir.rglob("*"))
    logging.info(f"Located {len(preprocessed_files)} preprocessed files in {preprocessed_files_dir}")

    return DataSources(
        base_map=base_map,
        metab_map=metab_map,
        survey_json=survey_json,
        manual_mask_map=manual_mask_map,
        found_files=found_files,
        preprocessed_files=preprocessed_files
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

        best_z = in_dir.joinpath("images", "raw_images", Path(manual_data["Best Z Filename"]).name)
        check_existence(best_z)
        manual_data["Best Z Filename"] = str(best_z)

        mask_path = in_dir.joinpath("masks", "manual", Path(manual_data["MT Mask Path"]).name)
        check_existence(mask_path)
        manual_data["MT Mask Path"] = str(mask_path)

        manual_mask_normalized[norm_key] = manual_data

    return manual_mask_normalized

def check_existence(file_path):
    """Check existence of file and raise an error if it does not exist."""
    if not file_path.exists():
        raise RuntimeError(f"Required file does not exist: {file_path}")

def build_processed_files_map(found_files, in_dir, infer_resized_dir):
    """Build and return a dictionary of processed file JSON data.

    Also update hardcoded paths to point to input files on the file system.
    """
    processed_map = {}
    for p in found_files:
        raw = load_json(p)
        for batch_data in raw.values():
            img_path = in_dir.joinpath("images", infer_resized_dir, Path(batch_data["img_path"]).name)
            check_existence(img_path)
            batch_data["img_path"] = str(img_path)

            mask_path = in_dir.joinpath("masks", "predicted", Path(batch_data["mask_path"]).name)
            check_existence(mask_path)
            batch_data["mask_path"] = str(mask_path)

        processed_map.update(raw)

    return processed_map

def build_preprocessed_map(files, in_dir, infer_resized_dir):
    """Build and return a dictionary of preprocessed JSON data."""
    preprocessed_map = {}
    for file in files:
        raw = load_json(file)
        for batch_data in raw:
            img_path = in_dir.joinpath("images", infer_resized_dir, Path(batch_data["img_path"]).name)
            check_existence(img_path)
            batch_data["img_path"] = str(img_path)

            mask_path = in_dir.joinpath("masks", "predicted", Path(batch_data["mask_path"]).name)
            check_existence(mask_path)
            batch_data["mask_path"] = str(mask_path)

            overlay_path = in_dir.joinpath("masks", "image_overlays", Path(batch_data["overlay_path"]).name)
            check_existence(overlay_path)
            batch_data["overlay_path"] = str(overlay_path)

            main_id = batch_data["id"].replace("DY", "Dy")
            preprocessed_map[main_id] = batch_data

    return preprocessed_map

def merge_data_sources(base_map, survey_map, metab_map, manual_mask_normalized,
                       processed_map, preprocessed_map):
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

        # Match preprocessed info
        preprocessed = preprocessed_map.get(raw_k) or preprocessed_map.get(normalized_parent_key(raw_k))
        if preprocessed:
            entry["preprocessed"] = preprocessed

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

def build_normalized_records(cfg, survey_matched_count, survey_not_matched_count, manual_mask_count, combined):
    builder = OrganoidRecordBuilder(
        min_survey_votes=cfg.min_survey_votes,
        survey_day=cfg.survey_day,
        target_size=(cfg.target_width, cfg.target_height),
        record_metrics = RecordMetrics()
    )
    records = { source_id.replace(" ", "_"): builder.build(source_id, entry) for source_id, entry in combined.items() }
    payload = {
        "schema_version": cfg.SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stats": {
            "total_records": len(records),
            "survey_matches": survey_matched_count,
            "survey_not_matched": survey_not_matched_count,
            "manual_masks": manual_mask_count,
        },
        "records": { source_id: record.to_dict() for source_id, record in records.items() },
    }
    payload["stats"].update(builder.record_metrics.to_dict())

    # ---------- sanitize and write output for all data ----------
    logging.info("Sanitizing data for JSON...")
    payload_clean = sanitize_for_json(payload)

    out_file = cfg.out_dir.joinpath("json", cfg.ALL_DATA_JSON)
    write_json(out_file, payload_clean)

    return records, payload["stats"]

def generate_views(records, cfg):
    """Generate image and survey classifier views on the data."""
    views = emit_views(
        records,
        [
            ImageClassifierEmitter(),
            SurveyClassifierEmitter(survey_day=cfg.survey_day, min_votes=cfg.min_survey_votes),
        ],
    )
    payload_clean = sanitize_for_json(views)

    out_file = cfg.out_dir.joinpath("json", cfg.IMAGE_CLASSIFIER)
    write_json(out_file, payload_clean["image_classifier"])

    out_file = cfg.out_dir.joinpath("json", cfg.SURVEY_CLASSIFIER)
    write_json(out_file, payload_clean["survey_classifier"])

    return {
        "num_days_image": len(views["image_classifier"].keys()),
        "num_days_survey": len(views["survey_classifier"].keys()),
    }

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

def write_json(out_file, payload):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

def print_stats(stats, out_file):
    logging.info(f"Wrote {stats['total_records']:,} merged records → {out_file}")
    logging.info(f"Survey matches: {stats['survey_matches']:,}")
    logging.info(f"Survey not matched: {stats['survey_not_matched']:,}")
    logging.info(f"Found {stats['manual_masks']:,} manual masks")
    logging.info(f"Number of days for image classifer: {stats['num_days_image']:,}")
    logging.info(f"Number of days for survey classifier {stats['num_days_survey']:,}")
    logging.info(f"Number of ambiguous labels: {stats['num_ambiguous_votes']:,}")
    logging.info(f"Number of metabolite outliers: {stats['num_metabolite_outliers']}")

def main():
    # ---------- command line arguments ----------
    cfg = get_args()

    # ---------- load sources ----------
    sources = load_data_sources(cfg)

    # Build survey map keyed by image_id or parent
    logging.info("Building survey map by (main_id, split_index)...")
    survey_map = build_survey_map(sources.survey_json)
    logging.info(f"Built survey map with {len(survey_map)} unique (main_id, split_index) pairs")

    # Build manual mask map with normalized keys
    logging.info("Normalizing keys for manual mask map...")
    manual_mask_normalized = normalize_manual_mask_map(sources.manual_mask_map, cfg.in_dir)

    # Load processed JSONs
    logging.info("Loading processed files JSON data...")
    processed_map = build_processed_files_map(sources.found_files, cfg.in_dir, cfg.infer_resized_dir)

    # Load preprocessed JSONs
    logging.info("Loading preprocessed files JSON data...")
    preprocessed_map = build_preprocessed_map(sources.preprocessed_files, cfg.in_dir, cfg.infer_resized_dir)

    # ---------- merge ----------
    logging.info("Merging data sources...")
    combined, survey_matched_count, survey_not_matched_count, manual_mask_count = merge_data_sources(
        sources.base_map, survey_map, sources.metab_map, manual_mask_normalized,
        processed_map, preprocessed_map
    )

    # ---------- normalize ----------
    logging.info("Normalizing merged records...")
    records, stats = build_normalized_records(cfg, survey_matched_count, survey_not_matched_count, manual_mask_count, combined)

    # ---------- emit views ----------
    logging.info("Generating derived views...")
    view_stats = generate_views(records, cfg)
    stats.update(view_stats)

    # ----------  print top-level data stats ----------
    print_stats(stats, cfg.ALL_DATA_JSON)



if __name__ == "__main__":
    main()
