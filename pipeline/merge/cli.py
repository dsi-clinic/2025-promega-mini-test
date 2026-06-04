#!/usr/bin/env python3
"""CLI parser + Config for the merge step (step 16)."""

import argparse
import dataclasses
import logging
import typing
from pathlib import Path

from pipeline.data_loader import MIN_VOTES


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
    min_survey_votes: int = dataclasses.field(default=MIN_VOTES, metadata={
        "help": "Minimum votes required to indicate acceptable / not acceptable survey results"
    })
    target_width: int = dataclasses.field(default=512, metadata={
        "help": "Target input image width (pixels)"
    })
    target_height: int = dataclasses.field(default=384, metadata={
        "help": "Target input image height (pixels)"
    })
    no_validate: bool = dataclasses.field(default=False, metadata={
        "help": "Skip schema validation of generated all_data.json"
    })
    out_file: Path = dataclasses.field(default=None, metadata={
        "help": "Output path for all_data.json (default: data/all_data.json in repo)"
    })
    summary_file: Path = dataclasses.field(default=None, metadata={
        "help": "Output path for summary.json (default: data/summary.json in repo)"
    })

    # Paths relative to data_dir / DATA_ROOT
    IDENTIFIERS_DIR: typing.ClassVar[str] = "intermediate/indexes"
    MASKS_DIR: typing.ClassVar[str] = "intermediate/indexes"
    MANUAL_THRESHOLD_MAPPING_JSON: typing.ClassVar[str] = "intermediate/indexes/image_mapping_thresholded_and_manual.json"
    METABOLITE_MAP_JSON: typing.ClassVar[str] = "intermediate/indexes/metabolite_map.json"
    SURVEY_AGGREGATED_JSON: typing.ClassVar[str] = "intermediate/indexes/survey_map.json"

    def __post_init__(self):
        if not self.data_dir.exists():
            raise RuntimeError(f"{self.data_dir} does not exist")
        if not self.image_mapping_json.exists():
            raise RuntimeError(f"{self.image_mapping_json} does not exist")
        if self.identifiers_map_json is None:
            self.identifiers_map_json = self.data_dir / self.IDENTIFIERS_DIR / "record_identifiers.json"
        if self.out_file is None:
            self.out_file = Path("data/all_data.json")
        if self.summary_file is None:
            self.summary_file = Path("data/summary.json")


def create_args() -> argparse.ArgumentParser:
    """ArgumentParser built from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Merge identifiers, image maps, surveys, metabolites → all_data.json")
    for field in dataclasses.fields(Config):
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {"help": field.metadata.get("help", ""), "default": field.default}
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        elif field.type == list:
            kwargs["type"] = int
            kwargs["nargs"] = "+"
            kwargs["default"] = field.default_factory()
        else:
            kwargs["type"] = field.type
        parser.add_argument(*flags, **kwargs)
    return parser


def get_args() -> Config:
    args = create_args().parse_args()
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    return Config(**vars(args))
