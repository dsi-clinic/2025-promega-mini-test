#!/usr/bin/env python3
"""Config dataclass + CLI parser for the image-survey classifier (Step 18)."""

import argparse
import dataclasses
from pathlib import Path


@dataclasses.dataclass
class Config:
    data_dir: Path = dataclasses.field(metadata={
        "help": "Path to base data directory (root) containing identifiers/, images/, classifiers/, etc."
    })
    all_data_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to all data JSON file (defaults to data_dir/identifiers/all_data.json)"
    })
    survey_classifier_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to survey classifier JSON file (defaults to data_dir/identifiers/survey_classifier.json)"
    })
    batch_size: int = dataclasses.field(default=8, metadata={"help": "Training batch size"})
    epoch1: int = dataclasses.field(default=50, metadata={
        "help": "Number of training epochs for phase 1 (frozen backbone)"
    })
    epoch2: int = dataclasses.field(default=150, metadata={
        "help": "Number of training epochs for phase 2 (unfrozen backbone)"
    })
    target_day: str = dataclasses.field(default="Dy30", metadata={"help": "Target day"})
    target_width: int = dataclasses.field(default=224, metadata={"help": "Target input image width (pixels)"})
    target_height: int = dataclasses.field(default=224, metadata={"help": "Target input image height (pixels)"})
    deterministic: bool = dataclasses.field(default=False, metadata={
        "help": "Use deterministic operations"
    })
    seed: int = dataclasses.field(default=1, metadata={"help": "Random seed for reproducibility"})

    def __post_init__(self):
        self.target_size: tuple = (self.target_width, self.target_height)
        self.out_dir = self.data_dir / "models" / "image_survey_classification"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.survey_classifier_json = (
            Path(self.survey_classifier_json) if self.survey_classifier_json
            else self.data_dir / "identifiers" / "survey_classifier.json"
        )
        self.all_data_json = (
            Path(self.all_data_json) if self.all_data_json
            else self.data_dir / "identifiers" / "all_data.json"
        )


def create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image survey classifier on organoid images")
    for field in dataclasses.fields(Config):
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {"help": field.metadata.get("help", ""), "default": field.default}
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type
        parser.add_argument(*flags, **kwargs)
    return parser


def get_args() -> Config:
    return Config(**vars(create_args().parse_args()))
