#!/usr/bin/env python3
"""Utilities for constructing normalized organoid records and derived views."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import logging
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Protocol


from rich.logging import RichHandler

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO,
                    handlers=[RichHandler()])

SchemaDict = Dict[str, Any]


@dataclass(frozen=True)
class OrganoidRecord:
    """Canonical representation of a single organoid entry."""

    source_id: str
    data: SchemaDict

    def to_dict(self) -> SchemaDict:
        return self.data

    @property
    def record_id(self) -> str:
        return self.data["id"]

    @property
    def day_id(self) -> Optional[str]:
        return self.data.get("day", {}).get("id")

    @property
    def processed_img_path(self) -> Optional[str]:
        return self.data.get("images", {}).get("processed", {}).get("img_path")

    @property
    def overlay_img_path(self) -> Optional[str]:
        return self.data.get("images", {}).get("processed", {}).get("overlay_path")

    @property
    def processed_mask_path(self) -> Optional[str]:
        return self.data.get("images", {}).get("processed", {}).get("mask_path")

    @property
    def image_quality_label(self) -> Optional[str]:
        label = self.data.get("images", {}).get("label", {})
        return label.get("acceptance_flag") if isinstance(label, dict) else None

    @property
    def survey_majority_label(self) -> Optional[str]:
        label = self.data.get("survey", {}).get("label", {})
        return label.get("acceptance_flag") if isinstance(label, dict) else None

    @property
    def survey_evaluation(self) -> Optional[List[dict]]:
        return self.data.get("survey", {}).get("evaluations", {})


@dataclass
class RecordMetrics:
    num_records: int = 0

    num_img_paths: int = 0
    num_img_split: int = 0
    num_img_stitched: int = 0
    num_img_no_label: int = 0
    num_manual_masks: int = 0
    num_mask_paths: int = 0
    num_overlay_paths: int = 0

    num_labels: int = 0
    num_no_labels: int = 0
    num_survey_labels: int = 0
    num_preprocessed_labels: int = 0

    num_no_metabolite: int = 0
    num_metabolites: int = 0
    num_metabolite_outliers: int = 0
    metabolite_outlier_counts: Counter = field(default_factory=Counter)

    num_survey: int = 0
    num_no_survey: int = 0
    num_acceptable_votes: int = 0
    num_not_acceptable_votes: int = 0
    num_majority: int = 0
    num_no_majority: int = 0
    total_votes: int = 0

    num_labels: int = 0

    SPLIT_OR_STITCHED: ClassVar[dict] = {
        "NoSplitNoStitched": (0, 0),
        "SplitNoStitched": (1, 0),
        "NoSplitStitched": (0, 1),
        "SplitStitched": (1, 1),
    }

    METABOLITES: ClassVar[list] = [
        "BCAAGlo",
        "GlucoseGlo",
        "GlutamateGlo",
        "LactateGlo",
        "MalateGlo",
        "PyruvateGlo"
    ]

    def to_dict(self) -> dict:
        self.metabolite_outlier_counts = dict(self.metabolite_outlier_counts)
        return asdict(self)


class OrganoidRecordBuilder:
    """Factory creating OrganoidRecord instances from merged payloads."""

    LABEL_MAP = {"Accepted": 1, "Not Accepted": 0, "Acceptable": 1, "Not Acceptable": 0}

    def __init__(self, *, min_survey_votes: int = 4, survey_day: int = 30,
                 target_size: tuple[int, int] = (512, 384), record_metrics: RecordMetrics):
        self.min_survey_votes = min_survey_votes
        self.survey_day = f"Dy{survey_day:02d}"
        self.target_size = target_size
        self.record_metrics = record_metrics

    def build(self, source_id: str, entry: SchemaDict) -> OrganoidRecord:
        survey = entry.get("survey", {})
        metabolite = entry.get("metabolite", {})

        manual_mask_path = entry.get("manual_mask_path")
        label = entry.get("label", {})

        day_value = entry.get("mdl_day")
        formatted_day = f"{day_value:.1f}".rstrip("0").rstrip(".") if day_value is not None else ""

        payload: SchemaDict = {
            "id": source_id,
            "day": {
                "id": f"Dy{formatted_day}",
                "number": day_value,
                "original": entry.get("original_day")
            },
            "cell_line": entry.get("cellLine"),
            "plate": {
                "batch": entry.get("BA"),
                "well": entry.get("wellID"),
            },
            "metadata": {
                "classification": entry.get("Classification"),
                "treatment": entry.get("treatment"),
                "verification": entry.get("verification"),
            },
            "images": self._build_images(entry, manual_mask_path),
            "metabolite": metabolite,
            "survey": self._build_surveys(survey),
            "label": label,
        }
        self._get_record_metrics(payload)
        return OrganoidRecord(source_id=source_id, data=payload)

    def _build_images(
        self,
        entry: SchemaDict,
        manual_mask_path: Optional[str],
    ) -> SchemaDict:
        raw_images = entry.get("all_files") or []

        best_z = {
            "index": entry.get("Best Z"),
            "path": entry.get("Best Z Filename"),
            "actual_z_value": entry.get("Actual Z Value"),
        }

        pre_split_days = entry.get("pre_split_days", [])
        if pre_split_days:
            entry["pre_split_days"] = pre_split_days

        return {
            "main_id": entry.get("verification", {}).get("main_id"),
            "img_path": entry.get("processed_image"),
            "mask_path": entry.get("predicted_mask_path"),
            "manual_mask_path": manual_mask_path,
            "overlay_path": entry.get("overlay_path"),
            "dimensions_px": {
                "orig": {
                    "width": entry.get("orig_width_px"),
                    "height": entry.get("orig_height_px"),
                },
                "target": {
                    "width": self.target_size[0],
                    "height": self.target_size[1],
                },
            },
            "um_per_px": {
                "orig": entry.get("orig_um_per_px_x"),
                "final": {
                    "x": entry.get("final_um_per_px_x"),
                    "y": entry.get("final_um_per_px_y"),
                },
            },
            "raw_images": raw_images,
            "best_z": best_z,
            "pre_split_days": pre_split_days,
        }

    def _build_surveys(self, survey: SchemaDict) -> Optional[SchemaDict]:
        if not survey:
            return {}
        evaluations = [
            {
                "reviewer": row.get("employee"),
                "original_image_ref": row.get("original_image_ref"),
                "evaluation": row.get("evaluation"),
                "source_file": row.get("source_file"),
                "main_id": row.get("main_id"),
                "raw_organoid_id": row.get("raw_organoid_id"),
                "split_index": row.get("split_index"),
            }
            for row in survey.get("evaluations") or []
        ]
        quality_scores = [
            {
                "quality": row.get("quality"),
                "source_file": row.get("source_file"),
                "main_id": row.get("main_id"),
                "raw_organoid_id": row.get("raw_organoid_id"),
                "split_index": row.get("split_index"),
            }
            for row in survey.get("quality_scores") or []
        ]
        return {
            "evaluations": evaluations,
            "quality_scores": quality_scores,
        }

    def _get_record_metrics(self, record: SchemaDict) -> SchemaDict:
        """Get metrics for a record.
        Args:
            record: The record to get metrics for

        Returns:
            Updates record_metrics attribute
        """
        main_id = record.get("id")
        self.record_metrics.num_records += 1

        # Track split and stitched images
        spl_stc_label = record.get("metadata", {}).get("verification", {}).get("classification_verification")
        split, stitched = self.record_metrics.SPLIT_OR_STITCHED[spl_stc_label]
        self.record_metrics.num_img_split += split
        self.record_metrics.num_img_stitched += stitched
        if split and stitched: logging.warning(f"{main_id}: Image has been split and stitched")

        # Track image paths
        img_path = record.get("images", {}).get("img_path")
        if img_path:
            self.record_metrics.num_img_paths += 1

        # Track predicted masks
        mask_path = record.get("images", {}).get("mask_path")
        if mask_path:
            self.record_metrics.num_mask_paths += 1

        # Track overlay paths
        overlay_path = record.get("images", {}).get("overlay_path")
        if overlay_path:
            self.record_metrics.num_overlay_paths += 1

        # Track manual masks
        manual_mask_path = record.get("images", {}).get("manual_mask_path")
        if manual_mask_path:
            self.record_metrics.num_manual_masks += 1

        # Track labels
        label = record.get("label", {})
        if not label.get("value"):
            self.record_metrics.num_no_labels += 1
        else:
            self.record_metrics.num_labels += 1

        # Track metabolites
        metabolite_data = record.get("metabolite", {})
        if not metabolite_data:
            self.record_metrics.num_no_metabolite += 1
        else:
            self.record_metrics.num_metabolites += 1
            missing = set(self.record_metrics.METABOLITES) - set(metabolite_data)
            extra = set(metabolite_data) - set(self.record_metrics.METABOLITES)
            if missing or extra:
                logging.warning(f"{main_id} Missing metabolites: {missing}, Extra metabolites: {extra}")
            for metabolite_key, details in metabolite_data.items():
                if details["is_outlier"]:
                    self.record_metrics.metabolite_outlier_counts[metabolite_key] += 1
                    self.record_metrics.num_metabolite_outliers += 1

        # Track surveys
        survey_data = record.get("survey", {})
        if survey_data:
            self.record_metrics.num_survey += 1
            survey_votes = record.get("label", {}).get("votes", {})
            self.record_metrics.total_votes += sum(survey_votes.values())
            if "Acceptable" in survey_votes.keys():
                self.record_metrics.num_acceptable_votes += survey_votes["Acceptable"]
            if "Not Acceptable" in survey_votes.keys():
                self.record_metrics.num_not_acceptable_votes += survey_votes["Not Acceptable"]

            survey_label = record.get("label", {}).get("value")
            if not survey_label:
                self.record_metrics.num_no_majority += 1
            else:
                self.record_metrics.num_majority += 1
        else:
            self.record_metrics.num_no_survey += 1
