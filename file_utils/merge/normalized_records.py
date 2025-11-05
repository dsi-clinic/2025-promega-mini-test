#!/usr/bin/env python3
"""Utilities for constructing normalized organoid records and derived views."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol


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
    def survey_majority_label(self) -> Optional[SchemaDict]:
        label = self.data.get("survey", {}).get("label", {})
        return label.get("acceptance_flag") if isinstance(label, dict) else None


class OrganoidRecordBuilder:
    """Factory creating OrganoidRecord instances from merged payloads."""

    LABEL_MAP = {"Accepted": 1, "Not Accepted": 0, "Acceptable": 1, "Not Acceptable": 0}

    def __init__(self, *, schema_version: int = 1, min_survey_votes: int = 4,
                 target_size: tuple[int, int] = (512, 384)):
        self.schema_version = schema_version
        self.min_survey_votes = min_survey_votes
        self.target_size = target_size

    def build(self, source_id: str, entry: SchemaDict) -> OrganoidRecord:
        processed = entry.get("processed") or {}
        preprocessed = entry.get("preprocessed") or {}
        survey = entry.get("survey") or {}
        metabolites = entry.get("metabolites") or {}
        manual_mask_path = entry.get("manual_mask_path")

        payload: SchemaDict = {
            "id": source_id,
            "schema_version": self.schema_version,
            "day": {
                "id": entry.get("dayID"),
                "number": entry.get("mdl_day"),
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
            "images": self._build_images(entry, processed, preprocessed, manual_mask_path),
            "metabolites": metabolites,
            "survey": self._build_surveys(survey)
        }
        return OrganoidRecord(source_id=source_id, data=payload)

    def _build_images(
        self,
        entry: SchemaDict,
        processed: SchemaDict,
        preprocessed: SchemaDict,
        manual_mask_path: Optional[str],
    ) -> SchemaDict:
        raw_images = entry.get("all_files") or []
        best_z = {
            "index": entry.get("Best Z"),
            "path": entry.get("Best Z Filename"),
            "actual_z_value": entry.get("Actual Z Value"),
        }
        processed_block = {
            "main_id": processed.get("main_id"),
            "img_path": processed.get("img_path"),
            "mask_path": processed.get("mask_path"),
            "overlay_path": preprocessed.get("overlay_path"),
            "dimensions_px": {
                "orig": {
                    "width": processed.get("orig_width_px"),
                    "height": processed.get("orig_height_px"),
                },
                "target": {
                    "width": self.target_size[0],
                    "height": self.target_size[1],
                },
            },
            "um_per_px": {
                "orig": processed.get("orig_um_per_px_x"),
                "final": {
                    "x": processed.get("final_um_per_px_x"),
                    "y": processed.get("final_um_per_px_y"),
                },
            },
            "is_stitched": processed.get("is_stitched"),
            "calibration_source": processed.get("calibration_source"),
        }
        preprocessing_block =  {
            "variant": preprocessed.get("variant"),
            "best_z_filename": preprocessed.get("Best Z Filename"),
            "metadata_key": preprocessed.get("metadata_key"),
        }
        label = preprocessed.get("label")
        label_block = {
            "value": label,
            "acceptance_flag": self.LABEL_MAP.get(label),
            "source": "preprocessed",
        }
        return {
            "raw_um_per_px": entry.get("um_per_px"),
            "raw_images": raw_images,
            "best_z": best_z,
            "processed": processed_block,
            "preprocessed": preprocessing_block,
            "manual_mask_path": manual_mask_path,
            "label": label_block,
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
        majority = self._compute_survey_majority(survey)
        summary = {
            "total_evaluations": majority["total_evaluations"],
            "votes": majority["votes"],
            "min_votes": majority["min_votes"],
        }
        label = {
            "value": majority["value"],
            "acceptance_flag": majority["acceptance_flag"],
            "source": majority["source"],
        }
        return {
            "evaluations": evaluations,
            "quality_scores": quality_scores,
            "summary": summary,
            "label": label,
        }

    def _compute_survey_majority(self, survey: SchemaDict) -> Optional[SchemaDict]:
        evaluations: List[SchemaDict] = survey.get("evaluations") or []
        votes = Counter()
        for eval_entry in evaluations:
            vote = eval_entry.get("evaluation")
            if vote:
                votes[vote] += 1
        if not votes:
            return None
        total = sum(votes.values())
        winning_label = next(
            (label for label, count in votes.items() if count >= self.min_survey_votes),
            None,
        )
        return {
            "value": winning_label,
            "acceptance_flag": self.LABEL_MAP.get(winning_label) if winning_label else None,
            "votes": dict(votes),
            "total_evaluations": total,
            "min_votes": self.min_survey_votes,
            "source": "survey.evaluations",
        }


class ViewEmitter(Protocol):
    """Protocol for view emitters."""

    name: str

    def process(self, record: OrganoidRecord) -> None:
        ...

    def finalize(self) -> SchemaDict:
        ...


class BaseViewEmitter:
    """Shared scaffolding for concrete emitters."""

    name: str
    label_list = [0,1]

    def process(self, record: OrganoidRecord) -> None:
        raise NotImplementedError

    def finalize(self) -> SchemaDict:
        raise NotImplementedError


class ImageClassifierEmitter(BaseViewEmitter):
    """Build view payload for the image classifier training script."""

    name = "image_classifier"

    def __init__(self):
        self._records_by_day: Dict[str, List[SchemaDict]] = defaultdict(list)

    def process(self, record: OrganoidRecord) -> None:
        if not record.day_id:
            return

        label = record.image_quality_label
        if label not in self.label_list:
            return

        img_path = record.processed_img_path
        if not img_path:
            return

        mask_path = record.processed_mask_path
        if not mask_path:
            return

        overlay_path = record.overlay_img_path

        payload = {
            "id": record.record_id,
            "img_path": img_path,
            "label": label,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
        }
        self._records_by_day[record.day_id].append(payload)

    def finalize(self) -> SchemaDict:
        return {
            day: {
                "id": [row.get("id") for row in rows],
                "img_path": [row.get("img_path") for row in rows],
                "label": [row.get("label") for row in rows],
                "mask_path": [row.get("mask_path") for row in rows],
                "overlay_path": [row.get("overlay_path") for row in rows],
            }
            for day, rows in self._records_by_day.items()
        }


class SurveyClassifierEmitter(BaseViewEmitter):
    """Build view payload for the human-survey grounded classifier."""

    name = "survey_classifier"

    def __init__(self, survey_day: int = 30, min_votes: int = 4):
        self.survey_day = f"Dy{survey_day:02d}"
        self.min_votes = min_votes
        self._records_by_day: Dict[str, List[SchemaDict]] = defaultdict(list)
        self._ambiguous_counts = 0

    def process(self, record: OrganoidRecord) -> None:
        if not record.day_id:
            return

        if record.day_id != self.survey_day:
            return

        img_path = record.processed_img_path
        if not img_path:
            return

        mask_path = record.processed_mask_path
        if not mask_path:
            return

        label = record.survey_majority_label
        if label not in self.label_list:
            self._ambiguous_counts += 1
            return

        payload = {
            "id": record.record_id,
            "img_path": img_path,
            "mask_path": mask_path,
            "label": label,
        }
        self._records_by_day[record.day_id].append(payload)

    def finalize(self) -> SchemaDict:
        return {
            day: {
                "id": [row.get("id") for row in rows],
                "img_path": [row.get("img_path") for row in rows],
                "label": [row.get("label") for row in rows],
                "mask_path": [row.get("mask_path") for row in rows],
                "ambiguous_labels_tally": self._ambiguous_counts,
            }
            for day, rows in self._records_by_day.items()
        }



def emit_views(records: Mapping[str, OrganoidRecord], emitters: Iterable[ViewEmitter]) -> Dict[str, SchemaDict]:
    """Run records through each emitter and collect the resulting views."""
    emitters = list(emitters)
    for record in records.values():
        for emitter in emitters:
            emitter.process(record)
    return {emitter.name: emitter.finalize() for emitter in emitters}

