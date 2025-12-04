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
    num_organoids: int = 0

    num_img_split: int = 0
    num_img_stitched: int = 0
    num_img_no_label: int = 0

    num_no_metabolite: int = 0
    num_metabolites: int = 0
    num_metabolite_outliers: int = 0
    metabolite_outlier_counts: Counter = field(default_factory=Counter)

    num_acceptable_votes: int = 0
    num_not_acceptable_votes: int = 0
    num_ambiguous_votes: int = 0

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
        processed = entry.get("processed") or {}
        preprocessed = entry.get("preprocessed") or {}
        survey = entry.get("survey") or {}
        metabolites = entry.get("metabolites") or {}
        manual_mask_path = entry.get("manual_mask_path")

        day_value = entry.get("mdl_day")
        formatted_day = f"{day_value:.1f}".rstrip("0").rstrip(".") if day_value is not None else ""

        payload: SchemaDict = {
            "id": source_id,
            "day": {
                "id": f"Dy{formatted_day}",
                "number": entry.get("mdl_day"),
                "original": entry.get("dayID")
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
        self._get_record_metrics(payload)
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
        inv_votes = Counter()
        reg_votes = Counter()
        for eval_entry in evaluations:
            vote = eval_entry.get("evaluation")
            if vote:
                original_image_ref = eval_entry.get("original_image_ref")
                if "INV" in original_image_ref:
                    inv_votes[vote] += 1
                else:
                    reg_votes[vote] += 1

        winning_inv_label = next(
            (label for label, count in inv_votes.items() if count >= self.min_survey_votes),
            None,
        )

        winning_reg_label = next(
            (label for label, count in reg_votes.items() if count >= self.min_survey_votes),
            None,
        )

        if inv_votes and inv_votes[winning_inv_label] != reg_votes[winning_reg_label]:
            main_id = survey.get("quality_scores", [])[0].get("main_id")
            logging.warning(f"{main_id}:  Inverted evaluation - {inv_votes[winning_inv_label]} '{winning_inv_label}' does not match regular evaluation - {reg_votes[winning_reg_label]} '{winning_reg_label}'")
            winning_reg_label = None

        total = sum(inv_votes.values()) + sum(reg_votes.values())

        return {
            "value": winning_reg_label,
            "acceptance_flag": self.LABEL_MAP.get(winning_reg_label) if winning_reg_label else None,
            "votes": dict(reg_votes + inv_votes),
            "total_evaluations": total,
            "min_votes": self.min_survey_votes,
            "source": "survey.evaluations",
        }

    def _get_record_metrics(self, record: SchemaDict) -> SchemaDict:
        main_id = record.get("id")
        self.record_metrics.num_organoids += 1

        spl_stc_label = record.get("metadata", {}).get("verification", {}).get("classification_verification")
        split, stitched = self.record_metrics.SPLIT_OR_STITCHED[spl_stc_label]
        self.record_metrics.num_img_split += split
        self.record_metrics.num_img_stitched += stitched
        if split and stitched: logging.warning(f"Image has been split and stitched: {main_id}")

        img_label = record.get("images", {}).get("label", {}).get("value")
        if not img_label:
            self.record_metrics.num_img_no_label += 1

        metabolite_data = record.get("metabolites", {})
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

        survey_votes = record.get("survey", {}).get("summary", {}).get("votes", {})
        if survey_votes:
            if "Acceptable" in survey_votes.keys():
                self.record_metrics.num_acceptable_votes += survey_votes["Acceptable"]
            if "Not Acceptable" in survey_votes.keys():
                self.record_metrics.num_not_acceptable_votes += survey_votes["Not Acceptable"]

            survey_label = record.get("survey", {}).get("label", {}).get("value")
            if not survey_label:
                self.record_metrics.num_ambiguous_votes += 1


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
        fields = ("id", "img_path", "label", "mask_path", "overlay_path")

        records = {"records": {}, "metadata": {}}
        for day, rows in self._records_by_day.items():
            day_data = {name: [row.get(name) for row in rows] for name in fields}

            # remove fields where every value is None
            for name, values in list(day_data.items()):
                if all(value is None for value in values):
                    day_data.pop(name)

            records["records"][day] = day_data

        for day, skipped in self._skipped_records_by_day.items():
            records["records"].setdefault(day, {})["skipped"] = skipped

        records["metadata"]["total_skipped"] = sum(
            len(skipped) for skipped in self._skipped_records_by_day.values()
        )

        return records


class ImageClassifierEmitter(BaseViewEmitter):
    """Build view payload for the image classifier training script."""

    name = "image_classifier"

    def __init__(self):
        self._records_by_day: Dict[str, List[SchemaDict]] = defaultdict(list)
        self._skipped_records_by_day: Dict[str, List[str]] = defaultdict(list)

    def process(self, record: OrganoidRecord) -> None:
        label = record.image_quality_label
        img_path = record.processed_img_path
        mask_path = record.processed_mask_path

        if label not in self.label_list or not img_path or not mask_path \
            or not record.day_id:
            self._skipped_records_by_day[record.day_id].append(record.record_id)
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

class SurveyClassifierEmitter(BaseViewEmitter):
    """Build view payload for the human-survey grounded classifier."""

    name = "survey_classifier"

    def __init__(self, survey_day: int = 30, min_votes: int = 4):
        self.survey_day = f"Dy{survey_day:02d}"
        self.min_votes = min_votes
        self._records_by_day: Dict[str, List[SchemaDict]] = defaultdict(list)
        self._skipped_records_by_day: Dict[str, List[str]] = defaultdict(list)

    def process(self, record: OrganoidRecord) -> None:
        if record.day_id != self.survey_day:
            return

        evals = record.survey_evaluation
        img_path = record.processed_img_path
        mask_path = record.processed_mask_path
        label = record.survey_majority_label

        if not eval or not record.day_id or not img_path or not mask_path \
            or label not in self.label_list:
            self._skipped_records_by_day[record.day_id].append(record.record_id)
            return

        payload = {
            "id": record.record_id,
            "img_path": img_path,
            "mask_path": mask_path,
            "label": label,
        }
        self._records_by_day[record.day_id].append(payload)


def emit_views(records: Mapping[str, OrganoidRecord], emitters: Iterable[ViewEmitter]) -> Dict[str, SchemaDict]:
    """Run records through each emitter and collect the resulting views."""
    emitters = list(emitters)
    for record in records.values():
        for emitter in emitters:
            emitter.process(record)
    return {emitter.name: emitter.finalize() for emitter in emitters}

