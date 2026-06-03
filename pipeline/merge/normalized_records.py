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

# Canonical survey/label day. Organoid-level label tracking and split-conflict
# detection consider ONLY records on this day: the survey is conducted on the
# final timepoint (Dy30) and downstream analysis reads the label from Dy30
# (see data_loader.LABEL_DAY). Secondary reviews on other days (e.g. a Dy28
# survey) keep their own per-day label but must NOT participate in conflict
# detection — otherwise a Dy28-vs-Dy30 disagreement is mistaken for a split
# conflict and the organoid's label is wiped everywhere.
SURVEY_LABEL_DAY = "Dy30"


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
    def organoid_id(self) -> str:
        return self.data["organoid_id"]

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

    num_label_skipped: int = 0

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

    def __init__(self, *, min_survey_votes: int = 4,
                 target_size: tuple[int, int] = (512, 384), record_metrics: RecordMetrics):
        self.min_survey_votes = min_survey_votes
        self.target_size = target_size
        self.record_metrics = record_metrics
        self.organoid_dict = {}
        self.conflicted_organoids: set = set()  # organoid_ids with split label conflicts

    def build(self, source_id: str, entry: SchemaDict) -> OrganoidRecord:
        survey = entry.get("survey", {})
        metabolite = entry.get("metabolite", {})
        manual_mask_path = entry.get("manual_mask_path")
        manual_mask_path_orginal = entry.get("manual_mask_path_original")

        label = entry.get("label", {})
        organoid_id = f"{entry.get('BA')} {entry.get('wellID')}".replace(" ", "_")
        day_id = entry.get('dayID') or ''
        if label:
            label = self._get_organoid_labels(organoid_id, source_id, label, day_id)

        day_value = entry.get("mdl_day")
        formatted_day = f"{day_value:.1f}".rstrip("0").rstrip(".") if day_value is not None else ""

        payload: SchemaDict = {
            "id": source_id,
            "organoid_id": organoid_id,
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
            "images": self._build_images(entry, manual_mask_path, manual_mask_path_orginal),
            "metabolite": metabolite,
            "survey": self._build_surveys(survey),
            "label": label,
        }
        self._get_record_metrics(payload)
        return OrganoidRecord(source_id=source_id, data=payload)

    def _get_organoid_labels(self, organoid_id: str, source_id: str, label: dict, day_id: str) -> dict:
        """Get organoid labels and track them in the organoid dictionary.
        Args:
            organoid_id: The organoid ID
            source_id: The source ID
            label: The label
            day_id: The day identifier (e.g. 'Dy30')

        Returns:
            The label
        """
        # Only the canonical survey day defines an organoid's label and can
        # raise a split conflict. Records on other days keep their own per-day
        # label but are not tracked organoid-wide (see SURVEY_LABEL_DAY) — this
        # prevents a cross-day disagreement (e.g. Dy28 vs Dy30) from being
        # misread as a split conflict.
        if day_id != SURVEY_LABEL_DAY:
            return label

        if organoid_id in self.organoid_dict:
            existing = self.organoid_dict[organoid_id]
            existing_value = existing["label"].get("value")
            new_value = label.get("value")

            if existing_value is not None and new_value is not None and existing_value != new_value:
                # Both splits have definitive labels that disagree — conflict
                logging.warning(f"Labels do not match between days or splits: {source_id}/{organoid_id}. All labels for this organoid will be cleared.")
                self._register_split_conflict(organoid_id, source_id)
                label = {}

            elif existing_value is None and new_value is not None:
                # Existing had no majority; upgrade to the definitive label
                self.organoid_dict[organoid_id] = {"source_id": source_id, "label": label, "day_id": day_id}

        # else: new_value is None (keep existing), or both same value — no change
        else:
            self.organoid_dict[organoid_id] = {"source_id": source_id, "label": label, "day_id": day_id}

        return label

    def _register_split_conflict(self, organoid_id: str, source_id: str) -> None:
        """Register a split label conflict for an organoid, clearing it from propagation tracking."""
        logging.warning(f"Split label conflict registered for {source_id}/{organoid_id}. All labels will be removed after propagation.")
        self.record_metrics.num_label_skipped += 1
        self.conflicted_organoids.add(organoid_id)
        if organoid_id in self.organoid_dict:
            del self.organoid_dict[organoid_id]

    def _build_images(
        self,
        entry: SchemaDict,
        manual_mask_path: Optional[str],
        manual_mask_path_orginal: Optional[str]
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

        # ---- clipped meanfill (merge std + ar into one field) ----
        cm = dict(entry.get("clipped_meanfill", {}) or {})

        if entry.get("clipped_meanfill_std"):
            cm["std"] = entry["clipped_meanfill_std"]

        if entry.get("clipped_meanfill_ar"):
            cm["ar"] = entry["clipped_meanfill_ar"]

        return {
            "main_id": entry.get("verification", {}).get("main_id"),
            "img_path": entry.get("processed_image"),
            "mask_path": entry.get("predicted_mask_path"),
            "manual_mask_path": manual_mask_path,
            "manual_mask_path_orginal": manual_mask_path_orginal,
            "overlay_path": entry.get("overlay_path"),
            "edge_fraction": entry.get("edge_fraction"),
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
            "aspect_ratio": entry.get("aspect_ratio", {}),
            "clipped_meanfill": cm,
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
