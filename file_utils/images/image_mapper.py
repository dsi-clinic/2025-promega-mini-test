# image_mapper.py
from __future__ import annotations
import argparse
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import re
from collections import defaultdict
from typing import Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

from file_utils.common.organoid_patterns import (
    OrganoidPatterns,
    OrganoidNormalizer,
    clean_id_for_json,
)
from file_utils.images.metadata_resolver import load_and_clean_metadata
from file_utils.images.verification import Verifier
from file_utils.images.image_resolver import (
    resolve_image,
    group_by_split,
    extract_z_level,
    find_best_focus,
    classify_image_file,
    list_image_files,
)

# Constants
DAY_20_21 = "Dy20.5"
STITCHED_FLAG = "Stitched"
SPLIT_AMBIGUOUS_FLAG = "SplitAmbiguous"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("image_mapper.log")
    ],
)


@dataclasses.dataclass
class ProcessingStats:
    """Statistics for image processing."""
    found_count: int = 0
    stitched_count: int = 0
    skipped: int = 0
    id_found: int = 0

    def __add__(self, other: ProcessingStats) -> ProcessingStats:
        """Add two ProcessingStats together."""
        return ProcessingStats(
            found_count=self.found_count + other.found_count,
            stitched_count=self.stitched_count + other.stitched_count,
            skipped=self.skipped + other.skipped,
            id_found=self.id_found + other.id_found,
        )


class ImageMapper:
    def __init__(
        self,
        base_dir: pathlib.Path,
        meta_csv: pathlib.Path,
        verify_csv: pathlib.Path | None = None,
        identifiers: pathlib.Path | None = None
    ):
        self.base_dir = pathlib.Path(base_dir)
        self.meta_csv = pathlib.Path(meta_csv)
        self.verifier = Verifier(verify_csv)
        self.identifiers = pathlib.Path(identifiers) if identifiers else None

    def make_mapping_json(self, out_json: pathlib.Path) -> None:
        """
        Make a mapping JSON file from the metadata CSV.

        Args:
            out_json: the path to the output JSON file
        """
        identifiers = load_identifiers(self.identifiers) if self.identifiers else set()
        cleaned = load_and_clean_metadata(self.meta_csv)

        grouped = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        presplit_wells = self._compute_presplit_wells(grouped)

        mapping: Dict[str, Dict[str, Any]] = {}
        total_groups = len(grouped)
        stats = ProcessingStats()

        logging.info(f"[ImageMapper] Processing {total_groups} unique (day,batchPlate,well) groups")
        logging_level = logging.getLevelName(logging.getLogger().level)
        if logging_level == "DEBUG":
            iterator = grouped
        else:
            iterator = tqdm(
                grouped,
                desc="Processing groups",
                mininterval=0.5,  # Update at most every 0.5 seconds
                maxinterval=1.0,   # Force update every 1 second
                file=None,         # Use stderr (default)
                dynamic_ncols=True,  # Adjust to terminal width
                leave=True,        # Keep progress bar after completion
            )

        for (day_id, batch_plate, well_id), group_df in iterator:
            batch_stats, mapped_entries = self.process_batches(
                identifiers, batch_plate, day_id, well_id, presplit_wells, group_df
            )
            mapping.update(mapped_entries)
            stats += batch_stats

        self.write_output(out_json, mapping, total_groups, stats)

    def _compute_presplit_wells(self, grouped) -> set[Tuple[str, str, str]]:
        """
        Determine (dayID, batchPlate, wellID) combos that occur BEFORE the first split
        for that (batchPlate, wellID).

        Args:
            grouped: DataFrameGroupBy over ["dayID", "batchPlate", "wellID"]

        Returns:
            set of tuples (dayID, batchPlate, wellID) that are presplit.
        """
        keys = [(day_id, batch_plate, well_id) for (day_id, batch_plate, well_id), _ in grouped]
        if not keys:
            logging.info("[ImageMapper] Detected 0 presplit wells")
            return set()

        df_keys = pd.DataFrame(keys, columns=["dayID", "batchPlate", "wellID"])
        df_keys["dnum"] = df_keys["dayID"].apply(OrganoidNormalizer.extract_day_number)

        presplit_wells: set[Tuple[str, str, str]] = set()
        for split_key in self.verifier.verify_splits.keys():
            batch, plate, day, well = split_key.split(" ")
            batch_plate = f"{batch} {plate}"
            split_day_num = OrganoidNormalizer.extract_day_number(day)

            # Query DataFrame: find all days before split day for this batch/plate/well
            mask = (
                (df_keys["batchPlate"].str.upper() == batch_plate.upper()) &
                (df_keys["wellID"] == well) &
                (df_keys["dnum"].notna()) &
                (df_keys["dnum"] < split_day_num)
            )
            presplit_rows = df_keys[mask]
            for _, row in presplit_rows.iterrows():
                presplit_wells.add((row["dayID"], row["batchPlate"], row["wellID"]))

        logging.info(f"[ImageMapper] Detected {len(presplit_wells)} presplit wells")
        return presplit_wells

    def _to_rel(self, p: pathlib.Path) -> pathlib.Path:
        """Convert absolute path to relative path from base_dir."""
        p = pathlib.Path(p)
        try:
            return p.relative_to(self.base_dir)
        except ValueError:
            return pathlib.Path(os.path.relpath(p, self.base_dir))

    def _normalize_day_in_id(self, identifier: str) -> str:
        """Normalize Dy20/Dy21 to Dy20.5 in identifier."""
        return identifier.replace("Dy20", DAY_20_21).replace("Dy21", DAY_20_21)

    def _normalize_batch_plate(self, batch_plate: str) -> str:
        """Normalize batch plate string (uppercase first part)."""
        parts = batch_plate.split()
        return " ".join([parts[0].upper(), *parts[1:]])

    def _build_full_id(self, ba_str: str, day_id: str, well_id: str) -> str:
        """Build and normalize full identifier."""
        raw_full_id = f"{ba_str} {day_id} {well_id}"
        full_id = clean_id_for_json(raw_full_id)
        return self._normalize_day_in_id(full_id)

    def _extract_common_fields(self, group_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract common fields from group dataframe."""
        return {
            "um_per_px": float(group_df["um_per_px"].iloc[0]),
            "cellLine": group_df["cellLine"].iloc[0],
            "treatment": group_df["treatment"].iloc[0],
        }

    def _build_base_entry(
        self,
        day_id: str,
        ba_str: str,
        well_id: str,
        group_df: pd.DataFrame,
        final_file: pathlib.Path,
        actual_z: int,
        classification: str,
        all_files: list[pathlib.Path],
        best_z: int = -1,
    ) -> Dict[str, Any]:
        """Build base entry dictionary with common fields."""
        common_fields = self._extract_common_fields(group_df)
        return {
            "dayID": day_id,
            "BA": ba_str,
            "wellID": well_id,
            "Best Z": best_z,
            "Best Z Filename": str(self._to_rel(final_file)),
            "Actual Z Value": actual_z,
            "Classification": classification,
            "all_files": [str(self._to_rel(f)) for f in sorted(all_files, key=lambda f: extract_z_level(f.name))],
            **common_fields,
        }

    def _check_identifier(
        self,
        identifier: str,
        identifiers: set,
        stats: ProcessingStats
    ) -> bool:
        """
        Check if identifier exists in identifiers set.

        Args:
            identifier: identifier to check
            identifiers: set of valid identifiers
            stats: ProcessingStats to update

        Returns:
            True if identifier exists, False otherwise
        """
        stats.id_found += 1
        if identifier not in identifiers:
            stats.skipped += 1
            # Use tqdm.write() to print above progress bar without interfering
            tqdm.write(f"WARNING: Skipping {identifier} not in identifiers", file=None)
            return False
        return True

    def process_batches(
        self,
        identifiers: set,
        batch_plate: str,
        day_id: str,
        well_id: str,
        presplit_wells: set[Tuple[str, str, str]],
        group_df: pd.DataFrame
    ) -> Tuple[ProcessingStats, Dict[str, Dict[str, Any]]]:
        """
        Process batch plate, day, well group and add to mapping dictionary.

        Args:
            identifiers: set of valid identifiers
            batch_plate: batch plate string
            day_id: day identifier
            well_id: well identifier
            presplit_wells: set of presplit wells
            group_df: dataframe containing group data

        Returns:
            Tuple of (ProcessingStats, mapping dictionary)
        """
        stats = ProcessingStats()
        mapping: Dict[str, Dict[str, Any]] = {}

        ba_str = self._normalize_batch_plate(batch_plate)
        full_id = self._build_full_id(ba_str, day_id, well_id)

        # Resolve image(s)
        chosen, stitched_flag, all_files, stitched_groups = resolve_image(
            base_dir=self.base_dir,
            day_id=day_id,
            well_id=well_id,
            file_photoID=f"{ba_str} {day_id} {well_id}",
        )

        if not all_files and stitched_flag != SPLIT_AMBIGUOUS_FLAG:
            return stats, mapping

        # Regroup on all_files (always every match)
        split_groups_all = group_by_split(all_files)
        child_groups = {k: v for k, v in split_groups_all.items() if k is not None}

        # CASE A: split children
        if len(child_groups) >= 1:
            logging.debug(f"[ImageMapper] Expanding into {len(child_groups)} split children for {full_id}")
            stats, mapping = self._define_entry_split(
                identifiers, child_groups, full_id, day_id, ba_str,
                batch_plate, well_id, group_df, presplit_wells
            )

        # CASE B: multiple stitched groups
        elif stitched_flag == SPLIT_AMBIGUOUS_FLAG and stitched_groups:
            logging.debug(f"[ImageMapper] Processing {len(stitched_groups)} stitched groups for {full_id}")
            stats, mapping = self._define_entry_multiple_stitched(
                identifiers, stitched_groups, full_id, day_id, ba_str,
                batch_plate, well_id, group_df, presplit_wells
            )

        # CASE C: single stitched or regular
        else:
            stats, mapping = self._define_entry_regular(
                identifiers, all_files, full_id, day_id, ba_str,
                batch_plate, well_id, group_df, presplit_wells,
                stitched_flag, chosen
            )

        return stats, mapping

    def _define_entry_split(
        self,
        identifiers: set,
        child_groups: dict,
        full_id: str,
        day_id: str,
        ba_str: str,
        batch_plate: str,
        well_id: str,
        group_df: pd.DataFrame,
        presplit_wells: set[Tuple[str, str, str]]
    ) -> Tuple[ProcessingStats, Dict[str, Dict[str, Any]]]:
        """
        Define entries for split images.

        Args:
            identifiers: set of valid identifiers
            child_groups: dictionary of split child groups
            full_id: the full identifier
            day_id: the day identifier
            ba_str: the batch string
            batch_plate: the batch plate identifier
            well_id: the well identifier
            group_df: the group dataframe
            presplit_wells: the presplit wells

        Returns:
            Tuple of (ProcessingStats, mapping dictionary)
        """
        stats = ProcessingStats(found_count=1)
        mapping: Dict[str, Dict[str, Any]] = {}

        for child_idx, group_files in sorted(child_groups.items()):
            final_file = pick_rep_file(group_files)
            clean_child_key = f"{full_id} split_{int(child_idx)}"

            if not self._check_identifier(clean_child_key, identifiers, stats):
                continue

            actual_z = extract_z_level(final_file.name)
            classification = classify_image_file(final_file.name)

            entry = self._build_base_entry(
                day_id, ba_str, well_id, group_df, final_file,
                actual_z, classification, group_files, best_z=-1
            )
            entry["split_index"] = int(child_idx)

            # Add verification
            split_idx = int(child_idx)
            is_presplit = (day_id, batch_plate, well_id) in presplit_wells
            entry["verification"] = self._get_verification_data(
                ba_str, day_id, well_id, split_idx, classification, is_presplit
            )

            mapping[clean_child_key] = entry

        return stats, mapping

    def _define_entry_multiple_stitched(
        self,
        identifiers: set,
        stitched_groups: dict,
        full_id: str,
        day_id: str,
        ba_str: str,
        batch_plate: str,
        well_id: str,
        group_df: pd.DataFrame,
        presplit_wells: set[Tuple[str, str, str]]
    ) -> Tuple[ProcessingStats, Dict[str, Dict[str, Any]]]:
        """
        Define entries for multiple stitched images.

        Args:
            identifiers: set of valid identifiers
            stitched_groups: dictionary of stitched groups
            full_id: the full identifier
            day_id: the day identifier
            ba_str: the batch string
            batch_plate: the batch plate identifier
            well_id: the well identifier
            group_df: the group dataframe
            presplit_wells: the presplit wells

        Returns:
            Tuple of (ProcessingStats, mapping dictionary)
        """
        stats = ProcessingStats(found_count=1)
        mapping: Dict[str, Dict[str, Any]] = {}

        for identifier, group_files in stitched_groups.items():
            group_files.sort(key=lambda f: extract_z_level(f.name))

            best_idx = find_best_focus(group_files)
            final_file = group_files[best_idx] if 0 <= best_idx < len(group_files) else group_files[0]

            safe_identifier = re.sub(r"[^\w\s]", "", identifier).strip().replace(" ", "_")
            clean_stitched_id = f"{full_id} stitched_{safe_identifier}"

            if not self._check_identifier(clean_stitched_id, identifiers, stats):
                continue

            actual_z = extract_z_level(final_file.name)
            stats.stitched_count += 1

            entry = self._build_base_entry(
                day_id, ba_str, well_id, group_df, final_file,
                actual_z, STITCHED_FLAG, group_files, best_z=best_idx
            )
            entry["stitched_identifier"] = identifier

            # Add verification
            is_presplit = (day_id, batch_plate, well_id) in presplit_wells
            entry["verification"] = self._get_verification_data(
                ba_str, day_id, well_id, None, STITCHED_FLAG, is_presplit
            )

            mapping[clean_stitched_id] = entry

        return stats, mapping

    def _define_entry_regular(
        self,
        identifiers: set,
        all_files: list[pathlib.Path],
        full_id: str,
        day_id: str,
        ba_str: str,
        batch_plate: str,
        well_id: str,
        group_df: pd.DataFrame,
        presplit_wells: set[Tuple[str, str, str]],
        stitched_flag: str,
        chosen: pathlib.Path
    ) -> Tuple[ProcessingStats, Dict[str, Dict[str, Any]]]:
        """
        Define entry for a regular (non-split) image.

        Args:
            identifiers: set of valid identifiers
            all_files: the list of all files
            full_id: the full identifier
            day_id: the day identifier
            ba_str: the batch string
            batch_plate: the batch plate identifier
            well_id: the well identifier
            group_df: the group dataframe
            presplit_wells: the presplit wells
            stitched_flag: the stitched flag
            chosen: the chosen file

        Returns:
            Tuple of (ProcessingStats, mapping dictionary)
        """
        stats = ProcessingStats()
        mapping: Dict[str, Dict[str, Any]] = {}

        if not self._check_identifier(full_id, identifiers, stats):
            return stats, mapping

        if not all_files:
            return stats, mapping

        stats.found_count = 1
        if stitched_flag == STITCHED_FLAG:
            stats.stitched_count = 1

        if stitched_flag == STITCHED_FLAG:
            focus_idx = -1
            final = chosen
            actual_z = extract_z_level(chosen.name)
        else:
            idx = find_best_focus(all_files)
            focus_idx = idx if 0 <= idx < len(all_files) else -1
            final = all_files[focus_idx] if focus_idx >= 0 else chosen
            actual_z = extract_z_level(final.name)

        classification = classify_image_file(final.name)

        entry = self._build_base_entry(
            day_id, ba_str, well_id, group_df, final,
            actual_z, classification, all_files, best_z=focus_idx
        )

        # Add verification
        is_presplit = (day_id, batch_plate, well_id) in presplit_wells
        entry["verification"] = self._get_verification_data(
            ba_str, day_id, well_id, None, classification, is_presplit
        )

        mapping[full_id] = entry
        return stats, mapping

    def _get_verification_data(
        self,
        ba_str: str,
        day_id: str,
        well_id: str,
        split_idx: int | None,
        classification: str,
        is_presplit: bool
    ) -> dict:
        """
        Get verification data for an entry.

        Args:
            ba_str: the batch string
            day_id: the day identifier
            well_id: the well identifier
            split_idx: the split index if applicable
            classification: the classification
            is_presplit: whether the well is a presplit well

        Returns:
            verification dictionary
        """
        gen_main_id = self.verifier.build_main_id(
            ba_str, day_id, well_id, split_idx, classification, presplit_flag=is_presplit
        )
        return self.verifier.lookup(
            ba_str, day_id, well_id, split_idx, classification, gen_main_id
        )

    def write_output(
        self,
        out_json: pathlib.Path,
        mapping: dict,
        total_groups: int,
        stats: ProcessingStats
    ) -> None:
        """
        Write the mapping dictionary to a JSON file.

        Args:
            out_json: the path to the output JSON file
            mapping: the mapping dictionary
            total_groups: total number of groups processed
            stats: ProcessingStats object
        """
        logging.info("=== MAPPING SUMMARY ===")
        logging.info(f"Total groups processed: {total_groups}")
        logging.info(f"Groups with files found: {stats.found_count}")
        logging.info(f"Total entries created: {len(mapping)}")
        logging.info(f"Stitched images detected: {stats.stitched_count}")
        if total_groups:
            logging.info(f"Success rate: {stats.found_count/total_groups*100:.1f}%")
        logging.info(f"Total identifiers found: {stats.id_found}")
        logging.info(f"Total identifiers skipped: {stats.skipped}")

        wrapped = {
            "_base_folder": str(self.base_dir.resolve()),
            "entries": mapping,
        }
        out_json = pathlib.Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(wrapped, indent=2))
        logging.info(f"[ImageMapper] Wrote mapping JSON to {out_json}")


def load_identifiers(identifiers_file: pathlib.Path) -> set:
    """
    Load identifiers from a JSON file.

    Args:
        identifiers_file: the path to the identifiers file

    Returns:
        set of identifier keys from the JSON file
    """
    with open(identifiers_file, "r") as f:
        data = json.load(f)
        return set(data.keys()) if isinstance(data, dict) else set()


def pick_rep_file(files_for_child: list[pathlib.Path]) -> pathlib.Path:
    """
    Pick the representative file from a list of files.

    Priority:
    1. Stitched files
    2. Partial images (best focus)
    3. Best focus file

    Args:
        files_for_child: list of file paths

    Returns:
        selected file path
    """
    files_for_child = sorted(files_for_child, key=lambda f: extract_z_level(f.name))
    stitched = [f for f in files_for_child if "(stitched)" in f.name.lower()]
    if stitched:
        return stitched[0]
    partials = [f for f in files_for_child if OrganoidPatterns.PARTIAL_IMAGE.search(f.name)]
    if partials:
        best_idx = find_best_focus(partials)
        return partials[best_idx if 0 <= best_idx < len(partials) else 0]
    best_idx = find_best_focus(files_for_child)
    return files_for_child[best_idx if 0 <= best_idx < len(files_for_child) else 0]


def get_args() -> argparse.Namespace:
    """
    Get arguments from the command line.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser(description='Map raw images to organoid identifiers')
    parser.add_argument('--base-dir', type=pathlib.Path, required=True,
                        help='The base directory containing the raw images')
    parser.add_argument('--verify-csv', type=pathlib.Path,
                        help='The file containing the image verification data')
    parser.add_argument('--meta-xlsx', type=pathlib.Path, required=True,
                        help='The file containing the metadata')
    parser.add_argument('--out-file', type=pathlib.Path, required=True,
                        help='The file to save the formatted image mapping to')
    parser.add_argument('--identifiers', type=pathlib.Path,
                        help='The file containing the identifiers to map to')
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    start_time = datetime.datetime.now()

    args = get_args()
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    mapper = ImageMapper(
        base_dir=args.base_dir,
        meta_csv=args.meta_xlsx,
        verify_csv=args.verify_csv,
        identifiers=args.identifiers,
    )
    mapper.make_mapping_json(args.out_file)

    end_time = datetime.datetime.now()
    logging.info(f"Elapsed time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
