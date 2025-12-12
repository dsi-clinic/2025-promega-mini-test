# image_mapper.py
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

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

from file_utils.common.organoid_patterns import (
    OrganoidPatterns,
    OrganoidNormalizer,
    clean_id_for_json,
)

from config import (
    RAW_IMAGE_DATA,
    META_FILE,
    IMAGE_VERIFICATION_FORM,
    RAW_IMAGE_MAPPING_JSON,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

class ImageMapper:
    def __init__(self, base_dir: Path, meta_csv: Path, verify_csv: Path | None = None):
        self.base_dir = Path(base_dir)
        self.meta_csv = Path(meta_csv)
        self.verifier = Verifier(verify_csv) if verify_csv else None

    def _compute_presplit_wells(self, cleaned) -> set[Tuple[str, str, str]]:
        grouped = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        by_well_days = defaultdict(list)
        actual_split_wells = set()

        for (dy, bp, w), _ in grouped:
            dnum = OrganoidNormalizer.extract_day_number(dy)
            by_well_days[(bp, w)].append(dnum)

            # flattened structure
            img_folder = self.base_dir

            has_split = False
            if img_folder.exists():
                day_files = list_image_files(img_folder)

                # filter to this well, same as before
                well_files = [
                    f for f in day_files
                    if OrganoidNormalizer.extract_well(f.name) == w
                ]
                if well_files:
                    has_split = any(
                        OrganoidNormalizer.extract_split_info(f.name)["is_split"]
                        for f in well_files
                    )

            if has_split:
                actual_split_wells.add((bp, w, dnum))

        presplit_wells: set[Tuple[str, str, str]] = set()
        for (bp, w), days in by_well_days.items():
            split_days = [d for (bp2, w2, d) in actual_split_wells if bp2 == bp and w2 == w]
            if split_days:
                first_split = min(split_days)
                for d in sorted(days):
                    if d < first_split:
                        presplit_wells.add((f"Dy{d:02d}", bp, w))

        log.info(f"[ImageMapper] Detected {len(presplit_wells)} presplit wells")
        return presplit_wells


    def _to_rel(self, p: Path) -> Path:
        p = Path(p)
        try:
            return p.relative_to(self.base_dir)
        except ValueError:
            return Path(os.path.relpath(p, self.base_dir))

    def make_mapping_json(self, out_json: Path) -> None:
        cleaned = load_and_clean_metadata(self.meta_csv)
        grouped = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        presplit_wells = self._compute_presplit_wells(cleaned)

        mapping: Dict[str, Dict[str, Any]] = {}

        total_groups = len(grouped)
        stitched_count = 0
        found_count = 0

        log.info(f"[ImageMapper] Processing {total_groups} unique (day,batchPlate,well) groups")

        for (day_id, batch_plate, well_id), group_df in grouped:
            log.info(f"[ImageMapper] {batch_plate} {day_id} {well_id}")

            parts = batch_plate.split()
            ba_str = " ".join([parts[0].upper(), *parts[1:]])
            raw_full_id = f"{ba_str} {day_id} {well_id}"
            full_id = clean_id_for_json(raw_full_id)

            # resolve image(s)
            chosen, stitched_flag, all_files, stitched_groups = resolve_image(
                base_dir=self.base_dir,
                day_id=day_id,
                well_id=well_id,
                file_photoID=raw_full_id,
            )

            if not all_files and stitched_flag not in ("SplitAmbiguous",):
                continue

            # regroup on all_files (always every match)
            split_groups_all = group_by_split(all_files)
            child_groups = {k: v for k, v in split_groups_all.items() if k is not None}


            # ---------- CASE A: split children ----------
            if len(child_groups) >= 1:
                log.info(f"[ImageMapper] Expanding into {len(child_groups)} split children for {full_id}")

                def pick_rep_file(files_for_child):
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


                for child_idx, group_files in sorted(child_groups.items()):
                    final_file = pick_rep_file(group_files)
                    clean_child_key = f"{full_id} split_{int(child_idx)}"

                    actual_z = extract_z_level(final_file.name)
                    classification = classify_image_file(final_file.name)

                    entry = {
                        "dayID": day_id,
                        "BA": ba_str,
                        "wellID": well_id,
                        "split_index": int(child_idx),
                        "Best Z": -1,
                        "Best Z Filename": str(self._to_rel(final_file)),
                        "Actual Z Value": actual_z,
                        "Classification": classification,
                        "um_per_px": float(group_df["um_per_px"].iloc[0]),
                        "all_files": [
                            str(self._to_rel(f))
                            for f in sorted(group_files, key=lambda f: extract_z_level(f.name))
                        ],
                        "cellLine": group_df["cellLine"].iloc[0],
                        "treatment": group_df["treatment"].iloc[0],
                    }

                    # verification block (child)
                    split_idx = int(child_idx)
                    is_presplit = (day_id, batch_plate, well_id) in presplit_wells
                    if self.verifier:
                        gen_main_id = self.verifier.build_main_id(
                            ba_str, day_id, well_id, split_idx, classification, presplit_flag=is_presplit
                        )
                        entry["verification"] = self.verifier.lookup(
                            ba_str, day_id, well_id, split_idx, classification, gen_main_id
                        )
                    else:
                        gen_main_id = Verifier.build_main_id(
                            ba_str, day_id, well_id, split_idx, classification, presplit_flag=is_presplit
                        )
                        entry["verification"] = {
                            "main_id": gen_main_id,
                            "gen_main_id": gen_main_id,
                            "classification_verification": Verifier.classification_label_for_verif(
                                split_idx, classification
                            ),
                            "blank_verified": None,
                            "blank": False,
                        }

                    mapping[clean_child_key] = entry

                continue  # done with this (day,ba,well) group

            # ---------- CASE B: multiple stitched groups ----------
            if stitched_flag == "SplitAmbiguous" and stitched_groups:
                log.info(f"[ImageMapper] Processing {len(stitched_groups)} stitched groups for {full_id}")
                for identifier, group_files in stitched_groups.items():
                    group_files.sort(key=lambda f: extract_z_level(f.name))

                    best_idx = find_best_focus(group_files)
                    final_file = group_files[best_idx] if 0 <= best_idx < len(group_files) else group_files[0]

                    safe_identifier = re.sub(r"[^\w\s]", "", identifier).strip().replace(" ", "_")
                    clean_stitched_id = f"{full_id} stitched_{safe_identifier}"

                    actual_z = extract_z_level(final_file.name)

                    found_count += 1
                    stitched_count += 1

                    entry = {
                        "dayID": day_id,
                        "BA": ba_str,
                        "wellID": well_id,
                        "stitched_identifier": identifier,
                        "Best Z": best_idx,
                        "Best Z Filename": str(self._to_rel(final_file)),
                        "Actual Z Value": actual_z,
                        "Classification": "Stitched",
                        "um_per_px": float(group_df["um_per_px"].iloc[0]),
                        "all_files": [
                            str(self._to_rel(f))
                            for f in sorted(group_files, key=lambda f: extract_z_level(f.name))
                        ],

                        "cellLine": group_df["cellLine"].iloc[0],
                        "treatment": group_df["treatment"].iloc[0],
                    }

                    split_idx = None
                    is_presplit = (day_id, batch_plate, well_id) in presplit_wells
                    classification = "Stitched"

                    if self.verifier:
                        gen_main_id = self.verifier.build_main_id(
                            ba_str, day_id, well_id, None, classification, presplit_flag=is_presplit
                        )
                        entry["verification"] = self.verifier.lookup(
                            ba_str, day_id, well_id, None, classification, gen_main_id
                        )
                    else:
                        gen_main_id = Verifier.build_main_id(
                            ba_str, day_id, well_id, None, classification, presplit_flag=is_presplit
                        )
                        entry["verification"] = {
                            "main_id": gen_main_id,
                            "gen_main_id": gen_main_id,
                            "classification_verification": Verifier.classification_label_for_verif(
                                split_idx, classification
                            ),
                            "blank_verified": None,
                            "blank": False,
                        }

                    mapping[clean_stitched_id] = entry
                continue

            # ---------- CASE C: single stitched or regular ----------
            if not all_files:
                continue

            found_count += 1
            if stitched_flag == "Stitched":
                stitched_count += 1

            if stitched_flag == "Stitched":
                focus_idx = -1
                final = chosen
                actual_z = extract_z_level(chosen.name)
            else:
                idx = find_best_focus(all_files)
                focus_idx = idx if 0 <= idx < len(all_files) else -1
                final = all_files[focus_idx] if focus_idx >= 0 else chosen
                actual_z = extract_z_level(final.name)


            classification = classify_image_file(final.name)

            entry = {
                "dayID": day_id,
                "BA": ba_str,
                "wellID": well_id,
                "Best Z": focus_idx,
                "Best Z Filename": str(self._to_rel(final)),
                "Actual Z Value": actual_z,
                "Classification": classification,
                "um_per_px": float(group_df["um_per_px"].iloc[0]),
                "all_files": [str(self._to_rel(f)) for f in all_files],
                "cellLine": group_df["cellLine"].iloc[0],
                "treatment": group_df["treatment"].iloc[0],
            }

            split_idx = None
            is_presplit = (day_id, batch_plate, well_id) in presplit_wells

            if self.verifier:
                gen_main_id = self.verifier.build_main_id(
                    ba_str, day_id, well_id, None, classification, presplit_flag=is_presplit
                )
                entry["verification"] = self.verifier.lookup(
                    ba_str, day_id, well_id, None, classification, gen_main_id
                )
            else:
                gen_main_id = Verifier.build_main_id(
                    ba_str, day_id, well_id, None, classification, presplit_flag=is_presplit
                )
                entry["verification"] = {
                    "main_id": gen_main_id,
                    "gen_main_id": gen_main_id,
                    "classification_verification": Verifier.classification_label_for_verif(
                        split_idx, classification
                    ),
                    "blank_verified": None,
                    "blank": False,
                }

            mapping[full_id] = entry

        log.info("=== MAPPING SUMMARY ===")
        log.info(f"Total groups processed: {total_groups}")
        log.info(f"Files found: {found_count}")
        log.info(f"Stitched images detected: {stitched_count}")
        if total_groups:
            log.info(f"Success rate: {found_count/total_groups*100:.1f}%")

        wrapped = {
            "_base_folder": str(self.base_dir.resolve()),
            "entries": mapping,
        }
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(wrapped, indent=2))
        log.info(f"[ImageMapper] Wrote mapping JSON to {out_json}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    base_dir = RAW_IMAGE_DATA        # root input with raw image data
    meta_xlsx = META_FILE            # Excel with the 'Images' sheet
    verify_csv = IMAGE_VERIFICATION_FORM  # verification CSV (blank wells)
    out_json = RAW_IMAGE_MAPPING_JSON     # where to write the mapping

    mapper = ImageMapper(
        base_dir=base_dir,
        meta_csv=meta_xlsx,
        verify_csv=verify_csv,
    )
    mapper.make_mapping_json(out_json)
