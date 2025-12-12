# image_mapper.py
from __future__ import annotations
import argparse
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

logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.DEBUG)

class ImageMapper:
    def __init__(self, base_dir: pathlib.Path, meta_csv: pathlib.Path, verify_csv: pathlib.Path | None = None):
        self.base_dir = pathlib.Path(base_dir)
        self.meta_csv = pathlib.Path(meta_csv)
        self.verifier = Verifier(verify_csv) if verify_csv else None

    def _compute_presplit_wells(self, grouped) -> set[Tuple[str, str, str]]:
        """
        Determine (dayID, batchPlate, wellID) combos that occur BEFORE the first split
        for that (batchPlate, wellID).

        grouped: DataFrameGroupBy over ["dayID", "batchPlate", "wellID"] (your current call site)
        Returns: set of tuples (dayID, batchPlate, wellID) that are presplit.
        """

        # --- 1) Materialize the group keys into a small DataFrame (fast; no per-row filesystem work)
        keys = [(day_id, batch_plate, well_id) for (day_id, batch_plate, well_id), _ in grouped]
        if not keys:
            logging.info("[ImageMapper] Detected 0 presplit wells")
            return set()

        df_keys = pd.DataFrame(keys, columns=["dayID", "batchPlate", "wellID"])
        df_keys["dnum"] = df_keys["dayID"].apply(OrganoidNormalizer.extract_day_number)

        # --- 2) Scan image files ONCE, compute which wells ever have split images
        has_split_by_well = defaultdict(bool)
        img_folder = self.base_dir

        if img_folder.exists():
            for f in list_image_files(img_folder):
                w = OrganoidNormalizer.extract_well(f.name)
                if not has_split_by_well[w]:  # short-circuit once True
                    if OrganoidNormalizer.extract_split_info(f.name)["is_split"]:
                        has_split_by_well[w] = True

        # If split detection actually depends on batchPlate too, change the key above to (batchPlate, well)
        df_keys["well_has_split"] = df_keys["wellID"].map(has_split_by_well).fillna(False)


        # --- 3) First split day per (batchPlate, wellID)
        first_split = (
            df_keys.loc[df_keys["well_has_split"]]
                    .groupby(["batchPlate", "wellID"], as_index=False)["dnum"]
                    .min()
                    .rename(columns={"dnum": "first_split"})
        )

        if first_split.empty:
            logging.info("[ImageMapper] Detected 0 presplit wells")
            return set()

        # --- 4) Mark presplit rows: day number < first_split day for that (batchPlate, wellID)
        out = df_keys.merge(first_split, on=["batchPlate", "wellID"], how="inner")
        out = out[out["dnum"] < out["first_split"]]

        presplit_wells = {
            (f"Dy{int(d):02d}", bp, w)
            for d, bp, w in out[["dnum", "batchPlate", "wellID"]].itertuples(index=False, name=None)
        }

        logging.info(f"[ImageMapper] Detected {len(presplit_wells)} presplit wells")
        return presplit_wells

    def _to_rel(self, p: pathlib.Path) -> pathlib.Path:
        p = pathlib.Path(p)
        try:
            return p.relative_to(self.base_dir)
        except ValueError:
            return pathlib.Path(os.path.relpath(p, self.base_dir))

    def make_mapping_json(self, out_json: pathlib.Path) -> None:
        cleaned = load_and_clean_metadata(self.meta_csv)
        grouped = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        presplit_wells = self._compute_presplit_wells(grouped)

        mapping: Dict[str, Dict[str, Any]] = {}

        total_groups = len(grouped)
        stitched_count = 0
        found_count = 0

        logging.info(f"[ImageMapper] Processing {total_groups} unique (day,batchPlate,well) groups")

        logging_level = logging.getLevelName(logging.getLogger().level)
        iterator = grouped if logging_level == "DEBUG" else tqdm(grouped, desc="Processing groups")

        for (day_id, batch_plate, well_id), group_df in iterator:
            fc, sc = self.process_batches(mapping, batch_plate, day_id, well_id, presplit_wells, group_df)
            found_count += fc
            stitched_count += sc

        logging.info("=== MAPPING SUMMARY ===")
        logging.info(f"Total groups processed: {total_groups}")
        logging.info(f"Files found: {found_count}")
        logging.info(f"Stitched images detected: {stitched_count}")
        if total_groups:
            logging.info(f"Success rate: {found_count/total_groups*100:.1f}%")

        wrapped = {
            "_base_folder": str(self.base_dir.resolve()),
            "entries": mapping,
        }
        out_json = pathlib.Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(wrapped, indent=2))
        logging.info(f"[ImageMapper] Wrote mapping JSON to {out_json}")

    def process_batches(self, mapping: dict, batch_plate: str, day_id: str, well_id: str, presplit_wells: set[Tuple[str, str, str]], group_df: pd.DataFrame):
        """Process batch plate, day, well group and add to mapping dictionary

        Args:
            mapping: dictionary to add entries to
            batch_plate: batch plate string
            day_id: day identifier
            well_id: well identifier
            presplit_wells: set of presplit wells
            group_df: dataframe containing group data
        """
        found_count = 0
        stitched_count = 0

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
            return 0, 0

        # regroup on all_files (always every match)
        split_groups_all = group_by_split(all_files)
        child_groups = {k: v for k, v in split_groups_all.items() if k is not None}

        # ---------- CASE A: split children ----------
        if len(child_groups) >= 1:
            logging.debug(f"[ImageMapper] Expanding into {len(child_groups)} split children for {full_id}")
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

            return found_count, stitched_count # done with this (day,ba,well) group

        # ---------- CASE B: multiple stitched groups ----------
        if stitched_flag == "SplitAmbiguous" and stitched_groups:
            logging.debug(f"[ImageMapper] Processing {len(stitched_groups)} stitched groups for {full_id}")
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
            return found_count, stitched_count

        # ---------- CASE C: single stitched or regular ----------
        if not all_files:
            return 0, 0

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
        return found_count, stitched_count

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

def get_args() -> argparse.Namespace:
    """Get arguments from the command line.

    Returns:
        args: The arguments
    """
    parser = argparse.ArgumentParser(description='Map raw images to organoid identifiers')
    parser.add_argument('--base-dir', type=pathlib.Path, help='The base directory containing the raw images')
    parser.add_argument('--verify-csv', type=pathlib.Path, help='The file containing the image verification data')
    parser.add_argument('--meta-xlsx', type=pathlib.Path, help='The file containing the metadata')
    parser.add_argument('--out-file', type=pathlib.Path, help='The file to save the formatted image mapping to')
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    for key, value in vars(args).items(): logging.info(f"  {key}: {value}")

    mapper = ImageMapper(
        base_dir=args.base_dir,
        meta_csv=args.meta_xlsx,
        verify_csv=args.verify_csv,
    )
    mapper.make_mapping_json(args.out_file)

if __name__ == "__main__":
    main()
