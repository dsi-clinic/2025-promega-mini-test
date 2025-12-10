"""
Map survey data to organoid identifiers.
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import re
import sys

from matplotlib import image
import pandas as pd

from file_utils.common.organoid_patterns import OrganoidNormalizer, clean_id_for_json


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


def get_args() -> argparse.Namespace:
    """Get arguments from the command line.

    Returns:
        args: The arguments
    """
    parser = argparse.ArgumentParser(description='Map survey data to organoid identifiers')
    parser.add_argument('--in-dir', type=pathlib.Path, help='The directory containing the survey results')
    parser.add_argument('--identifiers', type=pathlib.Path, help='The file containing the identifiers to map to')
    parser.add_argument('--out-file', type=pathlib.Path, help='The file to save the formatted survey data to')
    args = parser.parse_args()
    return args

def get_excel_files(directory: pathlib.Path) -> list[str]:
    """Get the excel files from the directory.

    Args:
        directory: The directory containing the survey results

    Returns:
        excel_files: The excel files
    """
    excel_files = [
        str(f) for f in pathlib.Path(directory).glob("*.xlsx")
        if ("Organoid Classification" in f.name or "Image Classification" in f.name)
        and "Organoid Classification (Form ABC)" not in f.name
    ]
    excel_files.sort()
    return excel_files

def process_excel_file(file: str, data: dict):
    """Process an excel file.

    Args:
        file: The file to process
        data: The data to update
    """
    is_quality_form = "Image Classification" in pathlib.Path(file).name
    basename = pathlib.Path(file).name
    df = pd.read_excel(file)

    for _, row in df.iterrows():
        employee_name = (
            f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
            if not is_quality_form else None
        )
        for col in row.index:    # Locate row,column index with organoid id data
            val = row[col]
            if pd.notna(val) and isinstance(val, str) and (
                "Organoid_" in val or any(x in val for x in ["Ba1", "Ba2", "Ba3", "Ba4", "Dy"])
            ):

                original_cell = val
                parts = [p.strip() for p in val.split(",")]

                organoid_id = next((p for p in parts if "Organoid_" in p), None)
                image_id = next((p for p in parts if any(x in p for x in ["Ba1","Ba2","Ba3","Ba4","Dy"])), None)

                if not organoid_id or not image_id:
                    continue

                # Detect and strip extra tokens (INV, STITCHED, PRE/POST, etc.)
                is_stitched = determine_stitched(image_id)

                # Detect split (safe even if none)
                split_info = OrganoidNormalizer.extract_split_info(image_id) or {}
                split_index = split_info.get("split_index", None)

                # Construct redcord_id and main_id mirroring all_data naming
                record_id, main_id = construct_identifiers(image_id, split_index, is_stitched)

                # Get batch, plate, day, well from image_id
                parsed_meta = parse_image_id(image_id)
                if parsed_meta == {}:
                    logging.warning(f" Could not parse image_id: {image_id} from {organoid_id} in {basename}")

                # Build organoid entry
                entry = {
                    "original_image_ref": original_cell,  # exact Excel cell text
                    "raw_organoid_id": organoid_id,
                    "image_id": record_id,           # base cleaned form
                    "main_id": main_id,                   # used for matching all_data
                    "split_index": split_index,
                    "source_file": basename,
                    **parsed_meta
                }

                # Assign to survey category
                if is_quality_form and any(q in parts for q in ["Good", "Bad", "Reasonable"]):
                    entry["quality"] = next(p for p in parts if p in ["Good", "Bad", "Reasonable"])
                    data[record_id]["quality_scores"].append(entry)

                elif not is_quality_form and any(e in parts for e in ["Acceptable", "Not Acceptable", "Not Loaded"]):
                    entry["evaluation"] = next(p for p in parts if p in ["Acceptable", "Not Acceptable", "Not Loaded"])
                    entry["employee"] = employee_name
                    data[record_id]["evaluations"].append(entry)

def determine_stitched(image_id: str) -> bool:
    """Determine if the image is stitched.

    Args:
        image_id: The image id

    Returns:
        stitched: True if the image is stitched, False otherwise
    """
    is_stitched = False
    if image_id:
        # detect stitch markers first
        if re.search(r"stitched|stitch", image_id, flags=re.IGNORECASE):
            is_stitched = True
    else:
        is_stitched = False

    return is_stitched

def construct_identifiers(image_id: str, split_index: int | None, is_stitched: bool) -> tuple[str, str]:
    """Construct the record_id and main_id.

    Args:
        image_id: The image id

    Returns:
        record_id: The record id
        main_id: The main id
    """
    # Strip unwanted suffixes like INV, PRE, POST, STITCH, etc.
    image_id_clean = clean_image_id(image_id)

    # Record identifier
    record_id = image_id_clean
    if split_index is not None:
        record_id = f"{record_id} split_{split_index}"

    # Main image identifier
    main_id_base = re.sub(r"\s+", "_", image_id_clean.strip()) if image_id_clean else None
    if main_id_base:
        if split_index is not None:
            main_id = f"{main_id_base}_split{split_index}_{'stitched' if is_stitched else 'nostitch'}"
        else:
            main_id = f"{main_id_base}_nosplit_{'stitched' if is_stitched else 'nostitch'}"
    else:
        main_id = None

    return record_id, main_id

def clean_image_id(image_id: str) -> str:
    """Clean the image id.

    Args:
        image_id: The image id

    Returns:
        image_id_clean: The cleaned image id
    """
    # Remove common suffix tokens
    image_id_core = re.sub(
        r"\b(INV|PRE|POST|STITCH|STITCHED|STCH|Z\d+|REV|BOT|TOP|ROI)\b",
        "",
        image_id,
        flags=re.IGNORECASE,
    )
    # also remove parenthetical annotations like "(stitched)"
    image_id_core = re.sub(r"\(.*?\)", "", image_id_core)
    image_id_core = re.sub(r"\s+", " ", image_id_core).strip()
    image_id_clean = clean_id_for_json(image_id_core)

    return image_id_clean

# --- Parse image_id into BA/day/well ---
def parse_image_id(image_id):
    cleaned = re.sub(r"\(.*?\)", "", image_id)                 # remove parentheses
    cleaned = re.sub(r"[^A-Za-z0-9\s_]", " ", cleaned)         # replace junk chars with space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()             # normalize whitespace
    parts = cleaned.split()
    try:
        ba_idx = next(i for i, p in enumerate(parts) if re.match(r"Ba\d+", p, re.IGNORECASE))
        ba = parts[ba_idx].upper()
        plate = parts[ba_idx + 1] if ba_idx + 1 < len(parts) and re.match(r"\d+_\d+", parts[ba_idx + 1]) else ""
        dy = next(p for p in parts if re.match(r"Dy\d+", p, re.IGNORECASE))
        well = next(p for p in parts if re.match(r"^[A-H]\d{1,2}$", p))
        return {"BA": f"{ba} {plate}".strip(), "dayID": dy, "wellID": well}
    except (IndexError, StopIteration):
        return {}


# --- Main processor ---
def process_organoid_files(directory):
    excel_files = get_excel_files(directory)
    logging.info("Total excel files found: %d", len(excel_files))

    data = collections.defaultdict(lambda: {"evaluations": [], "quality_scores": []})
    for file in excel_files:
        logging.info("Processing file: %s", file)

        is_quality_form = "Image Classification" in pathlib.Path(file).name
        basename = pathlib.Path(file).name

        try:
            process_excel_file(file, data)
        except Exception as e:
            logging.exception(f" Error processing file {file}: {e}")
            continue

    return data

def main():
    args = get_args()
    data = process_organoid_files(args.in_dir)
    logging.info("Final organoid count: %d", len(data))

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f" Wrote: {args.out_file}")

# --- Run ---
if __name__ == "__main__":
    main()
