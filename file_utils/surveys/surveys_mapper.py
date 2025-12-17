"""
Map survey data to organoid identifiers.
"""

import argparse
import collections
import json
import logging
import pathlib
import re
import typing

import pandas as pd

from file_utils.common.organoid_patterns import OrganoidNormalizer, clean_id_for_json


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

# Constants
DAY_20_21 = 20.5
LABEL_MAP = {"Accepted": 1, "Not Accepted": 0, "Acceptable": 1, "Not Acceptable": 0}

def get_args() -> argparse.Namespace:
    """Get arguments from the command line.

    Returns:
        args: The arguments
    """
    parser = argparse.ArgumentParser(description='Map survey data to organoid identifiers')
    parser.add_argument('--in-dir', type=pathlib.Path, help='The directory containing the survey results')
    parser.add_argument('--identifiers', type=pathlib.Path, help='The file containing the identifiers to map to')
    parser.add_argument('--out-file', type=pathlib.Path, help='The file to save the formatted survey data to')
    parser.add_argument('--min-survey-votes', type=int, default=4, help='The minimum number of votes required to determine a majority')
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

def is_organoid_cell_value(val) -> bool:
    """Check if a cell value contains organoid data.

    Args:
        val: The cell value to check

    Returns:
        bool: True if the value contains organoid identifiers
    """
    return pd.notna(val) and isinstance(val, str) and (
        "Organoid_" in val or any(x in val for x in ["Ba1", "Ba2", "Ba3", "Ba4", "Dy"])
    )


def extract_organoid_ids(cell_value: str) -> tuple[str | None, str | None]:
    """Extract organoid_id and image_id from a cell value.

    Args:
        cell_value: The cell value containing organoid data

    Returns:
        tuple: (organoid_id, image_id) or (None, None) if not found
    """
    parts = [p.strip() for p in cell_value.split(",")]
    organoid_id = next((p for p in parts if "Organoid_" in p), None)
    image_id = next((p for p in parts if any(x in p for x in ["Ba1", "Ba2", "Ba3", "Ba4", "Dy"])), None)
    return organoid_id, image_id


def process_image_id(image_id: str) -> tuple[str, str, int | None, dict]:
    """Process an image_id to extract metadata and construct identifiers.

    Args:
        image_id: The raw image identifier

    Returns:
        tuple: (record_id, main_id, split_index, parsed_meta)
    """
    is_stitched = determine_stitched(image_id)
    split_info = OrganoidNormalizer.extract_split_info(image_id) or {}
    split_index = split_info.get("split_index", None)
    record_id, main_id = construct_identifiers(image_id, split_index, is_stitched)
    parsed_meta = parse_image_id(image_id)
    return record_id, main_id, split_index, parsed_meta


def build_entry(
    original_cell: str,
    organoid_id: str,
    record_id: str,
    main_id: str,
    split_index: int | None,
    parsed_meta: dict,
    source_file: str,
) -> dict:
    """Build an organoid entry dictionary.

    Args:
        original_cell: The original Excel cell text
        organoid_id: The raw organoid identifier
        record_id: The cleaned record identifier
        main_id: The main identifier for matching
        split_index: The split index if applicable
        parsed_meta: Parsed metadata (batch, plate, day, well)
        source_file: The source file name

    Returns:
        dict: The entry dictionary
    """
    return {
        "original_image_ref": original_cell,
        "raw_organoid_id": organoid_id,
        "image_id": record_id,
        "main_id": main_id,
        "split_index": split_index,
        "source_file": source_file,
        **parsed_meta,
    }


def categorize_entry(
    entry: dict,
    parts: list[str],
    is_quality_form: bool,
    employee_name: str | None,
) -> tuple[str | None, dict]:
    """Categorize an entry and add category-specific fields.

    Args:
        entry: The entry dictionary to categorize
        parts: The parsed parts from the cell value
        is_quality_form: Whether this is a quality form
        employee_name: The employee name (for evaluation forms)

    Returns:
        tuple: (entry_type, entry) or (None, entry) if no category matches
    """
    if is_quality_form and any(q in parts for q in ["Good", "Bad", "Reasonable"]):
        entry["quality"] = next(p for p in parts if p in ["Good", "Bad", "Reasonable"])
        return "quality_scores", entry

    elif not is_quality_form and any(e in parts for e in ["Acceptable", "Not Acceptable", "Not Loaded"]):
        entry["evaluation"] = next(p for p in parts if p in ["Acceptable", "Not Acceptable", "Not Loaded"])
        entry["employee"] = employee_name
        return "evaluations", entry

    return None, entry


def process_excel_file(file: str) -> typing.Generator[tuple[str, str, dict], None, None]:
    """Process an excel file and yield entries.

    Args:
        file: The file to process

    Yields:
        tuple: (record_id, entry_type, entry) where entry_type is either "quality_scores" or "evaluations"
    """
    is_quality_form = "Image Classification" in pathlib.Path(file).name
    basename = pathlib.Path(file).name
    df = pd.read_excel(file)

    for _, row in df.iterrows():
        employee_name = (
            f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
            if not is_quality_form else None
        )

        for col in row.index:
            val = row[col]
            if not is_organoid_cell_value(val):
                continue

            organoid_id, image_id = extract_organoid_ids(val)
            if not organoid_id or not image_id:
                continue

            record_id, main_id, split_index, parsed_meta = process_image_id(image_id)
            if not parsed_meta:
                logging.warning(f"Could not parse image_id: {image_id} from {organoid_id} in {basename}")

            entry = build_entry(val, organoid_id, record_id, main_id, split_index, parsed_meta, basename)

            parts = [p.strip() for p in val.split(",")]
            entry_type, categorized_entry = categorize_entry(entry, parts, is_quality_form, employee_name)

            if entry_type:
                yield (record_id, entry_type, categorized_entry)


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
        day = record_id.split(" ")[2]
        if day == "Dy20" or day == "Dy21":
            day = f"Dy{DAY_20_21}"
            record_id = record_id.replace(day, f"Dy{DAY_20_21}")

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
    # Also remove parenthetical annotations like "(stitched)"
    image_id_core = re.sub(r"\(.*?\)", "", image_id_core)
    image_id_core = re.sub(r"\s+", " ", image_id_core).strip()
    image_id_clean = clean_id_for_json(image_id_core)

    # Capitalize Batch identifier
    image_id_clean = image_id_clean.replace("Ba", "BA")
    return image_id_clean


def parse_image_id(image_id):
    """Parse the image id into BA/day/well.

    Args:
        image_id: The image id

    Returns:
        dict: The parsed image id
    """
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


def compute_survey_majority(evaluations: list[dict], min_survey_votes: int = 4) -> dict:
    """Compute the majority vote label for a list of evaluations.

    Args:
        evaluations: The list of evaluations

    Returns:
        dict: The majority vote label
    """
    inv_votes = collections.Counter()
    reg_votes = collections.Counter()
    for eval_entry in evaluations:
        vote = eval_entry.get("evaluation")
        if vote:
            original_image_ref = eval_entry.get("original_image_ref")
            if "INV" in original_image_ref:
                inv_votes[vote] += 1
            else:
                reg_votes[vote] += 1

    winning_inv_label = next(
        (label for label, count in inv_votes.items() if count >= min_survey_votes),
        None,
    )

    winning_reg_label = next(
        (label for label, count in reg_votes.items() if count >= min_survey_votes),
        None,
    )

    if inv_votes and inv_votes[winning_inv_label] != reg_votes[winning_reg_label]:
        main_id = evaluations[0].get("image_id")
        logging.warning(f"{main_id}:  Inverted evaluation - {inv_votes[winning_inv_label]} '{winning_inv_label}' does not match regular evaluation - {reg_votes[winning_reg_label]} '{winning_reg_label}'")
        winning_reg_label = None

    total = sum(inv_votes.values()) + sum(reg_votes.values())

    return {
        "value": winning_reg_label,
        "acceptance_flag": LABEL_MAP.get(winning_reg_label) if winning_reg_label else None,
        "votes": dict(reg_votes + inv_votes),
        "total_evaluations": total,
        "min_votes": min_survey_votes,
        "source": "survey.evaluations",
    }


def process_organoid_files(directory, identifiers_file: pathlib.Path, min_survey_votes: int = 4) -> dict:
    """Process the organoid files.

    Args:
        directory: The directory containing the organoid files
        identifiers_file: The file containing the identifiers to map to
        min_survey_votes: The minimum number of votes required to determine a majority
    Returns:
        data: The data dictionary
    """
    excel_files = get_excel_files(directory)
    logging.info("Total excel files found: %d", len(excel_files))

    with open(identifiers_file, "r") as f:
        identifiers = json.load(f)
    logging.info("Total identifiers found: %d", len(identifiers))

    # Nested defaultdict: record_id -> organoid_id -> {"evaluations": [], "quality_scores": []}
    data = collections.defaultdict(lambda: {"evaluations": [], "quality_scores": []})
    for file in excel_files:
        logging.info("Processing file: %s", file)
        try:
            for record_id, entry_type, entry in process_excel_file(file):
                if record_id not in identifiers:
                    logging.warning(f"Identifier {record_id} not found in identifiers")
                    continue
                data[record_id][entry_type].append(entry)
        except Exception as e:
            logging.exception(f" Error processing file {file}: {e}")
            continue

    # Compute majority vote label for each record
    total_votes = 0
    for record_id, entries in data.items():
        majority = compute_survey_majority(entries["evaluations"], min_survey_votes)
        data[record_id]["label"] = majority
        total_votes += majority["total_evaluations"]
    logging.info("Total votes: %d", total_votes)

    return data


def main():
    args = get_args()
    for key, value in vars(args).items(): logging.info(f"  {key}: {value}")

    data = process_organoid_files(args.in_dir, args.identifiers, args.min_survey_votes)
    logging.info("Final organoid count: %d", len(data))

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f" Wrote: {args.out_file}")


if __name__ == "__main__":
    main()
