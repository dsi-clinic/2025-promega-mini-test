"""
This script reads the matabolite data and creates an output file
containing the metabolite data for each organoid.
"""

import argparse
import json
import logging
import pathlib

import pandas as pd

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


class MetaboliteError(Exception):
    pass

def get_args() -> argparse.Namespace:
    """Get arguments from the command line.

    Returns:
        args: The arguments
    """
    parser = argparse.ArgumentParser(description='Map metabolite data to organoid identifiers')
    parser.add_argument('--in-file', type=pathlib.Path, help='The CSV file to retrieve main identifiers from')
    parser.add_argument('--identifiers', type=pathlib.Path, help='The file containing the identifiers to map to')
    parser.add_argument('--out-file', type=pathlib.Path, help='The file to save the metabolite data to')
    args = parser.parse_args()
    return args

def get_main_identifiers(indentifier_file: pathlib.Path) -> tuple[list[str], dict[str, list[str]]]:
    """Get main identifiers and split identifiers from a file.

    Args:
        indentifier_file: The file containing the identifiers to map to

    Returns:
        main_identifiers: The main identifiers
        split_identifiers: The split identifiers
    """
    with open(indentifier_file, 'r') as f:
        main_identifiers = json.load(f)
    split_identifiers = {}
    for identifier in main_identifiers:
        if "split" in identifier:
            main_identifier = identifier.replace("split_1", "").replace("split_2", "").strip()
            split_identifiers.setdefault(main_identifier, []).append(identifier)
    return main_identifiers, split_identifiers


def get_metabolite_data(in_file: pathlib.Path) -> pd.DataFrame:
    """Get metabolite data from a file.

    Args:
        in_file: The file containing the metabolite data

    Returns:
        df: The metabolite data
    """
        # Read Excel sheet
    df = pd.read_excel(in_file, sheet_name="Experimental Values")

    # Normalize column names (strip and lowercase for consistency)
    df.columns = [col.strip().lower() for col in df.columns]
    logging.info(f"Loaded {len(df)} rows from {in_file}")

    return df

def get_organoid_id(row: pd.Series) -> str:
    """Get organoid ID from a row of data.

    Args:
        row: The row of data

    Returns:
        organoid_id: The organoid ID
    """
    batch = str(int(row["batch"]))  # e.g. 1
    plate = str(int(row["starting plate"]))  # e.g. 2
    ba = f"BA{batch} 96_{plate}"  # e.g. "BA2 96_1"
    day = f'Dy{int(row["day"]):02d}'  # e.g. "Dy28"
    well = row["96 well"].strip().upper()  # e.g. "A5"

    organoid_id = f"{ba} {day} {well}"
    return organoid_id

def get_assay_dict(row: pd.Series) -> dict:
    """Get assay dictionary from a row of data.

    Args:
        row: The row of data

    Returns:
        assay_dict: The assay dictionary
    """
    assay = row["assay"].strip()
    conc = row.get("concentration um")
    init_conc = row.get("initial  concentration")
    is_outlier = str(row.get("rlu outside 3 stdev")).strip().lower() == "outlier"
    well_384 = row.get("384 well", "").strip().upper()
    assay_dict = {
        assay: {
            "concentration_uM": conc,
            "initial_concentration": init_conc,
            "is_outlier": is_outlier,
            "well_384": well_384
        }
    }
    return assay_dict

def get_missing_identifiers(metabolite_map: dict, main_identifiers: list[str]) -> list[str]:
    """Get missing identifiers from a metabolite map.

    Args:
        metabolite_map: The metabolite map
        main_identifiers: The main identifiers
        split_identifiers: The split identifiers
    """
    identifiers_set = set(main_identifiers)
    metabolite_map_keys = set(metabolite_map.keys())
    missing_metabolite_data = identifiers_set - metabolite_map_keys

    missing_metabolite_data_list = list(missing_metabolite_data)
    missing_metabolite_data_list.sort()
    return missing_metabolite_data_list

def main():
    args = get_args()
    for key, value in vars(args).items(): logging.info(f"  {key}: {value}")

    main_identifiers, split_identifiers = get_main_identifiers(args.identifiers)
    df = get_metabolite_data(args.in_file)

    # Initialize output dict
    metabolite_map = {}
    skipped_organoids = 0
    for _, row in df.iterrows():
        try:
            organoid_id = get_organoid_id(row)
            assay_dict = get_assay_dict(row)

            if organoid_id not in main_identifiers:
                if organoid_id in split_identifiers.keys():
                    for split_identifier in split_identifiers[organoid_id]:
                        metabolite_map.setdefault(split_identifier, {}).update(assay_dict)
                else:
                    logging.debug(f"Organoid ID {organoid_id} not found in identifiers")
                    skipped_organoids += 1
                    continue
            else:
                metabolite_map.setdefault(organoid_id, {}).update(assay_dict)

        except Exception:
            logging.exception("Skipping row due to unknown error")

    # Save the JSON
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(metabolite_map, f, indent=2)

    # Locate missing identifiers
    missing_metabolite_data = get_missing_identifiers(metabolite_map, main_identifiers)

    # Summary of results
    logging.info("Metabolite map saved to: %s (%d entries)", args.out_file, len(metabolite_map))
    logging.info("Located a total of %d identifiers with metabolite data", len(metabolite_map))
    logging.warning("Skipped %d organoids due to missing identifiers", skipped_organoids)
    if missing_metabolite_data:
        logging.warning("Found %d identifiers with no metabolite data", len(missing_metabolite_data))
        logging.info("Sample of missing identifiers: %s", list(missing_metabolite_data)[:10])


if __name__ == "__main__":
    main()