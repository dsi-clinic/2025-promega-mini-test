"""Module for loading JSON mapping from a file using environment variables."""

import json
import logging
import os
from pathlib import Path

from config import ORIGINAL_MAPPING
JSON_MAPPING_PATH = ORIGINAL_MAPPING


def load_json_mapping(json_path: str | Path) -> dict:
    """Loads the JSON mapping file and returns it as a dictionary.

    Args:
        json_path (str | Path): Path to the JSON file.

    Returns:
        dict: JSON data as a dictionary (empty if file is missing or corrupted).
    """
    json_path = Path(json_path)

    # Check if the file exists
    if not json_path.exists():
        logging.error(f"JSON mapping file not found: {json_path}")
        return {}

    try:
        with json_path.open() as f:
            data = json.load(f)
            logging.info(f"Successfully loaded JSON mapping from {json_path}")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {json_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading JSON {json_path}: {e}")

    return {}  # Return empty dictionary if loading fails
