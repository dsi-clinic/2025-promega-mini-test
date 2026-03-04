#!/usr/bin/env python3
# Split the *training* mapping into train/val/test (and optional early/late)

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

# --- locate repo root (has paths.py + .env) ---
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

# Constants
DEFAULT_TARGET_WIDTH = 512
DEFAULT_TARGET_HEIGHT = 384
EXPECTED_RECORDS_NUM = 5168    # This is the number of resized imgs records
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_VAL_FRACTION = 0.1
EARLY_DAYS = {"Dy03", "Dy06", "Dy08", "Dy10"}
LATE_DAYS  = {"Dy13", "Dy15", "Dy17", "Dy20", "Dy21", "Dy24", "Dy28", "Dy30"}
# Known-bad entries to drop (BA, dayID, wellID)
BAD_ENTRIES = {
    ("BA1 96_1", "Dy30", "A4"),
    ("BA1 96_1", "Dy17", "A7"),
    ("BA1 96_1", "Dy17", "A8"),
    ("BA2 96_1", "Dy30", "C1"),
    ("BA2 96_1", "Dy30", "D10"),
    ("BA2 96_1", "Dy30", "E8"),
    ("BA2 96_1", "Dy28", "D10"),
    ("BA2 96_1", "Dy28", "E7"),
    ("BA2 96_1", "Dy21", "E7"),
    ("BA2 96_2", "Dy30", "A7"),
    ("BA2 96_2", "Dy30", "B8"),
    ("BA2 96_2", "Dy30", "D5"),
    ("BA2 96_2", "Dy30", "D7"),
    ("BA2 96_2", "Dy30", "D12"),
    ("BA2 96_2", "Dy30", "E3"),
    ("BA2 96_2", "Dy28", "B3"),
    ("BA2 96_2", "Dy28", "B8"),
    ("BA2 96_2", "Dy28", "D5"),
    ("BA2 96_2", "Dy28", "E3"),
    ("BA2 96_2", "Dy28", "E12"),
    ("BA2 96_2", "Dy08", "D5"),
}

def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for splitting training data.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - resized_json: Path to the resized image mapping JSON file
            - train_frac: Fraction of data to use for training (default: 0.8)
            - val_frac: Fraction of data to use for validation (default: 0.1)
            - splits_dir: Path to the output splits directory
            - split_days: Whether to split data by early/late days
            - target_width: Target width of images/masks (default: 512)
            - target_height: Target height of images/masks (default: 384)

    Raises:
        SystemExit: If --resized-json or --splits-dir is not provided
    """
    parser = argparse.ArgumentParser(
        description='Map manual masks to image mapping JSON'
    )
    parser.add_argument(
        '--resized-json',
        type=Path,
        help='Path to the resized image mapping JSON file'
    )
    parser.add_argument(
        '--train-frac',
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help='Fraction of data to use for training'
    )
    parser.add_argument(
        '--val-frac',
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help='Fraction of data to use for validation'
    )
    parser.add_argument(
        '--splits-dir',
        type=Path,
        default=None,
        help='Path to the output splits directory'
    )
    parser.add_argument(
        '--split-days',
        action='store_true',
        help='Split the data by days'
    )
    parser.add_argument(
        '--target-width',
        type=int,
        default=DEFAULT_TARGET_WIDTH,
        help='Target width of the images/masks (pixels)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=DEFAULT_TARGET_HEIGHT,
        help='Target height of the images/masks (pixels)'
    )
    args = parser.parse_args()

    # Validate required paths
    if not args.resized_json:
        parser.error("--resized-json is required")
    if not args.splits_dir:
        parser.error("--splits-dir is required")

    return args

def load_and_filter(mapping_path: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Load mapping JSON file and filter out known bad entries.

    Removes entries that match any (BA, dayID, wellID) tuple in BAD_ENTRIES.

    Args:
        mapping_path: Path to the mapping JSON file to load.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], int]: Tuple containing:
            - Filtered mapping dictionary (keyed by image ID)
            - Number of bad entries removed

    Raises:
        FileNotFoundError: If the mapping file doesn't exist.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    with open(mapping_path, "r") as f:
        full_map = json.load(f)
    full_map = full_map.get("entries", {})
    before = len(full_map)
    full_map = {k: v for k, v in full_map.items() if (v.get("BA"), v.get("dayID"), v.get("wellID")) not in BAD_ENTRIES}
    after = len(full_map)
    logging.info("Removed %d bad entries; %d remain.", before - after, after)
    return full_map, before - after

def split_and_save(
    full_map: Dict[str, Dict[str, Any]],
    out_prefix: str,
    split_dir: Path,
    train_frac: float,
    val_frac: float
) -> int:
    """
    Split mapping dictionary into train/val/test sets and save to JSON files.

    Randomly shuffles entries (with fixed seed for reproducibility) and splits
    them according to the specified fractions. Remaining entries go to test set.

    Args:
        full_map: Dictionary of mapping entries to split.
        out_prefix: Prefix for output filenames (e.g., "mapping_processed_total_512x384").
        split_dir: Directory to save split JSON files.
        train_frac: Fraction of data for training (e.g., 0.8 for 80%).
        val_frac: Fraction of data for validation (e.g., 0.1 for 10%).

    Returns:
        int: Total number of entries across all splits (should equal len(full_map)).

    Note:
        Uses random seed 42 for reproducibility. Test set gets remaining entries
        after train and val splits (1 - train_frac - val_frac).
    """
    keys = list(full_map.keys())
    random.seed(42)
    random.shuffle(keys)

    n = len(keys)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    splits = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }

    n_total = 0
    for name, klist in splits.items():
        out_map = {k: full_map[k] for k in klist}
        out_file = split_dir / f"{out_prefix}_{name}.json"
        with open(out_file, "w") as f:
            json.dump(out_map, f, indent=2)
        logging.info("Wrote %d entries → %s", len(klist), out_file)
        n_total += len(klist)

    return n_total

def main() -> None:
    """
    Main entry point for splitting training data into train/val/test sets.

    Orchestrates the complete workflow:
    1. Parse command-line arguments
    2. Load and filter mapping JSON (remove bad entries)
    3. Split full mapping into train/val/test sets
    4. Optionally split by early/late days if --split-days is specified
    5. Validate that all records are accounted for

    Raises:
        FileNotFoundError: If the resized JSON mapping file doesn't exist.
        AssertionError: If record counts don't match expected totals.
    """
    args = get_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(args.resized_json)
    logging.info("Using training mapping: %s", mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    full_map, bad_entries = load_and_filter(mapping_path)
    split_dir = args.splits_dir
    split_dir.mkdir(parents=True, exist_ok=True)


    base = mapping_path.stem
    logging.info("[Full Mapping] Splitting: %s", base)
    n_train = split_and_save(full_map, base, split_dir, args.train_frac, args.val_frac)

    # Validate full mapping split
    assert n_train + bad_entries == EXPECTED_RECORDS_NUM, \
        f"Expected {EXPECTED_RECORDS_NUM} records, got {n_train + bad_entries}"

    if args.split_days:
        early_map = {k: v for k, v in full_map.items() if v.get("dayID") in EARLY_DAYS}
        late_map = {k: v for k, v in full_map.items() if v.get("dayID") in LATE_DAYS}

        logging.info("[Early Days] Splitting %d entries", len(early_map))
        n_early = split_and_save(early_map, "mapping_days0310", split_dir, args.train_frac, args.val_frac)

        logging.info("[Late Days] Splitting %d entries", len(late_map))
        n_late = split_and_save(late_map, "mapping_days1330", split_dir, args.train_frac, args.val_frac)

        # Validate day-based splits
        assert n_early + n_late + bad_entries == EXPECTED_RECORDS_NUM, \
            f"Expected {EXPECTED_RECORDS_NUM} records, got {n_early + n_late + bad_entries}"

if __name__ == "__main__":
    main()
