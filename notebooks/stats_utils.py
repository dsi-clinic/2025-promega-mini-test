"""Utility functions for promega descriptive statistics notebook."""

import collections
import json
import pathlib

import pandas as pd


def load_json(path: str | pathlib.Path) -> dict:
    """Load JSON data from file."""
    with open(path) as f:
        return json.load(f)


def save_results(stats_dict: dict, path: str | pathlib.Path) -> None:
    """Save stats dict to JSON.

    Args:
        stats_dict: Statistics dictionary to save.
        path: Output file path.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Results saved to {path}")


def initialize_stats_dict() -> dict:
    """Initialize stats dictionary.

    Returns:
        Dict of stats to gather
    """
    return {
        "num_organoids": 0,
        "organoid_distr": {},
        "num_records": 0,
        "num_images": 0,
        "image_distr": {},
        "num_masks_manual": 0,
        "masks_manual_distr": {},
        "num_masks_predicted": 0,
        "masks_predicted_distr": {},
        "num_labels": 0,
        "label_distr": {},
        "label_day_distr": {},
        "num_metabolites": 0,
        "metabolite_distr": {},
        "metabolite_day_distr": {},
        "num_surveys": 0,
        "survey_day_distr": {},
        "survey_vote_distr": {},
        "survey_votes_by_day": {},
    }


def count_by_value_and_day(json_data: dict, values_fn) -> tuple[dict, dict]:
    """Count occurrences of values total and per day.

    Args:
        json_data: Full record dict loaded from all_data.json.
        values_fn: Callable that returns a list of values for a record,
                   or an empty list / None if the record should be skipped.

    Returns:
        Tuple of (total_counter, by_day_counter) as plain dicts.
    """
    total = collections.Counter()
    by_day = collections.defaultdict(collections.Counter)
    for record in json_data.values():
        values = values_fn(record)
        if not values:
            continue
        day = record.get("day", {}).get("number")
        for value in values:
            total[value] += 1
            by_day[day][value] += 1
    return dict(total), {day: dict(counts) for day, counts in by_day.items()}


def get_distribution_by_day(json_data: dict, value_fn) -> dict:
    """Count unique values per day using a value extractor function.

    Args:
        json_data: Full record dict loaded from all_data.json.
        value_fn: Callable that extracts the field value from a record.

    Returns:
        Dict mapping day number to count of unique non-null values.
    """
    elements_by_day = collections.defaultdict(set)
    for record in json_data.values():
        day = record.get("day", {}).get("number")
        value = value_fn(record)
        if value is not None:
            elements_by_day[day].add(value)
    return {day: len(elements) for day, elements in elements_by_day.items()}


def print_table(data: dict, col1: str = "Key", col2: str = "Count",
                col1_width: int = 20, col2_width: int = 10) -> None:
    """Print a dict as a left-aligned table."""
    print(f"{col1:<{col1_width}} {col2:>{col2_width}}")
    for key in sorted(data):
        print(f"{key:<{col1_width}} {data[key]:>{col2_width}}")


def to_dataframe(distr_dict: dict, key_col: str, value_col: str = "Count", title: str | None = None):
    """Convert a flat distribution dict to a sorted DataFrame.

    Args:
        distr_dict: Dict mapping keys to counts.
        key_col: Column name for the keys.
        value_col: Column name for the counts.
        title: Optional caption rendered above the table in Jupyter.

    Returns:
        Styler with caption if title is provided, otherwise a plain DataFrame.
    """
    df = pd.DataFrame(list(distr_dict.items()), columns=[key_col, value_col])
    df = df.sort_values(key_col).reset_index(drop=True)
    if title:
        return df.style.set_caption(title)
    return df
