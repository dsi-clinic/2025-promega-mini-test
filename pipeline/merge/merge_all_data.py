#!/usr/bin/env python3
"""Step 16: merge identifiers, images, surveys, and metabolites into all_data.json.

Thin orchestrator. The work lives in sibling modules:
    cli.py        — Config dataclass + argument parser
    loaders.py    — load_data_sources + per-source readers + path verification
    merge.py      — merge_data_sources, build_normalized_records, label propagation,
                    sanitize_for_json, write_json, print_stats, extract_mdl_day
    validation.py — validate_data + validate_json (defers to validate_schema)
"""

import logging

from rich.logging import RichHandler

from pipeline.merge.cli import get_args
from pipeline.merge.loaders import load_data_sources
from pipeline.merge.merge import (
    build_normalized_records,
    extract_mdl_day,  # re-exported: filter_complete_series imports from here
    merge_data_sources,
    print_stats,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    handlers=[RichHandler()],
)

__all__ = ["extract_mdl_day", "main"]


def main() -> None:
    cfg = get_args()
    sources = load_data_sources(cfg)

    logging.info("Merging data sources...")
    combined = merge_data_sources(sources)

    logging.info("Normalizing merged records...")
    _, stats = build_normalized_records(cfg, combined, sources.image_meta)

    print_stats(stats, cfg.out_file, cfg.no_validate)


if __name__ == "__main__":
    main()
