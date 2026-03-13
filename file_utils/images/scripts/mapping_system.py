#!/usr/bin/env python3
import sys
import json
import logging
from pathlib import Path

# --- Locate repo root (needs config.py only) ---
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py")

import config as P
from .image_mapper import ImageMapper


class MappingSystem:
    """Wrapper to initialize ImageMapper and build the mapping JSON."""

    def __init__(self):
        logging.info(f"RAW_IMAGE_DATA          = {P.RAW_IMAGE_DATA}")
        logging.info(
            f"META_FILE               = {P.META_FILE} (exists={P.META_FILE.exists()})"
        )
        logging.info(f"RAW_IMAGE_MAPPING_JSON  = {P.RAW_IMAGE_MAPPING_JSON}")

        # Optional verification CSV (for blank annotations)
        verify_csv = getattr(P, "IMAGE_VERIFICATION_FORM", None)
        if verify_csv:
            logging.info(f"IMAGE_VERIFICATION_FORM = {verify_csv}")
        else:
            logging.info(
                "IMAGE_VERIFICATION_FORM not provided; skipping blank verification overlay."
            )

        # Initialize ImageMapper
        self.mapper = ImageMapper(
            base_dir=P.RAW_IMAGE_DATA,
            meta_csv=P.META_FILE,
            verify_csv=verify_csv if verify_csv else None,
        )

        self.out_json = P.RAW_IMAGE_MAPPING_JSON

    def generate_key_mapping(self) -> dict:
        """Generate the key-mapping JSON and return the wrapped dict."""
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.mapper.make_mapping_json(self.out_json)

        # Read it back (wrapped with _base_folder + entries)
        with self.out_json.open() as f:
            mapping = json.load(f)

        n_entries = len(mapping.get("entries", {}))
        logging.info(f"Mapping complete: {n_entries} entries → {self.out_json}")
        return mapping


if __name__ == "__main__":
    logging.debug("Logger alive?")  # quick sanity
    MappingSystem().generate_key_mapping()
