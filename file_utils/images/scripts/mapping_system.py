#!/usr/bin/env python3
import json
import logging
from pathlib import Path
import sys

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MappingSystem:
    def __init__(self):
        logging.info(f"RAW_IMAGE_DATA          = {P.RAW_IMAGE_DATA}")
        logging.info(f"META_FILE               = {P.META_FILE} (exists={P.META_FILE.exists()})")
        logging.info(f"RAW_IMAGE_MAPPING_JSON  = {P.RAW_IMAGE_MAPPING_JSON}")

        self.mapper = ImageMapper(base_dir=P.RAW_IMAGE_DATA, meta_csv=P.META_FILE)
        self.out_json = P.RAW_IMAGE_MAPPING_JSON

    def generate_key_mapping(self) -> dict:
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.mapper.make_mapping_json(self.out_json)
        with self.out_json.open() as f:
            mapping = json.load(f)
        logging.info(f"Mapping complete: {len(mapping)} entries → {self.out_json}")
        return mapping

if __name__ == "__main__":
    MappingSystem().generate_key_mapping()

