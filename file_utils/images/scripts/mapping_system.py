#!/usr/bin/env python3
import json
import logging
from pathlib import Path
import sys

# --- Locate repo root (must contain paths.py and .env) ---
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "paths.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing paths.py and .env")

# --- Imports that rely on repo root ---
import paths as P
from .image_mapper import ImageMapper

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class MappingSystem:
    def __init__(self):
        # paths.py already loaded .env; just log what we’re using
        logging.info(f"BASE_PATH               = {P.BASE_PATH}")
        logging.info(f"META_FILE               = {P.META_FILE} (exists={P.META_FILE.exists()})")
        logging.info(f"RAW_IMAGE_MAPPING_JSON  = {P.RAW_IMAGE_MAPPING_JSON}")

        # pick the raw image root; use RAW_IMAGES_DIR if you’ve added it, else BASE_PATH
        raw_images_root = getattr(P, "RAW_IMAGES_DIR", P.BASE_PATH)
        logging.info(f"RAW_IMAGES_ROOT         = {raw_images_root}")

        self.mapper = ImageMapper(
            base_dir=raw_images_root,
            meta_csv=P.META_FILE,
        )
        self.out_json = P.RAW_IMAGE_MAPPING_JSON

    def generate_key_mapping(self) -> dict:
        """Build image_mapping.json and return it as a dict."""
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.mapper.make_mapping_json(self.out_json)

        with self.out_json.open() as f:
            mapping = json.load(f)
        logging.info(f"Mapping complete: {len(mapping)} entries → {self.out_json}")
        return mapping

if __name__ == "__main__":
    ms = MappingSystem()
    ms.generate_key_mapping()
