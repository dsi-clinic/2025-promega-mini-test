#!/usr/bin/env python3
"""Entry point for building the RAW image mapping JSON."""

import logging
import sys
from pathlib import Path

# --- Locate repo root (must contain paths.py and .env) ---
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "paths.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing paths.py and .env")


from file_utils.images.scripts.mapping_system import MappingSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def main() -> dict:
    """Initialize the MappingSystem and generate the mapping."""
    ms = MappingSystem()
    mapping = ms.generate_key_mapping()  # writes to P.RAgW_IMAGE_MAPPING_JSON
    logging.info("Mapping generation completed successfully.")
    return mapping

if __name__ == "__main__":
    main()
