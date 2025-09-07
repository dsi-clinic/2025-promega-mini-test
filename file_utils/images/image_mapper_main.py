#!/usr/bin/env python3
"""Entry point for building the RAW image mapping JSON."""

import logging
import sys
from pathlib import Path
from pathlib import Path
import os, sys, subprocess

def _find_root(start: Path) -> Path | None:
    p = start
    for _ in range(8):
        # accept either a top-level paths.py OR file_utils/common/paths.py
        if (p / ".env").exists() and ((p / "paths.py").exists() or (p / "file_utils/common/paths.py").exists()):
            return p
        p = p.parent
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        p = Path(root)
        if (p / "file_utils").exists():
            return p
    except Exception:
        pass
    return None

ROOT = _find_root(Path(__file__).resolve())
if ROOT:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)  # lets find_dotenv() see the right .env
else:
    print("Couldn’t auto-locate repo root; relying on CWD/PYTHONPATH and existing env.")

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
