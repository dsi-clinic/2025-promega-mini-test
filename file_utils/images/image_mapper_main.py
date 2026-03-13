#!/usr/bin/env python3
"""Entry point for building the RAW image mapping JSON."""

import sys, logging, os, subprocess
from pathlib import Path

# ---- Force console logging early and unconditionally ----
root = logging.getLogger()
root.setLevel(logging.DEBUG)
for h in list(root.handlers):
    root.removeHandler(h)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root.addHandler(sh)
logging.captureWarnings(True)


def _find_root(start: Path) -> Path | None:
    p = start
    for _ in range(8):
        if (p / ".env").exists() and (
            (p / "paths.py").exists() or (p / "file_utils/common/paths.py").exists()
        ):
            return p
        p = p.parent
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        p = Path(root)
        if (p / "file_utils").exists():
            return p
    except Exception:
        pass
    return None


# --- Locate repo root ---
ROOT = _find_root(Path(__file__).resolve())
if ROOT:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
else:
    logging.warning(
        "Couldn’t auto-locate repo root; relying on CWD/PYTHONPATH and existing env."
    )

from file_utils.images.scripts.mapping_system import MappingSystem

if __name__ == "__main__":
    logging.debug("Logger alive?")
    ms = MappingSystem()
    mapping = ms.generate_key_mapping()
    logging.info("Mapping generation completed successfully.")
