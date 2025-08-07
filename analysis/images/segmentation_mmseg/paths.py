from pathlib import Path
import glob
import os
from dotenv import load_dotenv

# Explicit path to .env (2 levels up from paths.py)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1].parents[1] / ".env")

BASE_PATH = Path(os.environ["BASE_PATH"])
ORIGINAL_MAPPING = Path(os.environ["ORIGINAL_MAPPING"])
MANUAL_MAPPING_OUTPUT_DIR = BASE_PATH / "manual_masks"
MANUAL_MASK_FOLDERS = [
    Path(p) for p in glob.glob(str(BASE_PATH / "manual_masks" / "masks-batch-*" / "*"))
    if Path(p).is_dir()
]
