import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment early, before accessing os.getenv()
load_dotenv(find_dotenv(), override=True)

BASE_PATH = Path(os.getenv("BASE_PATH"))
META_FILE = Path(os.getenv("META_FILE"))
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER"))
