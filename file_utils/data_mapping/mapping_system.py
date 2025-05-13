import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from .image_mapper import ImageMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class MappingSystem:
    def __init__(self):
        # auto-find the nearest .env (and overwrite any existing env vars)
        env_file = find_dotenv()
        logging.info(f"Auto-discovered .env at {env_file}")
        load_dotenv(env_file, override=True)

        # now these will actually be set
        for var in ("BASE_PATH","META_FILE","OUTPUT_FOLDER"):
            logging.info(f"{var} = {os.getenv(var)}")

        self.base_path     = Path(os.getenv("BASE_PATH"))
        self.meta_file     = Path(os.getenv("META_FILE"))
        self.output_folder = Path(os.getenv("OUTPUT_FOLDER"))
        logging.info(f"-> Using meta_file path: {self.meta_file} (exists? {self.meta_file.exists()})")
                # **Instantiate the mapper here!**
        self.mapper = ImageMapper(
            base_dir = self.base_path,
            meta_csv = self.meta_file
        )
        
    def generate_key_mapping(self) -> dict:
        """Have the mapper build and write image_mapping.json, then return it."""
        logging.info("Generating key mapping JSON…")
        # Ensure output dir exists
        self.output_folder.mkdir(parents=True, exist_ok=True)

        out_json = self.output_folder / "image_mapping.json"
        # This single call reads metadata, resolves files, picks best focus,
        # and writes the JSON with um_per_px included.
        self.mapper.make_mapping_json(out_json)

        # Load and return for downstream use
        with out_json.open() as f:
            mapping = json.load(f)
        logging.info(f"Mapping complete: {len(mapping)} entries")
        return mapping
