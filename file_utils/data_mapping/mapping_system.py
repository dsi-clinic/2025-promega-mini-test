"""Complete Image Mapping System

Combines ImageMapper with JSON mapping functionality
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .image_mapper import ImageMapper  # Ensure ImageMapper is correctly implemented

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MappingSystem:
    """Handles generating the full key mapping.

    Loads environment variables, sets up paths, and initializes the ImageMapper.
    """

    def __init__(self: "MappingSystem") -> None:
        """Initialize the MappingSystem.

        Loads environment variables, sets up paths, and initializes the ImageMapper.
        """
        # Load environment variables
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path)

        # Initialize paths
        self.base_path = Path(os.getenv("BASE_PATH", "./"))
        self.meta_file = Path(os.getenv("META_FILE", "./metadata.xlsx"))
        self.output_folder = Path(os.getenv("OUTPUT_FOLDER", "./output"))
        self.img_subfolder = Path(os.getenv("IMG_SUBFOLDER", "./images"))

        # Initialize ImageMapper
        self.mapper = ImageMapper(self.img_subfolder)

    def load_metadata(self: "MappingSystem") -> pd.DataFrame:
        """Load and clean metadata from an Excel file.

        Loops through batches and day columns to consolidate metadata.
        """
        try:
            logging.info(f"Loading metadata from {self.meta_file}")
            meta_df = pd.read_excel(self.meta_file, sheet_name="Photographs")

            # Number of day columns in the metadata (e.g., "Photo ID", "Photo ID.1", etc.)
            num_days = 11  # Adjust based on your actual metadata structure
            all_metadata = []

            # Loop through each batch (e.g., BA1, BA2, BA3, BA4) and each day column.
            for ba in range(4):  # For each batch
                for diffday_j in range(num_days):  # For each day column/index
                    logging.info(
                        f"Processing batch: BA{ba + 1}, day index: {diffday_j}"
                    )
                    range_row = (96 * ba, 96 + 96 * ba)
                    batch_metadata = self.mapper.clean_metadata(
                        range_row, meta_df, diffday_j
                    )
                    all_metadata.append(batch_metadata)

            # Concatenate all batch and day data into a single DataFrame
            cleaned_df = pd.concat(all_metadata, ignore_index=True)

            return cleaned_df
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            raise

    def generate_key_mapping(self: "MappingSystem") -> dict:
        """Generate a mapping between metadata and image files.

        Processes the cleaned metadata, resolves filenames, and selects the best focus for each well.
        """
        logging.info("Starting key mapping generation...")

        try:
            metadata = self.load_metadata()
            # Group by day, batchPlate, and wellID
            grouped = metadata.groupby(["dayID", "batchPlate", "wellID"])
            mapping_dict = {}

            for (day_id, batch_plate, well_id), _well_data in grouped:
                logging.info(
                    f"Processing batch_plate: {batch_plate}, day: {day_id}, well: {well_id}"
                )
                # Split batch_plate into tokens (e.g., "Ba1 96_1")
                parts = batch_plate.split()
                batch = parts[0]  # e.g., "Ba1", "Ba2", or "Ba3"

                # Construct full_id; include the second token if it exists
                if len(parts) > 1:
                    full_id = f"{batch} {parts[1]} {day_id} {well_id}"
                else:
                    full_id = f"{batch} {day_id} {well_id}"

                # Normalize BA2 to include the second token if needed; for BA1/BA3, use title case
                if batch.upper() == "BA2" and len(parts) > 1:
                    batch = f"BA2 {parts[1]}"
                else:
                    batch = batch.title()

                # Get the folder and list of TIFF files for the given day
                img_folder, file_list = self.mapper.list_tif_files(batch, day_id)
                if not file_list:
                    logging.warning(
                        f"No image folder found for {batch_plate} {day_id} {well_id}, skipping..."
                    )
                    continue

                # Resolve filename using full_id and the image folder
                matched_file, is_stitched, filtered_files = (
                    self.mapper.resolve_filename(full_id, img_folder)
                )

                if not matched_file:
                    logging.warning(f"No files found for {full_id}, skipping...")
                    continue

                resolved_filename = str(matched_file)
                NO_Z_STACK = -1

                # Determine the best focus index using the filtered list of files
                best_z = self.mapper.find_best_focus(filtered_files)
                if best_z != NO_Z_STACK and best_z < len(filtered_files):
                    final_filename = str(filtered_files[best_z])
                else:
                    final_filename = resolved_filename

                # Build mapping entry for this well
                mapping_dict[full_id] = {
                    "dayID": day_id,
                    "BA": batch,
                    "wellID": well_id,
                    "Best Z": best_z,
                    "Best Z Filename": final_filename,
                    "Stitched": is_stitched,
                }

            self._save_mapping(mapping_dict)
            return mapping_dict

        except Exception as e:
            logging.error(f"Error generating mapping: {e}")
            raise

    def _save_mapping(self: "MappingSystem", mapping_dict: dict) -> None:
        """Save the mapping dictionary to a JSON file."""
        try:
            if not mapping_dict:
                logging.error("No data to save in JSON! Mapping dictionary is empty!")
                return

            # Ensure output directory exists
            self.output_folder.mkdir(parents=True, exist_ok=True)

            # Save JSON file
            output_path = self.output_folder / "image_mapping.json"
            with output_path.open("w") as f:
                json.dump(mapping_dict, f, indent=4)

            logging.info(f"Mapping saved to {output_path}")

        except Exception as e:
            logging.error(f"Error saving mapping: {e}")
