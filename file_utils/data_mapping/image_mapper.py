"""Image Mapper Module

Handles mapping and processing of microscopy image files across different batches.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from tifffile import imread


class ImageMapper:
    """Handles mapping and processing of image files across batches."""

    def __init__(self: ImageMapper, img_subfolder: str | Path) -> None:
        """Initialize the ImageMapper with the given image subfolder."""
        self.img_subfolder = Path(img_subfolder)
        self.BA_FOLDER_MAP = {0: "BA1/", 1: "BA2/96_1/", 2: "BA2/96_2/", 3: "BA3/"}
        self.mapping_cache: dict[str, list[Path]] = {}

    def list_tif_files(
        self: ImageMapper, ba: str, dayID: str, max_files: int = 5
    ) -> tuple[Path, list[Path]]:
        """Lists all .tif files in the appropriate BA subfolder for a given day.

        BA2 uses two tokens for the folder path; BA1 and BA3 use only the first token.
        """
        tokens = ba.split()
        if len(tokens) > 1:
            if tokens[0].upper() == "BA2":
                img_subfolder = tokens[0].upper() + "/" + tokens[1]
            else:
                img_subfolder = tokens[0].upper()
        else:
            img_subfolder = ba.upper()

        img_folder = (
            Path("/net/projects2/promega/data-analysis") / img_subfolder / dayID
        )
        # Use rglob to include files in subdirectories if needed
        filenames = list(img_folder.rglob("*.tif"))
        filenames.sort()
        return img_folder, filenames

    def clean_metadata(
        self: ImageMapper,
        range_row: tuple[int, int],
        meta_file: DataFrame,
        diffday_j: int,
    ) -> DataFrame:
        """Clean and organize metadata for the specified day.

        Handles repeated day columns by selecting the appropriate suffix.
        """
        cleaned_metadata: DataFrame = pd.DataFrame(
            columns=[
                "photoID",
                "plateType",
                "numofFocus",
                "Objective",
                "imHeight",
                "imWidth",
                "dayID",
                "wellID",
                "batchPlate",
            ]
        )

        for row_i in range(range_row[0], range_row[1]):
            if diffday_j == 0:
                photoID = meta_file.iloc[row_i]["Photo ID"]
                numofFocus = meta_file.iloc[row_i]["Number of Focus"]
                plateType = meta_file.iloc[row_i]["Plate Type"]
                Objective = meta_file.iloc[row_i]["Objective / Microscope"]
            else:
                photoID = meta_file.iloc[row_i]["Photo ID." + str(diffday_j)]
                numofFocus = meta_file.iloc[row_i]["Number of Focus." + str(diffday_j)]
                plateType = meta_file.iloc[row_i]["Plate Type." + str(diffday_j)]
                Objective = meta_file.iloc[row_i][
                    "Objective / Microscope." + str(diffday_j)
                ]

            splitted = photoID.split()
            dayID = splitted[-2]
            wellID = splitted[-1]

            # Extract batchPlate from photoID (all tokens except the last two)
            batchPlate = " ".join(splitted[:-2])
            cleaned_metadata = pd.concat(
                [
                    cleaned_metadata,
                    pd.DataFrame(
                        [
                            {
                                "photoID": photoID,
                                "plateType": plateType,
                                "numofFocus": numofFocus,
                                "Objective": Objective,
                                "dayID": dayID,
                                "wellID": wellID,
                                "batchPlate": batchPlate,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        return cleaned_metadata

    def resolve_filename(
        self: ImageMapper, file_photoID: str, img_folder: str | Path
    ) -> tuple[Path | None, str, list[Path]]:
        """Finds the correct file based on naming rules.

        Uses a regex with a word-boundary to avoid partial matches (e.g., A1 vs A10).
        Returns a tuple: (resolved file, stitched status, list of filtered files).
        """
        img_folder = Path(img_folder)
        logging.info(f"Resolving filename for {file_photoID} in {img_folder}")

        # Build regex pattern; ensures an exact match followed by a word boundary
        pattern = re.compile(re.escape(file_photoID) + r"(\b|$)", re.IGNORECASE)
        filtered_files = [f for f in img_folder.rglob("*.tif") if pattern.match(f.name)]

        # Function to extract Z-index from filename
        def extract_z(file: Path) -> int:
            m = re.search(r" Z(\d+)\.tif$", file.name, re.IGNORECASE)
            if m:
                return int(m.group(1))
            return -1

        # Sort the filtered files based on the extracted Z-index
        filtered_files.sort(key=extract_z)

        stitched_file = None

        # Check if the file is part of a stitched set (identified by a pattern like "(1 of 2)")
        for file in filtered_files:
            if re.search(r"\(\d+ *of *\d+\)", file.name):
                stitched_file = file
                break

        if stitched_file:
            return stitched_file, "Yes", filtered_files

        # Handle BA3-specific naming inconsistencies by replacing '96_1' with 'Pt1'
        if "ba3" in file_photoID.lower():
            pt1_with_z = img_folder / f"{file_photoID.replace('96_1', 'Pt1')} Z0.tif"
            pt1_without_z = img_folder / f"{file_photoID.replace('96_1', 'Pt1')}.tif"

            if pt1_with_z.is_file():
                return pt1_with_z, "No", filtered_files
            elif pt1_without_z.is_file():
                return pt1_without_z, "No", filtered_files
            else:
                file_photoID = file_photoID.replace("Pt1", "96_1")

        # Standard checks: look for files with or without an explicit Z index in the filename
        file_with_z = img_folder / f"{file_photoID} Z0.tif"
        file_without_z = img_folder / f"{file_photoID}.tif"

        if file_with_z.is_file():
            return file_with_z, "No", filtered_files
        elif file_without_z.is_file():
            return file_without_z, "No", filtered_files
        elif filtered_files:
            return filtered_files[0], "No", filtered_files

        logging.warning(f"No files found for {file_photoID} in {img_folder}")
        return None, "No", []

    # Finds best focused image
    def find_best_focus(self: ImageMapper, files: list[Path]) -> int:
        """Finds the best-focused Z-stack slice using Laplacian variance.

        Returns the index of the file with the highest focus quality.
        """
        # If only one Z stack exists, store as -1
        if not files:
            return -1

        best_z = 0
        max_variance = float("-inf")
        channel_axis = 2

        for i, file in enumerate(files):
            try:
                img = imread(str(file))
                # If the image has more than 2 dimensions, average over the channel axis
                if img.ndim > channel_axis:
                    img = img.mean(axis=channel_axis).astype(np.uint8)
                variance = cv2.Laplacian(img, cv2.CV_64F).var()

                if variance > max_variance:
                    max_variance = variance
                    best_z = i
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
                continue

        return best_z
