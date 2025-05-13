from __future__ import annotations
import logging
import pandas as pd
import re
import json
from pathlib import Path
from tifffile import TiffFile  # if you ever need it
import cv2
from skimage.io import imread

class ImageMapper:
    BA_FOLDER_MAP = {
        "BA1": "BA1",
        "BA2": ["BA2/96_1", "BA2/96_2"],
        "BA3": "BA3",
        "BA4": "BA4"
    }

    def __init__(self, base_dir: Path, meta_csv: Path):
        self.base_dir = Path(base_dir)
        self.meta = pd.read_excel(meta_csv, sheet_name="Images")
        self._precompute_um_per_px()

    def _precompute_um_per_px(self):
        cols = list(self.meta.columns)
        px_candidates = [c for c in cols if "Image Width" in c and "Pixel" in c]
        um_candidates = [c for c in cols if "Image Width" in c and "µm" in c]

        if not px_candidates or not um_candidates:
            raise ValueError(f"Could not find width columns in {cols}")

        px_col = px_candidates[0]
        um_col = um_candidates[0]
        logging.info(f"Using pixel-col={px_col!r}, micron-col={um_col!r}")

        # Coerce to numeric
        self.meta[px_col] = (
            self.meta[px_col].astype(str)
                .str.replace(",", "")
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
        )
        self.meta[um_col] = (
            self.meta[um_col].astype(str)
                .str.replace(",", "")
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
        )

        self.meta["um_per_px"] = self.meta[um_col] / self.meta[px_col]

    def clean_metadata(self) -> pd.DataFrame:
        df = self.meta.rename(columns={
            "Photo ID (Batch Plate Day Well)":        "photoID",
            "Organoid ID (Same as in Organoid Info)": "orgID",
            "Picture Day":                            "dayID",
            "Objective":                              "objective",
            "Number of Focus":                        "numFocus",
            "First Focus":                            "firstZ",
            "Last Focus":                             "lastZ",
            "Focus Step (µm)":                        "dz"
        })

        def split_pid(pid: str) -> pd.Series:
            parts = pid.split()
            
            # 1. Extract the batch and plate parts
            batch_parts = []
            i = 0
            while i < len(parts):
                part = parts[i]
                # Match Ba1, Ba2, Ba3, Ba4, etc.
                if re.match(r"^Ba\d+$", part, re.IGNORECASE):
                    batch_parts.append(part)
                    # Check if the next part is 96_1 or 96_2 (for Ba2 or Ba3)
                    if i+1 < len(parts) and re.match(r"^(?:96_[12]|Pt1)$", parts[i+1]):
                        batch_parts.append(parts[i+1])
                        i += 1  # Skip the next part as we've included it
                    break  # We've found the batch part
                i += 1
            
            batchPlate = " ".join(batch_parts)
            
            # 2. Find day part (always starts with Dy)
            day_index = next((i for i, part in enumerate(parts) if part.upper().startswith("DY")), -1)
            dayID = parts[day_index] if day_index >= 0 else ""
            
            # 3. Extract well ID (everything after day)
            well_parts = parts[day_index+1:] if day_index >= 0 else []
            raw_well = " ".join(well_parts)
            
            # 4. Clean the well ID - keep the core ID part but strip trailing special chars
            # First, identify the well pattern: usually letter followed by number (B6, H12, etc.)
            well_match = re.match(r"^([A-Za-z]\d+).*$", raw_well)
            if well_match:
                # We found a standard well pattern - use it as is
                wellID = well_match.group(1)
            else:
                # If no standard pattern found, just remove trailing special chars
                wellID = re.sub(r"[#%()]+$", "", raw_well).strip()
            
            # Special case for capturing "(1" as part of the ID if it exists
            if "(" in raw_well and not raw_well.endswith(")"):
                paren_match = re.search(r"([A-Za-z]\d+\([^)]*)", raw_well)
                if paren_match:
                    wellID = paren_match.group(1)
            
            logging.debug(f"Split {pid!r} into: batch={batchPlate!r}, day={dayID!r}, well={wellID!r}")
            return pd.Series([batchPlate, dayID, wellID])

        df[["batchPlate", "dayID", "wellID"]] = df["photoID"].apply(split_pid)

        return df[[
            "photoID", "orgID", "batchPlate", "dayID", "wellID",
            "Microscope", "objective", "Image Width (Pixel)",
            "Image Width (µm)", "um_per_px", "numFocus", "firstZ", "lastZ", "dz"
        ]]

    def resolve_filename(
        self, file_photoID: str, img_folder: str|Path, batch_plate: str = None
    ) -> tuple[Path|None, str, list[Path]]:
        """1) prefix‐match  2) sort by Z  3) stitched‐tile  4) BA3 Pt1  5) Z0/.tif  6) fallback."""
        img_folder = Path(img_folder)
        logging.info(f"Resolving filename for {file_photoID} in {img_folder}")

        # If we have information about 96_1 or 96_2 in batch_plate, include it in the search pattern
        search_id = file_photoID
        
        # Handle special case for BA3 Pt1 conversion
        if "ba3" in search_id.lower() and "96_1" in batch_plate.lower() and "96_1" not in search_id:
            # Try both versions - with Pt1 and with 96_1
            search_ids = [
                search_id,
                re.sub(r"BA3\b", "BA3 96_1", search_id, flags=re.IGNORECASE), 
                re.sub(r"BA3\b", "BA3 Pt1", search_id, flags=re.IGNORECASE)
            ]
            logging.info(f"Using multiple search patterns for BA3: {search_ids}")
        elif batch_plate and "96_" in batch_plate and "96_" not in search_id:
            # For other batches, add the plate info (96_1 or 96_2) if missing
            plate_suffix = re.search(r"96_[12]", batch_plate)
            if plate_suffix:
                ba_match = re.search(r"BA\d+", search_id.upper())
                if ba_match:
                    ba_part = ba_match.group(0)
                    adjusted_id = re.sub(
                        rf"{ba_part}\b", 
                        f"{ba_part} {plate_suffix.group(0)}", 
                        search_id, 
                        flags=re.IGNORECASE
                    )
                    search_ids = [search_id, adjusted_id]
                    logging.info(f"Using multiple search patterns: {search_ids}")
                else:
                    search_ids = [search_id]
            else:
                search_ids = [search_id]
        else:
            search_ids = [search_id]

        # Try all search IDs until we find matches
        candidates = []
        for sid in search_ids:
            # Strip any trailing special characters for search pattern
            clean_sid = re.sub(r"[%#()]+$", "", sid).strip()
            
            # Try two matching approaches: word boundary and more flexible
            patterns = [
                rf"\b{re.escape(clean_sid)}(?=[\s(]|$)",  # Word boundary
                rf"{re.escape(clean_sid)}(?:\s|\.|$)"      # More flexible
            ]
            
            for pattern in patterns:
                search_re = re.compile(pattern, re.IGNORECASE)
                these_candidates = [f for f in img_folder.rglob("*.tif") if search_re.search(f.name)]
                
                if these_candidates:
                    candidates = these_candidates
                    logging.info(f"Found {len(candidates)} matches with pattern: {pattern}")
                    break
            
            if candidates:
                break

        # If still no candidates, try more permissive search
        if not candidates:
            # Extract well ID from the search pattern
            well_match = re.search(r"[A-Za-z]\d+(?:\([^)]*)?", search_id)
            if well_match:
                well_id = well_match.group(0)
                logging.info(f"Trying fallback search with well ID: {well_id}")
                
                # Look for files with the well ID in the filename
                candidates = [f for f in img_folder.rglob("*.tif") 
                             if re.search(rf"\b{re.escape(well_id)}\b", f.name, re.IGNORECASE)]

        # 2) sort by Z-index
        def extract_z(f: Path) -> int:
            m = re.search(r" Z(\d+)\.tif$", f.name, re.IGNORECASE)
            return int(m.group(1)) if m else -1
        candidates.sort(key=extract_z)

        # 3) stitched‐tile check
        for f in candidates:
            if re.search(r"\(\d+\s*of\s*\d+\)", f.name):
                return f, "Yes", candidates

        # 4) BA3 Pt1 / 96_1 special case handling
        if "ba3" in search_id.lower():
            # Try both Pt1 and 96_1 versions
            for version, replacement in [("96_1", "Pt1"), ("Pt1", "96_1")]:
                if version in search_id.lower():
                    for suffix in (" Z0.tif", ".tif"):
                        alt_file = img_folder / (search_id.replace(version, replacement) + suffix)
                        if alt_file.is_file():
                            return alt_file, "No", candidates

        # 5) explicit Z0 or bare .tif
        for sid in search_ids:
            z0 = img_folder / f"{sid} Z0.tif"
            bare = img_folder / f"{sid}.tif"
            if z0.is_file():    return z0,   "No", candidates
            if bare.is_file():  return bare, "No", candidates

        # 6) fallback to first candidate
        if candidates:
            return candidates[0], "No", candidates

        logging.warning(f"No files found for {search_id} in {img_folder}")
        return None, "No", []

    def make_mapping_json(self, out_json: Path):
        """Loop through metadata, call resolve_filename once, then pick focus or keep stitched."""
        logging.info("Generating key mapping JSON…")

        cleaned = self.clean_metadata()
        grouped = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        mapping = {}

        for (day_id, batch_plate, well_id), group_df in grouped:
            logging.info(f"Processing {batch_plate} {day_id} {well_id}")

            # Standardize the BA part to uppercase (BA2 instead of Ba2)
            parts = batch_plate.split()
            ba_part = parts[0].upper()
            
            # Handle BA specific logic for full ID construction
            if ba_part == "BA2" and len(parts) > 1 and "96_" in parts[1]:
                ba_str = f"{ba_part} {parts[1]}"
                # Include the plate info in the ID
                full_id = f"{ba_str} {day_id} {well_id}"
            elif ba_part == "BA3" and len(parts) > 1:
                # For BA3, handle both cases: 96_1 and Pt1
                if "96_1" in parts[1]:
                    ba_str = f"{ba_part} {parts[1]}"
                elif "Pt1" in parts[1]:
                    ba_str = f"{ba_part} {parts[1]}"
                else:
                    ba_str = ba_part
                full_id = f"{ba_str} {day_id} {well_id}"
            else:
                ba_str = ba_part
                full_id = f"{ba_str} {day_id} {well_id}"
            
            logging.debug(f"Constructed full ID: {full_id}")

            # resolve folder based on batch type
            sub = self.BA_FOLDER_MAP[ba_part]
            if isinstance(sub, list):
                # Handle BA2 special case with 96_1 and 96_2 subfolders
                if len(parts) > 1 and any(plate in parts[1] for plate in ["96_1", "96_2"]):
                    plate_suffix = parts[1]
                    sub = next((s for s in sub if plate_suffix in s), sub[0])
                    logging.debug(f"Selected subfolder {sub} for {plate_suffix}")
                else:
                    # Default to first subfolder if not specified
                    sub = sub[0]
                    logging.debug(f"Using default subfolder {sub}")
            
            img_folder = self.base_dir / sub / day_id
            if not img_folder.exists():
                logging.warning(f"Image folder does not exist: {img_folder}")
                continue

            # Pass batch_plate to help with special case resolution
            chosen, stitched_flag, all_files = self.resolve_filename(full_id, img_folder, batch_plate)
            if chosen is None:
                continue  # already logged

            # compute Best Z (or -1 if stitched)
            if stitched_flag == "Yes":
                focus_idx = -1
                final = chosen
            else:
                idx = self.find_best_focus(all_files)
                focus_idx = idx if 0 <= idx < len(all_files) else -1
                final = all_files[focus_idx] if focus_idx >= 0 else chosen

            # write out
            mapping[full_id] = {
                "dayID": day_id,
                "BA": ba_str,
                "wellID": well_id,
                "Best Z": focus_idx,
                "Best Z Filename": str(final),
                "Stitched": stitched_flag,
                "um_per_px": float(group_df["um_per_px"].iloc[0]),
                "all_files": [str(f) for f in all_files],
            }

        out_json.write_text(json.dumps(mapping, indent=2))
        logging.info(f"Wrote mapping JSON to {out_json}")

    def find_best_focus(self, files: list[Path]) -> int:
        if not files:
            return -1
            
        best_i = 0
        best_var = -1
        for i, f in enumerate(files):
            img = imread(str(f))
            if img.ndim == 3:
                img = img.mean(axis=2).astype("uint8")
            var = cv2.Laplacian(img, cv2.CV_64F).var()
            if var > best_var:
                best_var = var
                best_i = i
        return best_i