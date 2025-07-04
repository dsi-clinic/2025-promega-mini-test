from __future__ import annotations
import logging
import pandas as pd
import re
import json
from pathlib import Path
from tifffile import TiffFile  # if you ever need it
import cv2
from skimage.io import imread

logging.basicConfig(level=logging.DEBUG)

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
            "Focus Step (µm)":                        "dz",
            "Cell line":                              "cellLine",
            "Treatments (AAV)":                       "treatment"
        })

        def split_pid(pid: str) -> pd.Series:
            """
            Break “Ba2 96_1 Dy21 D12(1 of 2) #% Z0.tif”-style strings into
            batchPlate = “Ba2 96_1” | “Ba1” | “Ba3 Pt1” …
            dayID      = “Dy21”
            wellID     = “D12”
            """
            parts = pid.split()

            # ── batchPlate ────────────────────────────────────────────
            #  Ex.:  Ba2 96_1 | Ba3 Pt1 | Ba1
            if len(parts) > 1 and re.match(r"^(96_[12]|Pt1)$", parts[1], re.I):
                batchPlate = f"{parts[0]} {parts[1]}"
                day_idx = 2
            else:
                batchPlate = parts[0]
                day_idx = 1

            # ── dayID (always starts with Dy..) ───────────────────────
            dayID = parts[day_idx]
            well_tokens = parts[day_idx + 1 :]

            # ── wellID  (strip EVERYTHING after the first letter+digits) ──
            m = re.search(r"[A-Za-z]\d{1,2}", " ".join(well_tokens))
            wellID = m.group(0) if m else " ".join(well_tokens).strip()

            logging.debug(
                f"[split_pid] {pid!r} → batch={batchPlate!r}, day={dayID!r}, well={wellID!r}"
            )
            return pd.Series([batchPlate, dayID, wellID])


        df[["batchPlate", "dayID", "wellID"]] = df["photoID"].apply(split_pid)

        return df[[
            "photoID", "orgID", "batchPlate", "dayID", "wellID",
            "Microscope", "objective", "Image Width (Pixel)",
            "Image Width (µm)", "um_per_px", "numFocus", "firstZ", "lastZ", "dz",
            "cellLine", "treatment"
        ]]


    def resolve_filename(
        self, file_photoID: str, img_folder: str|Path, batch_plate: str = None
    ) -> tuple[Path|None, str, list[Path]]:
        """1) prefix‐match  2) sort by Z  3) stitched‐tile  4) BA3 Pt1  5) Z0/.tif  6) fallback."""
        img_folder = Path(img_folder)
        logging.info(f"Resolving filename for {file_photoID} in {img_folder}")

        # Extract well ID once from the original input
        well_match = re.search(r'([A-Za-z]\d+)', file_photoID)
        well_id = well_match.group(1) if well_match else ""
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
        elif batch_plate:
            plate_suffix_match = re.search(r"(96_[12]|Pt1)", batch_plate, re.IGNORECASE)
            ba_match = re.search(r"BA\d+", search_id, re.IGNORECASE)

            if ba_match and plate_suffix_match:
                base_id = search_id.strip()
                ba_part = ba_match.group(0)
                plate_suffix = plate_suffix_match.group(1)

                # Add versions: raw, with 96_1 or 96_2, and Pt1 (to handle bad filenames)
                with_suffix = re.sub(rf"{ba_part}\b", f"{ba_part} {plate_suffix}", base_id, flags=re.IGNORECASE)
                alt_suffix = "Pt1" if "96_" in plate_suffix else "96_1"
                alt_version = re.sub(rf"{ba_part}\b", f"{ba_part} {alt_suffix}", base_id, flags=re.IGNORECASE)

                search_ids = [base_id, with_suffix, alt_version]
                logging.info(f"Using multiple search patterns: {search_ids}")
            else:
                search_ids = [search_id]


        # Try all search IDs until we find matches
        candidates = []
        for sid in search_ids:
            # DON'T strip special characters - keep the full identifier
            # Just clean up any trailing whitespace
            clean_sid = sid.strip()
            
            # Create multiple search patterns to handle different file naming conventions
            patterns = []
            
            # 1. Exact match with word boundary (for standard cases)
            patterns.append(rf"\b{re.escape(clean_sid)}(?=[\s._Z(]|$)")
            
            # 2. More flexible pattern that handles special characters
            # This pattern looks for the exact string followed by common delimiters
            patterns.append(rf"{re.escape(clean_sid)}(?=[\s._Z]|$)")
            
            # 3. Handle cases where special characters might be represented differently
            # Create a version that treats parentheses content as optional
            if '(' in clean_sid and ')' in clean_sid:
                # Extract the part before parentheses and the part in parentheses
                base_part = clean_sid.split('(')[0].strip()
                paren_content = re.search(r'\(([^)]*)\)', clean_sid)
                if paren_content:
                    paren_part = paren_content.group(1)
                    # Try pattern that matches base part followed by parentheses with any content
                    patterns.append(rf"\b{re.escape(base_part)}\s*\([^)]*{re.escape(paren_part)}[^)]*\)")
                    # Also try pattern that matches base part with flexible parentheses content
                    patterns.append(rf"\b{re.escape(base_part)}\s*\([^)]*\)")
            
            # 4. Fallback pattern - just match the well ID part flexibly
            well_match = re.search(r'([A-Za-z]\d+)', clean_sid)
            if well_match:
                well_id = well_match.group(1)
                # Look for well ID followed by any special characters
                patterns.append(rf"\b{re.escape(well_id)}\s*[(%#]*[^A-Za-z]*")
            
            logging.debug(f"Trying patterns for {clean_sid}: {patterns}")
            
            for pattern in patterns:
                try:
                    search_re = re.compile(pattern, re.IGNORECASE)
                    these_candidates = [f for f in img_folder.rglob("*.tif") if search_re.search(f.name)]
                    
                    if these_candidates:
                        candidates = these_candidates
                        logging.info(f"Found {len(candidates)} matches with pattern: {pattern}")
                        break
                except re.error as e:
                    logging.warning(f"Invalid regex pattern {pattern}: {e}")
                    continue
            
            if candidates:
                break

        # If still no candidates, try more permissive search
        if not candidates:
            # Extract well ID from the search pattern
            well_match = re.search(r'([A-Za-z]\d+)', file_photoID)
            if well_match:
                well_id = well_match.group(1)
                logging.info(f"Trying fallback search with well ID: {well_id}")
                
                # Look for files with the well ID in the filename - very permissive
                candidates = []
                for f in img_folder.rglob("*.tif"):
                    # Check if the well ID appears in the filename
                    if re.search(rf"\b{re.escape(well_id)}\b", f.name, re.IGNORECASE):
                        candidates.append(f)
                
                # If still no matches, try even more permissive search
                if not candidates:
                    for f in img_folder.rglob("*.tif"):
                        if well_id.lower() in f.name.lower():
                            candidates.append(f)

        if not candidates:
            logging.warning(f"No files found for {file_photoID} in {img_folder}")
            return None, "No", []

        # 2) sort by Z-index
        def extract_z(f: Path) -> int:
            m = re.search(r" Z(\d+)", f.name, re.IGNORECASE)
            return int(m.group(1)) if m else -1
        candidates.sort(key=extract_z)

        logging.debug(f"[STITCH CHECK] well_id = {well_id}")
        logging.debug(f"[STITCH CHECK] candidate files = {[f.name for f in candidates]}")


        # 3) stitched‐tile check - detect numeric tile indices like A1(1).tif
        # 3) stitched‐tile check - detect numeric tile indices like A1(1).tif
        stitched_files = []
        for f in candidates:
            fname = f.name.lower()
            if "(" in fname and ")" in fname and re.search(r"\(\d+\)", fname):
                logging.info(f"[STITCH CHECK] Found candidate stitched tile: {f.name}")
                stitched_files.append(f)

        if stitched_files:
            stitched_files.sort(key=extract_z)
            logging.info(f"[STITCH CHECK] Final stitched file chosen: {stitched_files[0].name}")
            return stitched_files[0], "Yes", candidates



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
        candidates = [f for f in candidates if well_id.lower() in f.name.lower()]

        if candidates:
            return candidates[0], "No", candidates

        logging.warning(f"No files found for {search_id} in {img_folder}")
        return None, "No", []

    def make_mapping_json(self, out_json: Path):
        """
        Loop through metadata → build one ‘full_id’ per (dayID, batchPlate, wellID)
        → ask resolve_filename() once → store results in a JSON file.
        """
        logging.info("Generating key-mapping JSON…")

        cleaned  = self.clean_metadata()
        grouped  = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        mapping  = {}

        for (day_id, batch_plate, well_id), group_df in grouped:
            logging.info(f"Processing {batch_plate} {day_id} {well_id}")

            # ───────────────────────────────────────────────────────── full_id
            parts   = batch_plate.split()                    # e.g. ['Ba2', '96_1']
            ba_str  = " ".join([parts[0].upper(), *parts[1:]])  # 'BA2 96_1' | 'BA4'
            full_id = f"{ba_str} {day_id} {well_id}"           # 'BA2 96_1 Dy21 D12'
            logging.debug(f"Constructed full ID: {full_id}")

            # ───────────────────────────────────────────────────────── pick folder
            ba_part = parts[0].upper()                        # BA1 / BA2 / BA3 …
            sub     = self.BA_FOLDER_MAP[ba_part]             # str or [str, str]

            if isinstance(sub, list):
                # BA2 ⇢ pick the 96_1 / 96_2 sub-folder that matches the batchPlate
                plate_suffix = parts[1] if len(parts) > 1 else ""
                sub = next((s for s in sub if plate_suffix in s), sub[0])
                logging.debug(f"Selected subfolder {sub} for {plate_suffix}")

            img_folder = self.base_dir / sub / day_id
            if not img_folder.exists():
                logging.warning(f"Image folder does not exist: {img_folder}")
                continue

            # ───────────────────────────────────────────────────────── find file
            chosen, stitched_flag, all_files = self.resolve_filename(
                full_id, img_folder, batch_plate
            )
            if chosen is None:        # resolve_filename already logged the failure
                continue

            # stitched stack → focus_idx = -1 ; otherwise choose best-focus Z
            if stitched_flag == "Yes":
                focus_idx = -1
                final     = chosen
            else:
                idx       = self.find_best_focus(all_files)
                focus_idx = idx if 0 <= idx < len(all_files) else -1
                final     = all_files[focus_idx] if focus_idx >= 0 else chosen

            # ───────────────────────────────────────────────────────── record row
            mapping[full_id] = {
                "dayID":      day_id,
                "BA":         ba_str,
                "wellID":     well_id,
                "Best Z":     focus_idx,
                "Best Z Filename": str(final),
                "Stitched":   stitched_flag,
                "um_per_px":  float(group_df["um_per_px"].iloc[0]),
                "all_files":  [str(f) for f in all_files],
                "cellLine":   group_df["cellLine"].iloc[0],
                "treatment":  group_df["treatment"].iloc[0],
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