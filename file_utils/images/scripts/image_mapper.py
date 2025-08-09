from __future__ import annotations
import logging
import pandas as pd
import re
import json
from pathlib import Path
from tifffile import TiffFile  # if you ever need it
import cv2
import numpy as np
from skimage.io import imread

logging.basicConfig(level=logging.DEBUG)

# ---- Fast I/O and caching ----
_IMG_CACHE: dict[tuple[str, tuple[int,int] | None], np.ndarray] = {}
FAST_EVAL_SIZE = (512, 512)   # downscale for fast stats; tweak if you want faster/slower

def load_gray_resized(path: Path, size: tuple[int,int] | None = FAST_EVAL_SIZE) -> np.ndarray | None:
    """Read once, convert to gray, optionally resize, cache by (path, size)."""
    key = (str(path), size)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    _IMG_CACHE[key] = img
    return img

def extract_z(f: Path) -> int:
    m = re.search(r" Z(\d+)", f.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1
def is_blankish_file(
    path: Path,
    # --- soft gates (recommended defaults) ---
    min_total_frac: float = 0.012,    # only declares blank on area alone if truly tiny
    min_largest_frac: float = 0.02,   # largest component must be < 2% of image
    min_component: int = 50,          # ignore specks at eval scale
    center_var_thresh: float = 25.0,  # Laplacian variance (texture) in center crop
    edge_frac_thresh: float = 0.025,  # Canny edge density

    # --- hysteresis band (stability) ---
    hysteresis_low: float = 0.018,    # definitely blank if <= this
    hysteresis_high: float = 0.028    # definitely not blank if >= this
) -> tuple[bool, float]:
    """
    Return (is_blank, total_area_frac) using:
      - Otsu both polarities (pick the *smaller* foreground)
      - small-component removal
      - largest component fraction
      - center-crop Laplacian variance (texture)
      - edge density
      - hysteresis on total area for stability
    """
    gray = load_gray_resized(path, FAST_EVAL_SIZE)
    if gray is None:
        return False, 0.0

    # Otsu both ways
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    m1 = (th1 > 0).astype(np.uint8)
    m2 = (th2 > 0).astype(np.uint8)

    # Use the *smaller* foreground to avoid the ~0.5 trap
    cand = m1 if m1.sum() <= m2.sum() else m2

    # Remove tiny specks + track largest component
    largest_area = 0
    if cand.sum() > 0 and min_component > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        keep = np.zeros_like(cand)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component:
                keep[labels == i] = 1
                if area > largest_area:
                    largest_area = area
        cand = keep

    H, W = cand.shape
    total_area_frac = float(cand.sum()) / float(H * W)
    largest_area_frac = float(largest_area) / float(H * W)

    # Center texture (quarter-size crop)
    ch0, ch1 = H // 4, 3 * H // 4
    cw0, cw1 = W // 4, 3 * W // 4
    center = gray[ch0:ch1, cw0:cw1]
    center_var = float(cv2.Laplacian(center, cv2.CV_64F).var())

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = float((edges > 0).sum()) / float(H * W)

    # --- Hysteresis first (stability) ---
    if total_area_frac <= hysteresis_low:
        is_blank = True
    elif total_area_frac >= hysteresis_high:
        is_blank = False
    else:
        # Composite decision in the gray band
        is_blank = (
            (largest_area_frac < min_largest_frac) and
            (center_var < center_var_thresh) and
            (edge_frac < edge_frac_thresh)
        ) or (total_area_frac < min_total_frac)

    # Clamp into [0, 1] for sanity
    total_area_frac = max(0.0, min(1.0, total_area_frac))
    return bool(is_blank), total_area_frac



def classify_image_file(fname: str) -> str:
    fname_lower = fname.lower()

    # 1. True stitched files
    if "(stitched)" in fname_lower:
        return "Stitched"

    # 2. Partial (multi-tile images without the stitched file)
    if re.search(r"\(\d+\s+of\s+\d+\)", fname_lower):
        return "Partial"

    # 3. Duplicate patterns — (1), (2), etc., but not (#)
    if re.search(r"\((\d+)\)", fname_lower):
        return "Duplicate"

    # 4. Patterns like (#)% — not duplicate, not stitched
    if re.search(r"\(#\)%", fname_lower):
        return "Regular"

    return "Regular"



def clean_id_for_json(s: str) -> str:
    s = re.sub(r"\[.*?\]", "", s)          # remove things in square brackets
    s = re.sub(r"\(.*?\)", "", s)          # remove things in parentheses
    s = re.sub(r"[^A-Za-z0-9\s_]", " ", s) # replace non-alphanumeric
    s = re.sub(r"\s+", " ", s).strip()     # normalize whitespace
    return s

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
            m = re.search(r'(?<!BA)\b([A-H]\d{1,2})\b', " ".join(well_tokens), re.IGNORECASE)
            wellID = m.group(1).upper() if m else " ".join(well_tokens).strip()


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
    ) -> tuple[Path|None, str, list[Path], dict|None]:
        """
        1) prefix‐match  2) sort by Z  3) stitched‐tile  4) BA3 Pt1  5) Z0/.tif  6) fallback.
        Now returns tuple of (chosen_file, stitched_flag, all_files, stitched_groups)
        where stitched_groups is dict of {identifier: [files]} for multiple stitched entries
        """
        img_folder = Path(img_folder)
        files = list(img_folder.rglob("*.tif"))  # <— once


        logging.info(f"Resolving filename for {file_photoID} in {img_folder}")

        # Extract well ID once from the original input
        well_match = re.search(r'(?<!BA)\b([A-H]\d{1,2})\b', file_photoID, re.IGNORECASE)
        well_id = well_match.group(1).upper() if well_match else ""
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
            ba_match = re.search(r"\bBA\d+\b", search_id, re.IGNORECASE)

            if ba_match and plate_suffix_match:
                base_id = search_id.strip()
                ba_part = ba_match.group(0)               # e.g. BA3
                plate_suffix = plate_suffix_match.group(1) # e.g. 96_1 or Pt1

                search_ids = [base_id]

                # If base_id lacks the suffix, add it
                if not re.search(r"\b(96_[12]|Pt1)\b", base_id, re.IGNORECASE):
                    search_ids.append(
                        re.sub(rf"{ba_part}\b", f"{ba_part} {plate_suffix}", base_id, flags=re.IGNORECASE)
                    )

                # Add the alternative form (Pt1 <-> 96_1) only if missing
                alt_suffix = "Pt1" if plate_suffix.lower().startswith("96_") else "96_1"
                if not re.search(rf"\b{re.escape(alt_suffix)}\b", base_id, re.IGNORECASE):
                    search_ids.append(
                        re.sub(rf"{ba_part}\b", f"{ba_part} {alt_suffix}", base_id, flags=re.IGNORECASE)
                    )

                logging.info(f"Using multiple search patterns: {search_ids}")
            else:
                search_ids = [search_id]


        # Try all search IDs until we find matches
        candidates = []
        for sid in search_ids:
            # DON'T strip special characters - keep the full identifier
            clean_sid = sid.strip()

            sid_well = None
            m = re.search(r'(?<!BA)\b([A-H]\d{1,2})\b', clean_sid, re.IGNORECASE)
            if m:
                sid_well = m.group(1).upper()

            
            # Create multiple search patterns to handle different file naming conventions
            patterns = []
            
            # 1. Exact match with word boundary (for standard cases)
            patterns.append(rf"\b{re.escape(clean_sid)}(?=[\s._Z(]|$)")
            
            # 2. More flexible pattern that handles special characters
            patterns.append(rf"{re.escape(clean_sid)}(?=[\s._Z]|$)")

            
            # 3. Handle cases where special characters might be represented differently
            if '(' in clean_sid and ')' in clean_sid:
                base_part = clean_sid.split('(')[0].strip()
                paren_content = re.search(r'\(([^)]*)\)', clean_sid)
                if paren_content:
                    paren_part = paren_content.group(1)
                    patterns.append(rf"\b{re.escape(base_part)}\s*\([^)]*{re.escape(paren_part)}[^)]*\)")
                    patterns.append(rf"\b{re.escape(base_part)}\s*\([^)]*\)")
            
            # 4. Fallback pattern - match well ID with flexible stitched patterns
            if sid_well:
                patterns.append(rf"\b{re.escape(sid_well)}\s*\([^)]*\)")
            
            logging.debug(f"Trying patterns for {clean_sid}: {patterns}")
            
            for pattern in patterns:
                try:
                    search_re = re.compile(pattern, re.IGNORECASE)
                    these_candidates = [f for f in files if search_re.search(f.name)]
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
        if not candidates and well_id:
            logging.info(f"Trying fallback search with well ID: {well_id}")
            candidates = [f for f in files if re.search(rf"\b{re.escape(well_id)}\b", f.name, re.IGNORECASE)]
            if not candidates:
                candidates = [f for f in files if well_id.lower() in f.name.lower()]

        if not candidates:
            logging.warning(f"No files found for {file_photoID} in {img_folder}")
            return None, "No", [], None

        # Log all candidates before processing
        logging.info(f"All candidates found for {file_photoID}:")
        for i, f in enumerate(candidates):
            logging.info(f"  {i}: {f.name}")

        # 2) sort by Z-index

        candidates.sort(key=extract_z)

        def extract_stitched_identifier(filename: str) -> str:
            """
            Extract the stitched identifier from filename to group similar stitched files.
            Returns the full stitched pattern like '(1 of 2)', '(#)%', '(stitched)', etc.
            """
            fname = filename
            
            # Look for various stitched patterns and return the full match
            patterns = [
                r'\(stitched\)',
                r'\(\d+\s+of\s+\d+\)',  # (1 of 2), (2 of 2), etc.
                r'\(\d+\)%',            # (1)%, (2)%, etc.
                r'\(#\)%',              # (#)%
                r'\([^)]*\)\([^)]*\)',  # Multiple parentheses like (1 of 2)(#)%
                r'\([^)]*\).*%',        # Any parentheses with % symbol
            ]
            
            for pattern in patterns:
                match = re.search(pattern, fname, re.IGNORECASE)
                if match:
                    return match.group(0)
            
            # Fallback - return any parentheses content
            match = re.search(r'\([^)]*\)', fname)
            return match.group(0) if match else ""

        # Classify all candidates
        stitched_files = []
        partial_files = []
        duplicate_files = []
        regular_files = []

        for f in candidates:
            label = classify_image_file(f.name)
            if label == "Stitched":
                stitched_files.append(f)
            elif label == "Partial":
                partial_files.append(f)
            elif label == "Duplicate":
                duplicate_files.append(f)
            else:
                regular_files.append(f)

        # Priority 1: real stitched file (has '(stitched)' in name)
        stitched_only = [f for f in stitched_files if "(stitched)" in f.name.lower()]
        if stitched_only:
            stitched_only.sort(key=extract_z)
            logging.info(f"[STITCHED] Found stitched file: {stitched_only[0].name}")
            return stitched_only[0], "Stitched", stitched_only, None

        # Priority 2: partial tiles like '(1 of 2)', if no stitched file
        if partial_files:
            partial_files.sort(key=extract_z)
            logging.info(f"[PARTIAL] Found {len(partial_files)} partial tiles.")
            idx = self.find_best_focus(partial_files)
            chosen = partial_files[idx] if 0 <= idx < len(partial_files) else partial_files[0]
            return chosen, "Partial", partial_files, None

        # Priority 3: duplicates
        if duplicate_files:
            duplicate_files.sort(key=extract_z)
            idx = self.find_best_focus(duplicate_files)
            chosen = duplicate_files[idx] if 0 <= idx < len(duplicate_files) else duplicate_files[0]
            return chosen, "Duplicate", duplicate_files, None



        # 4) BA3 Pt1 / 96_1 special case handling
        if "ba3" in search_id.lower():
            # Try both Pt1 and 96_1 versions
            for version, replacement in [("96_1", "Pt1"), ("Pt1", "96_1")]:
                if version in search_id.lower():
                    for suffix in (" Z0.tif", ".tif"):
                        alt_file = img_folder / (search_id.replace(version, replacement) + suffix)
                        if alt_file.is_file():
                            return alt_file, "No", candidates, None

        # 5) explicit Z0 or bare .tif
        for sid in search_ids:
            z0 = img_folder / f"{sid} Z0.tif"
            bare = img_folder / f"{sid}.tif"
            if z0.is_file():    
                logging.info(f"Found Z0 file: {z0.name}")
                return z0, "No", candidates, None
            if bare.is_file():  
                logging.info(f"Found bare .tif file: {bare.name}")
                return bare, "No", candidates, None

        # 6) fallback to first candidate
        candidates = [f for f in candidates if well_id.lower() in f.name.lower()]

        if candidates:
            logging.info(f"Using fallback candidate: {candidates[0].name}")
            return candidates[0], "No", candidates, None

        logging.warning(f"No files found for {search_id} in {img_folder}")
        return None, "No", [], None


    def make_mapping_json(self, out_json: Path):
        """
        Loop through metadata → build one 'full_id' per (dayID, batchPlate, wellID)
        → ask resolve_filename() once → store results in a JSON file.
        Enhanced logging for debugging.
        """
        logging.info("Generating key-mapping JSON…")

        cleaned = self.clean_metadata()
        grouped = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        mapping = {}
        
        # Statistics for debugging
        total_groups = len(grouped)
        stitched_count = 0
        found_count = 0
        
        logging.info(f"Processing {total_groups} unique combinations")

        for (day_id, batch_plate, well_id), group_df in grouped:
            logging.info(f"Processing {batch_plate} {day_id} {well_id}")

            # ───────────────────────────────────────────────────────── full_id
            parts = batch_plate.split()                    # e.g. ['Ba2', '96_1']
            ba_str = " ".join([parts[0].upper(), *parts[1:]])  # 'BA2 96_1' | 'BA4'
            raw_full_id = f"{ba_str} {day_id} {well_id}"
            full_id = clean_id_for_json(raw_full_id)
            logging.debug(f"Constructed full ID: {full_id}")

            # ───────────────────────────────────────────────────────── pick folder
            ba_part = parts[0].upper()                        # BA1 / BA2 / BA3 …
            sub = self.BA_FOLDER_MAP[ba_part]             # str or [str, str]

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
            result = self.resolve_filename(full_id, img_folder, batch_plate)
            chosen, stitched_flag, all_files, stitched_groups = result
            # extract expected well from the mapping loop vars
            expected_well = well_id  # from the grouped key (e.g., "A1")

            def has_well(fname: str, well: str) -> bool:
                return re.search(rf"\b{re.escape(well)}\b", fname, re.IGNORECASE) is not None

            # If the chosen file doesn't contain the expected well ID,
            # try to pick a candidate that *does*.
            if chosen is not None and not has_well(chosen.name, expected_well):
                good = [f for f in (all_files or []) if has_well(f.name, expected_well)]
                if good:
                    # pick best focus among the matching-well candidates
                    idx = self.find_best_focus(good)
                    chosen = good[idx if 0 <= idx < len(good) else 0]
                    stitched_flag = "No" if stitched_flag not in ("Stitched", "Multiple_Stitched") else stitched_flag
                    logging.warning(
                        f"Well mismatch: expected {expected_well}, switching to {chosen.name} "
                        f"from candidates that matched."
                    )
                else:
                    # If nothing matches the expected well, safest is to SKIP this mapping
                    logging.error(
                        f"Well mismatch for {full_id}: expected {expected_well}, "
                        f"no candidate contains it. Skipping."
                    )

            
            if chosen is None and stitched_flag != "Multiple_Stitched":
                continue  # resolve_filename already logged the failure

            # Handle multiple stitched groups - create separate entries for each
            if stitched_flag == "Multiple_Stitched" and stitched_groups:
                logging.info(f"Processing {len(stitched_groups)} stitched groups for {full_id}")
                
                for identifier, group_files in stitched_groups.items():
                    # Sort files in this group by Z-index
                    group_files.sort(key=extract_z)
                    
                    logging.info(f"  Processing stitched group '{identifier}' with {len(group_files)} files:")
                    for i, f in enumerate(group_files):
                        z_val = extract_z(f)
                        logging.info(f"    {i}: {f.name} (Z={z_val})")
                    
                    # Find best focus within this stitched group
                    best_idx = self.find_best_focus(group_files)
                    if 0 <= best_idx < len(group_files):
                        final_file = group_files[best_idx]
                        focus_idx = best_idx
                    else:
                        final_file = group_files[0]  # Fallback to first file
                        focus_idx = 0
                    
                    logging.info(f"    Best focus for group '{identifier}': {final_file.name} (idx {focus_idx})")
                    
                    # Create unique full_id for this stitched group
                    # Clean the stitched identifier for JSON key use
                    safe_identifier = re.sub(r"[^\w\s]", "", identifier).strip().replace(" ", "_")
                    stitched_full_id = f"{full_id} [{safe_identifier}]"

                    
                    # Extract actual Z-value for reference
                    z_match = re.search(r' Z(\d+)', final_file.name, re.IGNORECASE)
                    actual_z = int(z_match.group(1)) if z_match else 0
                    
                    found_count += 1
                    stitched_count += 1
                    
                    # Record this stitched group
                    clean_stitched_id = clean_id_for_json(stitched_full_id)
                    is_blank, area_frac = is_blankish_file(final_file)

                    mapping[clean_stitched_id] = {
                        "dayID": day_id,
                        "BA": ba_str,
                        "wellID": well_id,
                        "stitched_identifier": identifier,
                        "Best Z": focus_idx,
                        "Best Z Filename": str(final_file),
                        "Actual Z Value": actual_z,
                        "Classification": "Stitched",
                        "um_per_px": float(group_df["um_per_px"].iloc[0]),
                        "all_files": [str(f) for f in group_files],
                        "cellLine": group_df["cellLine"].iloc[0],
                        "treatment": group_df["treatment"].iloc[0],
                        "Blank": bool(is_blank),
                        "blank_area_frac": float(area_frac)

                    }
                
                continue  # Move to next group - we've processed all stitched variants

            # Handle single stitched file or regular files
            found_count += 1
            if stitched_flag == "Stitched":
                stitched_count += 1

            # stitched stack → find best focus among stitched files; otherwise choose best-focus Z
            if stitched_flag == "Stitched":
                focus_idx = -1
                final = chosen
                z_match = re.search(r' Z(\d+)', chosen.name, re.IGNORECASE)
                actual_z = int(z_match.group(1)) if z_match else 0
                logging.info(f"Using stitched image: {chosen.name} (Z={actual_z})")
            else:
                idx = self.find_best_focus(all_files)
                focus_idx = idx if 0 <= idx < len(all_files) else -1
                final = all_files[focus_idx] if focus_idx >= 0 else chosen
                logging.info(f"Using best focus image (idx {focus_idx}): {final.name}")

            # ───────────────────────────────────────────────────────── record row
            classification = classify_image_file(final.name)
            is_blank, area_frac = is_blankish_file(final)

            row_data = {
                "dayID": day_id,
                "BA": ba_str,
                "wellID": well_id,
                "Best Z": focus_idx,
                "Best Z Filename": str(final),
                "Classification": classification,
                "um_per_px": float(group_df["um_per_px"].iloc[0]),
                "all_files": [str(f) for f in all_files],
                "cellLine": group_df["cellLine"].iloc[0],
                "treatment": group_df["treatment"].iloc[0],
                "Blank": bool(is_blank),
                "blank_area_frac": float(area_frac)

            }

            clean_full_id = clean_id_for_json(full_id)
            mapping[clean_full_id] = row_data


        # Final statistics
        logging.info(f"=== MAPPING SUMMARY ===")
        logging.info(f"Total groups processed: {total_groups}")
        logging.info(f"Files found: {found_count}")
        logging.info(f"Stitched images detected: {stitched_count}")
        logging.info(f"Success rate: {found_count/total_groups*100:.1f}%")
        logging.info(f"Stitched rate: {stitched_count/found_count*100:.1f}%" if found_count > 0 else "Stitched rate: 0%")

        out_json.write_text(json.dumps(mapping, indent=2))
        logging.info(f"Wrote mapping JSON to {out_json}")


    def find_best_focus(self, files: list[Path]) -> int:
        if not files:
            return -1

        best_i = -1
        best_var = -1.0
        for i, f in enumerate(files):
            gray = load_gray_resized(f, FAST_EVAL_SIZE)
            if gray is None:
                continue
            # Laplacian variance on downscaled gray
            var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if var > best_var:
                best_var = var
                best_i = i
        return best_i if best_i >= 0 else 0
