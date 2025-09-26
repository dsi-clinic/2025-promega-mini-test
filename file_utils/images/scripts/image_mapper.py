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
from file_utils.common.organoid_patterns import OrganoidPatterns, OrganoidNormalizer, clean_id_for_json


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
    return OrganoidNormalizer.extract_z_level(f.name)
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

def split_info_for_file(f: Path) -> dict:
    return OrganoidNormalizer.extract_split_info(f.name)  # already in your Normalizer

def group_by_split(candidates: list[Path]) -> dict[int|None, list[Path]]:
    groups: dict[int|None, list[Path]] = {}
    for f in candidates:
        info = split_info_for_file(f)
        if info.get("is_split"):
            groups.setdefault(info["split_index"], []).append(f)
        else:
            groups.setdefault(None, []).append(f)  # unsplit
    return groups

def choose_best_in_group(files: list[Path]) -> tuple[Path, str, list[Path]]:
    files = sorted(files, key=extract_z)
    stitched = [f for f in files if "(stitched)" in f.name.lower()]
    if stitched:
        return stitched[0], "Stitched", files
    partial = [f for f in files if OrganoidPatterns.PARTIAL_IMAGE.search(f.name)]
    if partial:
        idx = find_best_focus(partial) if 'find_best_focus' in globals() else 0
        return partial[idx], "Partial", files
    regular = [f for f in files if classify_image_file(f.name) == "Regular"]
    if regular:
        return regular[0], "Regular", files
    return files[0], "Regular", files


def classify_image_file(fname: str) -> str:
    info = OrganoidNormalizer.extract_split_info(fname)  # unified split parsing
    f = fname.lower()

    if OrganoidPatterns.STITCHED.search(f):
        return "SplitStitched" if info["is_split"] else "Stitched"

    if OrganoidPatterns.PARTIAL_IMAGE.search(f):
        return "SplitPartial" if info["is_split"] else "Partial"

    if info["is_split"]:
        return "Split"

    if OrganoidPatterns.DUPLICATE_IMAGE.search(f):
        return "Duplicate"

    return "Regular"


# clean_id_for_json now imported from organoid_patterns module

class ImageMapper:
    BA_FOLDER_MAP = {
        "BA1": "Ba1",
        "BA2": ["Ba2/96_1", "Ba2/96_2"],
        "BA3": "Ba3",
        "BA4": "Ba4"
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
            Break 'Ba2 96_1 Dy21 D12(1 of 2) #% Z0.tif' into:
            batchPlate = 'Ba2 96_1' | 'Ba1' | 'Ba3 Pt1' …
            dayID      = 'Dy21'
            wellID     = 'D12'
            """
            parts = pid.split()

            # batchPlate
            if len(parts) > 1 and OrganoidPatterns.PLATE_PATTERN.match(parts[1]):
                batchPlate = f"{parts[0]} {parts[1]}"
                day_idx = 2
            else:
                batchPlate = parts[0]
                day_idx = 1

            # dayID
            dayID = parts[day_idx]

            # WELL: take everything after day to parse well
            well_tokens = parts[day_idx + 1:]              # <-- missing before
            tokens = " ".join(well_tokens)

            m = OrganoidPatterns.WELL_STRICT.search(tokens)
            if m:
                wellID = f"{m.group(1).upper()}{m.group(2)}"  # e.g., 'E10'
            else:
                # fallback like 'H9(1 of 2)' -> 'H9'
                m2 = re.search(r'([A-Ha-h]\s*\d{1,2})', tokens)
                wellID = m2.group(1).replace(" ", "").upper() if m2 else tokens.strip().upper()

            logging.debug(f"[split_pid] {pid!r} → batch={batchPlate!r}, day={dayID!r}, well={wellID!r}")
            return pd.Series([batchPlate, dayID, wellID])



        df[["batchPlate", "dayID", "wellID"]] = df["photoID"].apply(split_pid)

        return df[[
            "photoID", "orgID", "batchPlate", "dayID", "wellID",
            "Microscope", "objective", "Image Width (Pixel)",
            "Image Width (µm)", "um_per_px", "numFocus", "firstZ", "lastZ", "dz",
            "cellLine", "treatment"
        ]]


    def resolve_filename(self, file_photoID: str, img_folder: str|Path, batch_plate: str = None):
        img_folder = Path(img_folder)
        if not img_folder.exists():
            logging.warning(f"[resolve_filename] Folder missing: {img_folder}")
            return None, "No", [], None

        # --- gather files ONCE (case-sensitive extensions) ---
        files: list[Path] = []
        files.extend(img_folder.rglob("*.tif"))
        files.extend(img_folder.rglob("*.tiff"))
        files.extend(img_folder.rglob("*.TIF"))
        files.extend(img_folder.rglob("*.TIFF"))
        # de-dupe while preserving order
        files = list(dict.fromkeys(files))

        logging.info(f"[resolve_filename] Scanned {img_folder} → {len(files)} image files")
        if not files:
            return None, "No", [], None

        # Strict well ID
        m = OrganoidPatterns.WELL_STRICT.search(file_photoID)
        well_id = f"{m.group(1).upper()}{m.group(2)}" if m else ""
        search_id = file_photoID

        # ALWAYS start with a default
        search_ids = [search_id]

        # BA3 special case (guard batch_plate; compare in lowercase)
        if (
            "ba3" in search_id.lower()
            and batch_plate
            and "96_1" in batch_plate.lower()
            and "96_1" not in search_id.lower()
        ):
            search_ids = [
                search_id,
                OrganoidPatterns.BA_SUBSTITUTE.sub("BA3 96_1", search_id),
                OrganoidPatterns.BA_SUBSTITUTE.sub("BA3 Pt1", search_id),
            ]
            logging.info(f"Using multiple search patterns for BA3: {search_ids}")
        elif batch_plate:
            plate_suffix_match = OrganoidPatterns.PLATE_PATTERN.search(batch_plate)
            ba_match = OrganoidPatterns.BATCH_FLEXIBLE.search(search_id)
            if ba_match and plate_suffix_match:
                base_id = search_id.strip()
                ba_part = ba_match.group(0)
                plate_suffix = plate_suffix_match.group(1)
                search_ids = [base_id]
                if not re.search(r"\b(96_[12]|Pt1)\b", base_id, re.IGNORECASE):
                    search_ids.append(re.sub(rf"{ba_part}\b", f"{ba_part} {plate_suffix}", base_id, flags=re.IGNORECASE))
                alt_suffix = "Pt1" if plate_suffix.lower().startswith("96_") else "96_1"
                if not re.search(rf"\b{re.escape(alt_suffix)}\b", base_id, re.IGNORECASE):
                    search_ids.append(re.sub(rf"{ba_part}\b", f"{ba_part} {alt_suffix}", base_id, flags=re.IGNORECASE))
                logging.info(f"Using multiple search patterns: {search_ids}")

            else:
                search_ids = [search_id]


        # Try all search IDs until we find matches
        candidates = []
        for sid in search_ids:
            # DON'T strip special characters - keep the full identifier
            clean_sid = sid.strip()

            sid_well = None
            m = OrganoidPatterns.WELL_STRICT.search(clean_sid)
            if m:
                sid_well = f"{m.group(1).upper()}{m.group(2)}" if m else None

            
            # Create multiple search patterns to handle different file naming conventions
            patterns = []
            # original patterns
            patterns.append(rf"\b{re.escape(clean_sid)}(?=[\s._Z(]|$)")
            patterns.append(rf"{re.escape(clean_sid)}(?=[\s._Z]|$)")

            # NEW: if key ends with a single row letter, allow well digits after it
            m_row_only = re.search(r'\b([A-Ha-h])$', clean_sid)
            if m_row_only:
                # match e.g. “… Dy30 D12(1)% Z0.tif” when key is “… Dy30 D”
                patterns.append(
                    rf"\b{re.escape(clean_sid)}\s*(?:[1-9]|1[0-2])(?=[\s._\-()%]|$)"
                )
            
            # 3. Handle cases where special characters might be represented differently
            if '(' in clean_sid and ')' in clean_sid:
                base_part = clean_sid.split('(')[0].strip()
                paren_content = OrganoidPatterns.REMOVE_PARENS.search(clean_sid)
                if paren_content:
                    paren_part = paren_content.group(1)
                    patterns.append(rf"\b{re.escape(base_part)}\s*\([^)]*{re.escape(paren_part)}[^)]*\)")
                    patterns.append(rf"\b{re.escape(base_part)}\s*\([^)]*\)")
            
            # 4. Fallback pattern - match well ID with flexible stitched patterns
            if sid_well:
                # e.g., H9(1 of 2), H9(2), H9(stitched)
                patterns.append(rf"\b{re.escape(sid_well)}\s*\([^)]*\)")
                # plain well token like 'H9' before space/underscore/dot/paren/end
                patterns.append(rf"\b{re.escape(sid_well)}(?=[\s._(]|$)")


            
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
        
        # Group all matches by split index (None = unsplit day)
        groups: dict[int|None, list[Path]] = {}
        for f in candidates:
            info = OrganoidNormalizer.extract_split_info(f.name)
            if info.get("is_split"):
                groups.setdefault(info["split_index"], []).append(f)
            else:
                groups.setdefault(None, []).append(f)

        # If the requested photoID explicitly names a child (e.g., ...C1(2)%...), honor it
        req_info = OrganoidNormalizer.extract_split_info(file_photoID)
        wanted = req_info.get("split_index")  # None if not specified

        split_groups = {k: v for k, v in groups.items() if k is not None}

        wanted = req_info.get("split_index")  # None if not specified

        if wanted in groups:
            chosen, label, group_files = choose_best_in_group(groups[wanted])
            return chosen, f"Split-{label}", candidates, {"split_index": wanted}

        # If there’s exactly one split group present, pick from it
        split_groups = {k: v for k, v in groups.items() if k is not None}
        if len(split_groups) == 1:
            k, files_k = next(iter(split_groups.items()))
            chosen, label, group_files = choose_best_in_group(files_k)
            return chosen, f"Split-{label}", candidates, {"split_index": k}

        # If no split groups exist, fall back to unsplit group
        if None in groups and groups[None]:
            chosen, label, group_files = choose_best_in_group(groups[None])
            return chosen, label, group_files, None

        # Multiple split groups but caller didn’t specify which → ambiguous
        if len(split_groups) > 1:
            stitched_groups = {f"split_{k}": sorted(v, key=extract_z) for k, v in split_groups.items()}
            return None, "SplitAmbiguous", candidates, stitched_groups

        # (Optional) exact-file checks: “... Z0.tif” or bare “.tif”
        for sid in search_ids:
            z0 = img_folder / f"{sid} Z0.tif"
            bare = img_folder / f"{sid}.tif"
            if z0.is_file():
                logging.info(f"Found Z0 file: {z0.name}")
                return z0, "No", candidates, None
            if bare.is_file():
                logging.info(f"Found bare .tif file: {bare.name}")
                return bare, "No", candidates, None

        # Fallback: try restricting by well token in the filename
        if well_id:
            by_well = [f for f in candidates if re.search(rf"\b{re.escape(well_id)}\b", f.name, re.IGNORECASE)]
            if by_well:
                logging.info(f"Using fallback candidate: {by_well[0].name}")
                return by_well[0], "No", candidates, None

        # Final fallback (don’t put anything after this!)
        if candidates:
            return candidates[0], "Regular", candidates, None

        logging.warning(f"No files found for {file_photoID} in {img_folder}")
        return None, "No", [], None


    def make_mapping_json(self, out_json: Path):
        """
        Build mapping and write a wrapped JSON:
        {
            "_base_folder": "<absolute prefix>",
            "entries": { ... per-image entries with RELATIVE paths ... }
        }
        """
        import os

        def to_rel(p: Path) -> Path:
            """Return p relative to base_dir if possible; otherwise best-effort relpath."""
            p = Path(p)
            try:
                return p.relative_to(self.base_dir)
            except ValueError:
                # Fall back to a relative path string (handles symlinks/mounts)
                return Path(os.path.relpath(p, self.base_dir))

        logging.info("Generating key-mapping JSON…")

        cleaned = self.clean_metadata()
        grouped = cleaned.groupby(["dayID", "batchPlate", "wellID"])
        mapping: dict[str, dict] = {}

        total_groups = len(grouped)
        stitched_count = 0
        found_count = 0
        logging.info(f"Processing {total_groups} unique combinations")

        for (day_id, batch_plate, well_id), group_df in grouped:
            logging.info(f"Processing {batch_plate} {day_id} {well_id}")

            parts = batch_plate.split()
            ba_str = " ".join([parts[0].upper(), *parts[1:]])
            raw_full_id = f"{ba_str} {day_id} {well_id}"
            full_id = clean_id_for_json(raw_full_id)

            ba_part = parts[0].upper()
            sub = self.BA_FOLDER_MAP[ba_part]  # str or [str, str]
            if isinstance(sub, list):
                plate_suffix = parts[1] if len(parts) > 1 else ""
                sub = next((s for s in sub if plate_suffix in s), sub[0])

            img_folder = self.base_dir / sub / day_id
            if not img_folder.exists():
                logging.warning(f"Image folder does not exist: {img_folder}")
                continue

            chosen, stitched_flag, all_files, stitched_groups = self.resolve_filename(
                raw_full_id, img_folder, batch_plate
            )

            expected_well = well_id

            def has_well(fname: str, well: str) -> bool:
                return re.search(rf"\b{re.escape(well)}\b", fname, re.IGNORECASE) is not None

            if chosen is not None and not has_well(chosen.name, expected_well):
                good = [f for f in (all_files or []) if has_well(f.name, expected_well)]
                if good:
                    idx = self.find_best_focus(good)
                    chosen = good[idx if 0 <= idx < len(good) else 0]
                    if stitched_flag not in ("Stitched", "Multiple_Stitched"):
                        stitched_flag = "No"
                    logging.warning(
                        f"Well mismatch: expected {expected_well}, switching to {chosen.name}"
                    )
                else:
                    logging.error(
                        f"Well mismatch for {raw_full_id}: expected {expected_well}, none matched. Skipping."
                    )

            if chosen is None and stitched_flag not in ("Multiple_Stitched", "SplitAmbiguous"):
                continue
            
            # We purposely regroup on *all_files* which now always contains every match
            split_groups_all = group_by_split(all_files)      # {None: [... unsplit ...], 1: [...], 2: [...]}
            child_groups = {k: v for k, v in split_groups_all.items() if k is not None}

            if len(child_groups) >= 1:
                logging.info(f"Expanding into {len(child_groups)} split children for {full_id}")

                def pick_rep_file(files_for_child):
                    files_for_child = sorted(files_for_child, key=extract_z)
                    stitched = [f for f in files_for_child if "(stitched)" in f.name.lower()]
                    if stitched:
                        return stitched[0]
                    partials = [f for f in files_for_child if OrganoidPatterns.PARTIAL_IMAGE.search(f.name)]
                    if partials:
                        best_idx = self.find_best_focus(partials)
                        return partials[best_idx if 0 <= best_idx < len(partials) else 0]
                    best_idx = self.find_best_focus(files_for_child)
                    return files_for_child[best_idx if 0 <= best_idx < len(files_for_child) else 0]

                # (optional) update counters if you want the summary to reflect each child
                # found_count += len(child_groups)
                # stitched_count += sum(1 for _, gf in child_groups.items()
                #                       if any("(stitched)" in f.name.lower() for f in gf))

                for child_idx, group_files in sorted(child_groups.items()):
                    final_file = pick_rep_file(group_files)
                    clean_child_key = f"{full_id} split_{int(child_idx)}"

                    
                    actual_z = OrganoidNormalizer.extract_z_level(final_file.name)
                    classification = classify_image_file(final_file.name)
                    is_blank, area_frac = is_blankish_file(final_file)

                    mapping[clean_child_key] = {
                        "dayID": day_id,
                        "BA": ba_str,
                        "wellID": well_id,
                        "split_index": int(child_idx),
                        "Best Z": -1,
                        "Best Z Filename": str(to_rel(final_file)),
                        "Actual Z Value": actual_z,
                        "Classification": classification,
                        "um_per_px": float(group_df["um_per_px"].iloc[0]),
                        "all_files": [str(to_rel(f)) for f in sorted(group_files, key=extract_z)],
                        "cellLine": group_df["cellLine"].iloc[0],
                        "treatment": group_df["treatment"].iloc[0],
                        "Blank": bool(is_blank),
                        "blank_area_frac": float(area_frac),
                    }

                # We’ve created explicit child entries; skip the single parent entry
                continue


            # --- Multiple stitched groups: create one entry per group
            if stitched_flag == "Multiple_Stitched" and stitched_groups:
                logging.info(f"Processing {len(stitched_groups)} stitched groups for {full_id}")
                for identifier, group_files in stitched_groups.items():
                    group_files.sort(key=extract_z)
                    best_idx = self.find_best_focus(group_files)
                    final_file = group_files[best_idx] if 0 <= best_idx < len(group_files) else group_files[0]

                    safe_identifier = re.sub(r"[^\w\s]", "", identifier).strip().replace(" ", "_")
                    clean_stitched_id = f"{full_id} stitched_{safe_identifier}"

                    actual_z = OrganoidNormalizer.extract_z_level(final_file.name)

                    found_count += 1
                    stitched_count += 1

                    is_blank, area_frac = is_blankish_file(final_file)

                    mapping[clean_stitched_id] = {
                        "dayID": day_id,
                        "BA": ba_str,
                        "wellID": well_id,
                        "stitched_identifier": identifier,
                        "Best Z": best_idx,
                        "Best Z Filename": str(to_rel(final_file)),                 # RELATIVE
                        "Actual Z Value": actual_z,
                        "Classification": "Stitched",
                        "um_per_px": float(group_df["um_per_px"].iloc[0]),
                        "all_files": [str(to_rel(f)) for f in group_files],         # RELATIVE
                        "cellLine": group_df["cellLine"].iloc[0],
                        "treatment": group_df["treatment"].iloc[0],
                        "Blank": bool(is_blank),
                        "blank_area_frac": float(area_frac),
                    }
                continue  # done with this (day,ba,well)

            # --- Single stitched or regular case
            found_count += 1
            if stitched_flag == "Stitched":
                stitched_count += 1

            if stitched_flag == "Stitched":
                focus_idx = -1
                final = chosen
                actual_z = OrganoidNormalizer.extract_z_level(chosen.name)
            else:
                idx = self.find_best_focus(all_files)
                focus_idx = idx if 0 <= idx < len(all_files) else -1
                final = all_files[focus_idx] if focus_idx >= 0 else chosen
                actual_z = OrganoidNormalizer.extract_z_level(final.name)

            classification = classify_image_file(final.name)
            is_blank, area_frac = is_blankish_file(final)

            mapping[full_id] = {
                "dayID": day_id,
                "BA": ba_str,
                "wellID": well_id,
                "Best Z": focus_idx,
                "Best Z Filename": str(to_rel(final)),                  # RELATIVE
                "Actual Z Value": actual_z,
                "Classification": classification,
                "um_per_px": float(group_df["um_per_px"].iloc[0]),
                "all_files": [str(to_rel(f)) for f in all_files],       # RELATIVE
                "cellLine": group_df["cellLine"].iloc[0],
                "treatment": group_df["treatment"].iloc[0],
                "Blank": bool(is_blank),
                "blank_area_frac": float(area_frac),
            }

        # Final stats
        logging.info("=== MAPPING SUMMARY ===")
        logging.info(f"Total groups processed: {total_groups}")
        logging.info(f"Files found: {found_count}")
        logging.info(f"Stitched images detected: {stitched_count}")
        logging.info(f"Success rate: {found_count/total_groups*100:.1f}%")
        logging.info(f"Stitched rate: {stitched_count/max(found_count,1)*100:.1f}%")

        # --- WRAP and write
        wrapped = {
            "_base_folder": str(self.base_dir.resolve()),
            "entries": mapping,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(wrapped, indent=2))
        logging.info(f"Wrote mapping JSON (wrapped) to {out_json}")


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