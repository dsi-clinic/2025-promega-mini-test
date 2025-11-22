# image_resolver.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, NamedTuple
import re
import logging

import cv2
import numpy as np


from file_utils.common.organoid_patterns import OrganoidPatterns, OrganoidNormalizer

log = logging.getLogger(__name__)

class ImageMatchResult(NamedTuple):
    chosen: Optional[Path]
    stitched_flag: str      # "Stitched", "Split-Stitched", "Regular", "SplitAmbiguous", "No"
    all_files: list[Path]
    stitched_groups: dict | None   # for Multiple_Stitched / SplitAmbiguous cases


# ---- Focus helpers (local, cached) ----

_FAST_EVAL_SIZE = (512, 512)
_IMG_CACHE: dict[tuple[str, tuple[int, int]], np.ndarray] = {}
# Cache image listings per folder so we don't re-scan every time
_FILE_LIST_CACHE: dict[Path, list[Path]] = {}

def list_image_files(img_folder: Path) -> list[Path]:
    """Return all .tif/.tiff files in a folder, cached by folder path."""
    if img_folder in _FILE_LIST_CACHE:
        return _FILE_LIST_CACHE[img_folder]

    files: list[Path] = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(img_folder.rglob(ext))
    files = list(dict.fromkeys(files))  # de-dup

    _FILE_LIST_CACHE[img_folder] = files
    log.info(f"[image_resolver] Cached {len(files)} images for {img_folder}")
    return files


def _load_gray_resized(path: Path, size: tuple[int, int] = _FAST_EVAL_SIZE) -> Optional[np.ndarray]:
    key = (str(path), size)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    _IMG_CACHE[key] = img
    return img


def find_best_focus(files: list[Path]) -> int:
    """Return index of file with highest Laplacian variance; 0 if no good read."""
    if not files:
        return -1
    best_i = -1
    best_var = -1.0
    for i, f in enumerate(files):
        gray = _load_gray_resized(f)
        if gray is None:
            continue
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if var > best_var:
            best_var = var
            best_i = i
    return best_i if best_i >= 0 else 0


# ---- Core helpers ----

def extract_z_level(fname: str) -> int:
    """Robust Z extract. Defaults to 0 if missing."""
    # Examples: "Ba2 96_1 Dy24 H4 Z3.tif", "Ba3 Pt1 Dy30 B9(stitched)", "Ba2 96_1 Dy30 B8(#)%"
    m = re.search(r"\bZ(\d+)\b", fname, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # fallback to previous behavior if defined
    try:
        return OrganoidNormalizer.extract_z_level(fname)
    except Exception:
        return 0


def classify_image_file(fname: str) -> str:
    """Return high-level classification label used elsewhere."""
    info = OrganoidNormalizer.extract_split_info(fname)
    f = fname.lower()

    if OrganoidPatterns.STITCHED.search(f):
        return "SplitStitched" if info.get("is_split") else "Stitched"

    if OrganoidPatterns.PARTIAL_IMAGE.search(f):
        return "SplitPartial" if info.get("is_split") else "Partial"

    if info.get("is_split"):
        return "Split"

    if OrganoidPatterns.DUPLICATE_IMAGE.search(f):
        return "Duplicate"

    return "Regular"


def get_ba_subfolder(base_dir: Path, batch_plate: str, ba_folder_map: dict) -> Path:
    """
    batch_plate: e.g. 'BA2 96_1', 'BA3 Pt1'
    BA_FOLDER_MAP: {'BA1': 'Ba1', 'BA2': ['Ba2/96_1', 'Ba2/96_2'], 'BA3': 'Ba3', ...}
    """
    parts = batch_plate.split()
    ba_token = parts[0].upper()
    plate_suffix = parts[1] if len(parts) > 1 else ""

    sub = ba_folder_map[ba_token]  # can be "Ba3" or ["Ba2/96_1", "Ba2/96_2"]

    if isinstance(sub, list):
        # BA2 example: choose folder whose suffix matches 96_1/96_2
        for candidate in sub:
            if plate_suffix and plate_suffix in candidate:
                return base_dir / candidate
        # fallback: first
        return base_dir / sub[0]

    # BA3 case: you have "Ba3" and inside it, Pt1 vs 96_1 may be encoded in filenames, not folder
    return base_dir / sub


def build_search_ids(search_id: str, batch_plate: str) -> list[str]:
    """
    search_id: e.g. 'Ba2 96_1 Dy24 H4'
    batch_plate: e.g. 'BA2 96_1', 'BA3 Pt1'

    Ported from your old resolve_filename: BA3 Pt1 ↔ 96_1 and generic plate-suffix logic.
    """
    search_ids = [search_id]

    # BA3 special case (Pt1 ↔ 96_1)
    if "ba3" in search_id.lower() and batch_plate:
        parts = batch_plate.split()
        plate_suffix = parts[1] if len(parts) > 1 else ""
        if "96_1" in plate_suffix.lower() and "96_1" not in search_id.lower():
            search_ids = [
                search_id,
                OrganoidPatterns.BA_SUBSTITUTE.sub("BA3 96_1", search_id),
                OrganoidPatterns.BA_SUBSTITUTE.sub("BA3 Pt1", search_id),
            ]
            log.info(f"[build_search_ids] BA3 variants: {search_ids}")
        elif "pt1" in plate_suffix.lower() and "pt1" not in search_id.lower():
            search_ids = [
                search_id,
                OrganoidPatterns.BA_SUBSTITUTE.sub("BA3 Pt1", search_id),
                OrganoidPatterns.BA_SUBSTITUTE.sub("BA3 96_1", search_id),
            ]
            log.info(f"[build_search_ids] BA3 variants: {search_ids}")

    # General BA plate-suffix logic (BA2 96_1 ↔ 96_2 ↔ Pt1)
    if batch_plate:
        plate_suffix_match = OrganoidPatterns.PLATE_PATTERN.search(batch_plate)
        ba_match = OrganoidPatterns.BATCH_FLEXIBLE.search(search_id)
        if ba_match and plate_suffix_match:
            base_id = search_id.strip()
            ba_part = ba_match.group(0)
            plate_suffix = plate_suffix_match.group(1)
            search_ids = [base_id]
            # If ID doesn't already specify 96_1/96_2/Pt1, add them
            if not re.search(r"\b(96_[12]|Pt1)\b", base_id, re.IGNORECASE):
                search_ids.append(
                    re.sub(
                        rf"{ba_part}\b",
                        f"{ba_part} {plate_suffix}",
                        base_id,
                        flags=re.IGNORECASE,
                    )
                )
            alt_suffix = "Pt1" if plate_suffix.lower().startswith("96_") else "96_1"
            if not re.search(rf"\b{re.escape(alt_suffix)}\b", base_id, re.IGNORECASE):
                search_ids.append(
                    re.sub(
                        rf"{ba_part}\b",
                        f"{ba_part} {alt_suffix}",
                        base_id,
                        flags=re.IGNORECASE,
                    )
                )
            log.info(f"[build_search_ids] General variants: {search_ids}")

    # de-dup + strip
    return list(dict.fromkeys(sid.strip() for sid in search_ids))


def _extract_well_from_id(s: str) -> str:
    """
    Use WELL_STRICT to pull a well like 'H4' from a string (or '' if none).
    """
    m = OrganoidPatterns.WELL_STRICT.search(s)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2)}"


def find_candidates(img_folder: Path, file_photoID: str, batch_plate: str) -> list[Path]:
    files = list_image_files(img_folder)
    if not files:
        return []

    search_ids = build_search_ids(file_photoID, batch_plate)
    candidates: list[Path] = []

    for sid in search_ids:
        clean_sid = sid.strip()

        sid_well = _extract_well_from_id(clean_sid)

        patterns: list[str] = []
        # core ID patterns
        END_CHARS = r"\s._Z()\-%#"

        patterns.append(rf"\b{re.escape(clean_sid)}(?=[{END_CHARS}]|$)")
        patterns.append(rf"{re.escape(clean_sid)}(?=[{END_CHARS}]|$)")


        # row-only: e.g. 'Ba2 96_1 Dy24 H'
        m_row_only = re.search(r'\b([A-Ha-h])$', clean_sid)
        if m_row_only:
            patterns.append(
                rf"\b{re.escape(clean_sid)}\s*(?:[1-9]|1[0-2])(?=[\s._\-()%]|$)"
            )

        # handle parentheses chunks, e.g. '(stitched)', '(#)%'
        if "(" in clean_sid and ")" in clean_sid:
            base_part = clean_sid.split("(")[0].strip()
            paren_content = OrganoidPatterns.REMOVE_PARENS.search(clean_sid)
            if paren_content:
                paren_part = paren_content.group(1)
                patterns.append(
                    rf"\b{re.escape(base_part)}\s*\([^)]*{re.escape(paren_part)}[^)]*\)"
                )
                patterns.append(rf"\b{re.escape(base_part)}\s*\([^)]*\)")

        # well-based patterns
        if sid_well:
            patterns.append(rf"\b{re.escape(sid_well)}\s*\([^)]*\)")
            patterns.append(rf"\b{re.escape(sid_well)}(?=[\s._(]|$)")

        log.debug(f"[find_candidates] Patterns for {clean_sid!r}: {patterns}")

        for pattern in patterns:
            try:
                search_re = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                log.warning(f"[find_candidates] Invalid regex pattern {pattern}: {e}")
                continue
            these = [f for f in files if search_re.search(f.name)]
            if these:
                candidates = these
                log.info(f"[find_candidates] Found {len(candidates)} matches for {pattern}")
                break
        if candidates:
            break

    # Fallback by well ID if nothing else worked
    if not candidates:
        well = _extract_well_from_id(file_photoID)
        if well:
            log.info(f"[find_candidates] Fallback by well ID: {well}")
            by_well = [f for f in files if re.search(rf"\b{re.escape(well)}\b", f.name, re.IGNORECASE)]
            if not by_well:
                by_well = [f for f in files if well.lower() in f.name.lower()]
            candidates = by_well

    return candidates



def group_by_split(candidates: list[Path]) -> dict[int | None, list[Path]]:
    groups: dict[int | None, list[Path]] = {}
    for f in candidates:
        info = OrganoidNormalizer.extract_split_info(f.name)
        if info.get("is_split"):
            groups.setdefault(info["split_index"], []).append(f)
        else:
            groups.setdefault(None, []).append(f)
    return groups


def choose_best_in_group(files: list[Path]) -> tuple[Path, str, list[Path]]:
    files = sorted(files, key=lambda f: extract_z_level(f.name))
    stitched = [f for f in files if "(stitched)" in f.name.lower()]
    if stitched:
        return stitched[0], "Stitched", files
    partial = [f for f in files if OrganoidPatterns.PARTIAL_IMAGE.search(f.name)]
    if partial:
        idx = find_best_focus(partial)
        return partial[idx], "Partial", files
    idx = find_best_focus(files)
    return files[idx], "Regular", files


def resolve_image(
    base_dir: Path,
    batch_plate: str,
    day_id: str,
    well_id: str,
    file_photoID: str,
    ba_folder_map: dict,
) -> ImageMatchResult:
    """
    Main entrypoint.

    Returns:
        ImageMatchResult(
            chosen,        # the single "representative" file, or None
            stitched_flag, # "Stitched", "Split-Stitched", "Regular", "SplitAmbiguous", "No"
            all_files,     # all candidates that matched this ID
            stitched_groups, # used when multiple split groups exist and caller must expand
        )
    """
    img_folder = get_ba_subfolder(base_dir, batch_plate, ba_folder_map) / day_id
    if not img_folder.exists():
        log.warning(f"[resolve_image] Image folder does not exist: {img_folder}")
        return ImageMatchResult(None, "No", [], None)

    well_id_strict = well_id or _extract_well_from_id(file_photoID)

    candidates = find_candidates(img_folder, file_photoID, batch_plate)

    if not candidates:
        log.warning(f"[resolve_image] No candidates for {file_photoID} in {img_folder}")
        return ImageMatchResult(None, "No", [], None)

    candidates.sort(key=lambda f: extract_z_level(f.name))
    groups = group_by_split(candidates)
    split_groups = {k: v for k, v in groups.items() if k is not None}

    # Case: explicit split index requested in photoID
    req_info = OrganoidNormalizer.extract_split_info(file_photoID)
    wanted = req_info.get("split_index")

    if wanted in groups:
        chosen, label, _ = choose_best_in_group(groups[wanted])
        return ImageMatchResult(chosen, f"Split-{label}", candidates, None)

    if len(split_groups) == 1:
        k, fs = next(iter(split_groups.items()))
        chosen, label, _ = choose_best_in_group(fs)
        return ImageMatchResult(chosen, f"Split-{label}", candidates, None)

    if None in groups and groups[None]:
        chosen, label, _ = choose_best_in_group(groups[None])
        return ImageMatchResult(chosen, label, candidates, None)

    if len(split_groups) > 1:
        stitched_groups = {
            f"split_{k}": sorted(v, key=lambda f: extract_z_level(f.name))
            for k, v in split_groups.items()
        }
        return ImageMatchResult(None, "SplitAmbiguous", candidates, stitched_groups)

    # fallback: just take first candidate
    return ImageMatchResult(candidates[0], "Regular", candidates, None)
