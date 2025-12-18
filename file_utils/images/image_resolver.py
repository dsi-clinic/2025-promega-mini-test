# image_resolver.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, NamedTuple
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import cv2
import numpy as np


from file_utils.common.organoid_patterns import OrganoidPatterns, OrganoidNormalizer

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
    logging.debug(f"[image_resolver] Cached {len(files)} images for {img_folder}")
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


def _compute_laplacian_variance(path: Path, size: tuple[int, int] = _FAST_EVAL_SIZE) -> float:
    """
    Compute Laplacian variance for a single image file.

    Args:
        path: Path to the image file
        size: Target size for resizing

    Returns:
        Laplacian variance value, or -1.0 if image cannot be loaded
    """
    gray = _load_gray_resized(path, size)
    if gray is None:
        return -1.0
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var


def find_best_focus(files: list[Path], max_workers: Optional[int] = None) -> int:
    """
    Return index of file with highest Laplacian variance; 0 if no good read.

    Uses parallel processing to speed up image loading and focus computation.

    Args:
        files: List of image file paths to evaluate
        max_workers: Maximum number of worker threads. If None, uses os.cpu_count()

    Returns:
        Index of the file with best focus, or 0 if no valid images found
    """
    if not files:
        return -1

    # For small lists, use sequential processing to avoid overhead
    if len(files) <= 3:
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

    # Parallel processing for larger lists
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    logging.debug(f"[find_best_focus] Using {max_workers} workers")

    best_i = -1
    best_var = -1.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their indices
        future_to_index = {
            executor.submit(_compute_laplacian_variance, f): i
            for i, f in enumerate(files)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                var = future.result()
                if var > best_var:
                    best_var = var
                    best_i = i
            except Exception as e:
                logging.debug(f"[find_best_focus] Error processing {files[i]}: {e}")
                continue

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


def build_search_ids(search_id: str) -> list[str]:
    """
    Flat layout:
    - Start from the exact metadata ID.
    - If it contains BA + (96_1 / 96_2 / Pt1), also generate the other suffix variants.
    - If it contains BA with no suffix, generate variants with all suffixes.
    This handles metadata saying 'Pt1' while filenames say '96_1', etc.
    """
    base = search_id.strip()
    search_ids = [base]

    # Look for a BA token (BA1, BA2, BA3, ...)
    ba_match = re.search(r"\b(BA\d+)\b", base, re.IGNORECASE)
    # Look for a plate suffix token (96_1, 96_2, Pt1)
    suffix_match = re.search(r"\b(96_[12]|Pt1)\b", base, re.IGNORECASE)

    if ba_match:
        ba_token = ba_match.group(0)  # e.g. 'BA3'

        if suffix_match:
            # Case 1: metadata already has BA + suffix, e.g. 'BA3 Pt1'
            suffix_token = suffix_match.group(0)  # e.g. 'Pt1'
            orig = f"{ba_token} {suffix_token}"

            for alt in ("96_1", "96_2", "Pt1"):
                if alt.lower() == suffix_token.lower():
                    continue  # already have this one
                new = f"{ba_token} {alt}"
                alt_id = base.replace(orig, new, 1)
                search_ids.append(alt_id)
        else:
            # Case 2: metadata has BA but no suffix, e.g. 'BA3 Dy24 H4'
            orig_ba = ba_token
            for alt in ("96_1", "96_2", "Pt1"):
                new = f"{orig_ba} {alt}"
                alt_id = base.replace(orig_ba, new, 1)
                search_ids.append(alt_id)

    # de-dup + strip
    search_ids = list(dict.fromkeys(s.strip() for s in search_ids))
    logging.debug(f"[build_search_ids] Variants: {search_ids}")
    return search_ids


def _extract_well_from_id(s: str) -> str:
    """
    Use WELL_STRICT to pull a well like 'H4' from a string (or '' if none).
    """
    m = OrganoidPatterns.WELL_STRICT.search(s)
    if not m:
        return ""
    return f"{m.group(1).upper()}{m.group(2)}"


def find_candidates(img_folder: Path, file_photoID: str) -> list[Path]:
    files = list_image_files(img_folder)
    if not files:
        return []

    search_ids = build_search_ids(file_photoID)
    candidates: list[Path] = []

    for sid in search_ids:
        clean_sid = sid.strip()
        sid_well = _extract_well_from_id(clean_sid)

        patterns: list[str] = []
        END_CHARS = r"\s._Z()\-%#"

        # core ID patterns
        patterns.append(rf"\b{re.escape(clean_sid)}(?=[{END_CHARS}]|$)")
        patterns.append(rf"{re.escape(clean_sid)}(?=[{END_CHARS}]|$)")

        # row-only: e.g. 'Ba2 Dy24 H'
        m_row_only = re.search(r"\b([A-Ha-h])$", clean_sid)
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

        logging.debug(f"[find_candidates] Patterns for {clean_sid!r}: {patterns}")

        for pattern in patterns:
            try:
                search_re = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logging.warning(f"[find_candidates] Invalid regex pattern {pattern}: {e}")
                continue
            these = [f for f in files if search_re.search(f.name)]
            if these:
                candidates = these
                logging.debug(f"[find_candidates] Found {len(candidates)} matches for {pattern}")
                break
        if candidates:
            break

    # Fallback by well ID if nothing else worked
    if not candidates:
        well = _extract_well_from_id(file_photoID)
        if well:
            logging.debug(f"[find_candidates] Fallback by well ID: {well}")
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
    day_id: str,
    well_id: str,
    file_photoID: str,
) -> ImageMatchResult:
    """
    Main entrypoint for FLAT layout.

    All images live directly in `base_dir` (possibly with multiple days mixed),
    and day/BA/well info is encoded in the filename itself.
    """
    img_folder = base_dir
    if not img_folder.exists():
        logging.warning(f"[resolve_image] Image folder does not exist: {img_folder}")
        return ImageMatchResult(None, "No", [], None)

    candidates = find_candidates(img_folder, file_photoID)

    if not candidates:
        logging.warning(f"[resolve_image] No candidates for {file_photoID} in {img_folder}")
        return ImageMatchResult(None, "No", [], None)

    candidates.sort(key=lambda f: extract_z_level(f.name))
    groups = group_by_split(candidates)
    split_groups = {k: v for k, v in groups.items() if k is not None}

    # explicit split index requested in photoID
    req_info = OrganoidNormalizer.extract_split_info(file_photoID)
    wanted = req_info.get("split_index")

    if wanted in groups:
        chosen, label, _ = choose_best_in_group(groups[wanted])
        return ImageMatchResult(chosen, f"Split-{label}", candidates, None)

    # exactly one split group
    if len(split_groups) == 1:
        k, fs = next(iter(split_groups.items()))
        chosen, label, _ = choose_best_in_group(fs)
        return ImageMatchResult(chosen, f"Split-{label}", candidates, None)

    # non-split group available
    if None in groups and groups[None]:
        chosen, label, _ = choose_best_in_group(groups[None])
        return ImageMatchResult(chosen, label, candidates, None)

    # multiple split groups and no non-split ⇒ let caller expand
    if len(split_groups) > 1:
        stitched_groups = {
            f"split_{k}": sorted(v, key=lambda f: extract_z_level(f.name))
            for k, v in split_groups.items()
        }
        return ImageMatchResult(None, "SplitAmbiguous", candidates, stitched_groups)

    # fallback: first candidate
    return ImageMatchResult(candidates[0], "Regular", candidates, None)
