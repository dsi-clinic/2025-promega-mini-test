# metadata_resolver.py
from __future__ import annotations

from pathlib import Path
import logging
import re
from typing import Tuple

import pandas as pd

from pipeline.common.organoid_patterns import OrganoidPatterns

def _precompute_um_per_px(meta: pd.DataFrame) -> pd.DataFrame:
    cols = list(meta.columns)
    px_candidates = [c for c in cols if "Image Width" in c and "Pixel" in c]
    um_candidates = [c for c in cols if "Image Width" in c and "µm" in c]

    if not px_candidates or not um_candidates:
        raise ValueError(f"Could not find width columns in {cols}")

    px_col = px_candidates[0]
    um_col = um_candidates[0]
    logging.debug(f"[metadata] Using pixel-col={px_col!r}, micron-col={um_col!r}")

    meta[px_col] = (
        meta[px_col]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )
    meta[um_col] = (
        meta[um_col]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    meta["um_per_px"] = meta[um_col] / meta[px_col]
    return meta


def _split_photo_id(pid: str) -> Tuple[str, str, str]:
    """
    Example inputs:
      'BA2 96_1 Dy24 H4'
      'BA3 Pt1 Dy30 B9'
    Returns:
      batchPlate, dayID, wellID
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
    well_tokens = parts[day_idx + 1 :]
    tokens = " ".join(well_tokens)

    m = OrganoidPatterns.WELL_STRICT.search(tokens)
    if m:
        wellID = f"{m.group(1).upper()}{m.group(2)}"
    else:
        m2 = re.search(r"([A-Ha-h]\s*\d{1,2})", tokens)
        wellID = m2.group(1).replace(" ", "").upper() if m2 else tokens.strip().upper()

    logging.debug(f"[metadata._split_photo_id] {pid!r} → batch={batchPlate!r}, day={dayID!r}, well={wellID!r}")
    return batchPlate, dayID, wellID


def load_and_clean_metadata(meta_csv: Path, sheet_name: str = "Images") -> pd.DataFrame:
    """
    Returns a cleaned DataFrame with:
      photoID, orgID, batchPlate, dayID, wellID,
      um_per_px, cellLine, treatment, and a few imaging cols.
    """
    meta = pd.read_excel(meta_csv, sheet_name=sheet_name)
    meta = meta.rename(
        columns={
            "Photo ID (Batch Plate Day Well)": "photoID",
            "Organoid ID (Same as in Organoid Info)": "orgID",
            "Picture Day": "dayID",
            "Objective": "objective",
            "Number of Focus": "numFocus",
            "First Focus": "firstZ",
            "Last Focus": "lastZ",
            "Focus Step (µm)": "dz",
            "Cell line": "cellLine",
            "Treatments (AAV)": "treatment",
        }
    )
    meta = _precompute_um_per_px(meta)

    bp_list = []
    dy_list = []
    well_list = []
    for pid in meta["photoID"]:
        b, d, w = _split_photo_id(str(pid))
        bp_list.append(b)
        dy_list.append(d)
        well_list.append(w)

    meta["batchPlate"] = bp_list
    meta["dayID"] = dy_list
    meta["wellID"] = well_list

    cols = [
        "photoID",
        "orgID",
        "batchPlate",
        "dayID",
        "wellID",
        "Microscope",
        "objective",
        "Image Width (Pixel)",
        "Image Width (µm)",
        "um_per_px",
        "numFocus",
        "firstZ",
        "lastZ",
        "dz",
        "cellLine",
        "treatment",
    ]

    # keep only those that exist (Excel can vary)
    cols = [c for c in cols if c in meta.columns]
    return meta[cols]
