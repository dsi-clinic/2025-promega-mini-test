from __future__ import annotations

import logging
import pandas as pd
import re
import json
from pathlib import Path
from tifffile import TiffFile

class ImageMapper:
    BA_FOLDER_MAP = {
        "BA1": "BA1",
        "BA2": ["BA2/96_1", "BA2/96_2"],
        "BA3": "BA3",
        "BA4": "BA4"
    }

    def __init__(self, base_dir: Path, meta_csv: Path):
        self.base_dir = Path(base_dir)
        # Read the "Images" sheet from the file they passed in:
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

        # **Coerce to numeric** (strip commas/spaces, convert to float)
        self.meta[px_col] = (
            self.meta[px_col]
                .astype(str)
                .str.replace(",", "")
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
        )
        self.meta[um_col] = (
            self.meta[um_col]
                .astype(str)
                .str.replace(",", "")
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
        )

        # Now safe to divide
        self.meta["um_per_px"] = self.meta[um_col] / self.meta[px_col]


    def clean_metadata(self) -> pd.DataFrame:
        """Return a DataFrame with one row per photoID, adding dayID, wellID, batchPlate."""
        df = self.meta.rename(columns={
            "Photo ID (Batch Plate Day Well)": "photoID",
            "Organoid ID (Same as in Organoid Info)": "orgID",
            "Picture Day": "dayID",
            "Objective": "objective",
            "Number of Focus": "numFocus",
            "First Focus": "firstZ",
            "Last Focus": "lastZ",
            "Focus Step (µm)": "dz"
        })

        # Split photoID into batchPlate, day, well
        def split_pid(pid):
            parts = pid.split()
            # e.g. ["Ba1","96_1","Dy03","A1"]
            batchPlate = " ".join(parts[:-2])
            dayID      = parts[-2]
            wellID     = parts[-1]
            return pd.Series([batchPlate, dayID, wellID])

        df[["batchPlate", "dayID", "wellID"]] = df["photoID"].apply(split_pid)

        return df[[
            "photoID", "orgID", "batchPlate", "dayID", "wellID",
            "Microscope", "objective", "Image Width (Pixel)",
            "Image Width (µm)", "um_per_px",
            "numFocus", "firstZ", "lastZ", "dz"
        ]]

    def make_mapping_json(self, out_json: Path):
        """Walk through clean metadata, detect stitched vs Z-stack, pick focus, and dump mapping."""
        cleaned = self.clean_metadata()
        mapping = {}
        stitch_re = re.compile(r"\(\d+\s*of\s*\d+\)", re.IGNORECASE)

        for _, row in cleaned.iterrows():
            pid = row.photoID

            # ── resolve img_folder for this row ──
            # row.batchPlate looks like "Ba2 96_1" or "Ba1"
            tokens = row.batchPlate.split()
            ba = tokens[0].upper()  # e.g. "BA2", "BA1", etc.

            sub = self.BA_FOLDER_MAP[ba]
            if isinstance(sub, list):
                # BA2 has two subfolders; pick the one ending in "96_1" or "96_2"
                subfolder = next(s for s in sub if s.endswith(tokens[1]))
            else:
                subfolder = sub

            img_folder = self.base_dir / subfolder / row.dayID

            # ── gather and filter TIFF files ──
            pattern    = re.compile(rf"\b{re.escape(pid)}\b", re.IGNORECASE)
            candidates = sorted(img_folder.rglob("*.tif"))
            filtered   = [f for f in candidates if pattern.search(f.name)]
            if not filtered:
                logging.warning(f"No files for {pid} in {img_folder}")
                continue

            # ── detect stitched vs Z-stack ──
            stitched_file = next((f for f in filtered if stitch_re.search(f.name)), None)
            if stitched_file:
                chosen, focus_idx, stitched_flag = stitched_file, -1, "Yes"
            else:
                focus_idx    = self.find_best_focus(filtered)
                chosen       = filtered[focus_idx]
                stitched_flag = "No"

            # ── record the mapping entry ──
            mapping[pid] = {
                "img_folder":      str(img_folder),
                "all_files":       [str(f) for f in filtered],
                "Stitched":        stitched_flag,
                "Best Z":          focus_idx,
                "Best Z Filename": str(chosen),
                "num_focus":       int(row.numFocus),
                "firstZ":          row.firstZ,
                "lastZ":           row.lastZ,
                "dz":              float(row.dz),
                "um_per_px":       float(row.um_per_px),
            }

        # write out the JSON
        out_json.write_text(json.dumps(mapping, indent=2))

    
    def find_best_focus(self, files: list[Path]) -> int:
        import cv2
        from skimage.io import imread

        best_i = 0
        best_var = -1
        for i,f in enumerate(files):
            img = imread(f)
            if img.ndim == 3:
                img = img.mean(axis=2).astype("uint8")
            var = cv2.Laplacian(img, cv2.CV_64F).var()
            if var > best_var:
                best_var = var
                best_i = i
        return best_i
