# verification.py
from __future__ import annotations

from pathlib import Path
import logging
import re
from typing import Dict, Tuple, List
from file_utils.common.organoid_patterns import OrganoidPatterns

import pandas as pd
from tqdm import tqdm

class Verifier:
    """
    Wraps the verification CSV:
      - maps normalized main_id -> YES/NO for blank wells
      - returns verification metadata for each mapping entry
    """

    def __init__(self, verify_csv: Path):
        self.verify_map: Dict[str, str] = {}
        self.verify_norm2orig: Dict[str, str] = {}
        self.verify_splits: Dict[str, List[str]] = {}
        self._load_csv(verify_csv)

    def _load_csv(self, verify_csv: Path) -> None:
        vdf = pd.read_csv(verify_csv)
        vdf.columns = [c.strip() for c in vdf.columns]

        for value in vdf.to_dict(orient="records"):
            if "_split" in value["main id"]:
                record_id = Verifier._extract_up_to_well(value["filename base"])
                self.verify_splits.setdefault(record_id, []).append(value["main id"])

        # main id column
        if any(c.lower() == "main id" for c in vdf.columns):
            col_main = next(c for c in vdf.columns if c.lower() == "main id")
        elif any(c.lower() == "main_id" for c in vdf.columns):
            col_main = next(c for c in vdf.columns if c.lower() == "main_id")
        else:
            raise ValueError("Verification CSV missing 'main id' / 'main_id' column.")

        # blank column
        if "Images taken from blank wells [YES/NO]" in vdf.columns:
            col_blank = "Images taken from blank wells [YES/NO]"
        else:
            cand = [c for c in vdf.columns if "blank" in c.lower()]
            if not cand:
                raise ValueError("Verification CSV missing blank YES/NO column.")
            col_blank = cand[0]

        vdf["main_id_orig"] = vdf[col_main].astype(str).str.strip()
        # Normalized key: remove spaces, collapse "_split_<n>" -> "_split<n>"
        vdf["main_id_norm"] = (
            vdf["main_id_orig"]
            .str.replace(r"\s+", "", regex=True)
            .str.replace(r"_split_(\d+)", r"_split\1", regex=True)
            .str.lower()
        )


        blank_norm = vdf[col_blank].astype(str).str.strip().str.upper()

        self.verify_map = dict(zip(vdf["main_id_norm"], blank_norm))
        self.verify_norm2orig = dict(zip(vdf["main_id_norm"], vdf["main_id_orig"]))

        logging.debug(f"[Verifier] Loaded {len(self.verify_map)} verification entries")

    @staticmethod
    def _extract_up_to_well(s: str) -> str:
        """
        Extract the string up to the well ID.

        Args:
            s: The string to extract the up to the well ID from.

        Returns:
            The string up to the well ID.
        """
        match = re.search(r'[A-H]\d{1,2}', s)
        return s[:match.end()] if match else s

    @staticmethod
    def classification_label_for_verif(split_index: int | None, classification: str) -> str:
        split = "Split" if split_index is not None else "NoSplit"
        stitch = "Stitched" if "stitched" in classification.lower() else "NoStitched"
        return f"{split}{stitch}"

    @staticmethod
    def build_main_id(
        ba_str: str,
        day_id: str,
        well_id: str,
        split_index: int | None,
        classification: str,
        presplit_flag: bool = False,
    ) -> str:
        """
        Construct the verification 'main id' string, e.g.:
        BA1_96_1_Dy03_A1_nosplit_nostitch
        BA2_96_1_Dy28_C7_split1_stitched
        BA2_96_1_Dy21_E1_presplit_nostitch
        """
        ba_token = ba_str.replace(" ", "_")
        if split_index is not None:
            split_token = f"split{int(split_index)}"
        else:
            split_token = "presplit" if presplit_flag else "nosplit"
        stitch_token = "stitched" if "stitched" in classification.lower() else "nostitch"
        return f"{ba_token}_{day_id}_{well_id}_{split_token}_{stitch_token}"

    def _candidate_keys(
        self,
        ba_str: str,
        day_id: str,
        well_id: str,
    ) -> Tuple[str, list[str]]:
        """
        Build normalized prefix and candidate keys from verify_map.
        """
        csv_key_prefix = f"{ba_str.replace(' ', '_')}_{day_id}_{well_id}"
        prefix_norm = (
            csv_key_prefix.replace(" ", "")
            .replace("__", "_")
            .lower()
        )
        prefix_norm = re.sub(r"_split_(\d+)", r"_split\1", prefix_norm)
        pattern = re.compile(
            rf"^{re.escape(prefix_norm)}_(split_?\d+|presplit|nosplit)_(stitched|nostitch)$"
        )
        keys = list(self.verify_map.keys())
        cand_keys = [k for k in keys if pattern.match(k)]
        return prefix_norm, cand_keys

    def lookup(
        self,
        ba_str: str,
        day_id: str,
        well_id: str,
        split_index: int | None,
        classification: str,
        gen_main_id: str | None,
    ) -> dict:
        """
        Return verification metadata dict:
          main_id, gen_main_id, classification_verification, blank_verified, blank
        """
        if not self.verify_map:
            return {
                "main_id": gen_main_id,
                "gen_main_id": gen_main_id,
                "classification_verification": self.classification_label_for_verif(split_index, classification),
                "blank_verified": None,
                "blank": False,
            }

        _, cand_keys = self._candidate_keys(ba_str, day_id, well_id)

        best_key = None
        if cand_keys:
            def key_score(k: str) -> tuple[int, int]:
                score = 0
                # split match
                if split_index is not None and (f"_split{split_index}_" in k or f"_split_{split_index}_" in k):
                    score += 100
                elif split_index is None:
                    if "_presplit_" in k:
                        score += 60
                    elif "_nosplit_" in k:
                        score += 50
                # stitched flag match
                if "_stitched" in k and "stitched" in classification.lower():
                    score += 10
                elif "_nostitch" in k and "stitched" not in classification.lower():
                    score += 5
                return score, len(k)

            best_key = max(cand_keys, key=key_score)

        if best_key:
            main_id_raw = self.verify_norm2orig.get(best_key, best_key)
            # normalize split_1 vs split1
            main_id = re.sub(r"(?<=_split)_(?=\d)", "", main_id_raw)
            verdict = self.verify_map[best_key]
        else:
            main_id = gen_main_id
            verdict = None

        is_blank = (verdict == "YES")

        if gen_main_id and best_key and best_key.upper() != gen_main_id.upper():
            # logging.warning(f"[Verifier] Mismatch: gen={gen_main_id} csv={best_key}")
            tqdm.write(f"WARNING: [Verifier] Mismatch: gen={gen_main_id} csv={best_key}", file=None)

        return {
            "main_id": main_id,
            "gen_main_id": gen_main_id,
            "classification_verification": self.classification_label_for_verif(split_index, classification),
            "blank_verified": verdict,
            "blank": is_blank,
        }
