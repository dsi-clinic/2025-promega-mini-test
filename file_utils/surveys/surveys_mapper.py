import sys, os, json, glob, re
from pathlib import Path
import pandas as pd
from collections import defaultdict

# --- Locate repo root (contains paths.py and .env) ---
HERE = Path(__file__).resolve()
root = next(
    (p for p in HERE.parents if (p / "config.py").exists() and (p / ".env").exists()),
    None,
)
if not root:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

sys.path.insert(0, str(root))
os.chdir(str(root))  # helps if config.py uses relative paths

from config import SURVEY_RESULTS, SURVEY_AGGREGATED_JSON
from file_utils.common.organoid_patterns import OrganoidNormalizer, clean_id_for_json


# --- Parse image_id into BA/day/well ---
def parse_image_id(image_id):
    cleaned = re.sub(r"\(.*?\)", "", image_id)  # remove parentheses
    cleaned = re.sub(r"[^A-Za-z0-9\s_]", " ", cleaned)  # replace junk chars with space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # normalize whitespace
    parts = cleaned.split()
    try:
        ba_idx = next(
            i for i, p in enumerate(parts) if re.match(r"Ba\d+", p, re.IGNORECASE)
        )
        ba = parts[ba_idx].upper()
        plate = (
            parts[ba_idx + 1]
            if ba_idx + 1 < len(parts) and re.match(r"\d+_\d+", parts[ba_idx + 1])
            else ""
        )
        dy = next(p for p in parts if re.match(r"Dy\d+", p, re.IGNORECASE))
        well = next(p for p in parts if re.match(r"^[A-H]\d{1,2}$", p))
        return {"BA": f"{ba} {plate}".strip(), "dayID": dy, "wellID": well}
    except (IndexError, StopIteration):
        return {}


# --- Main processor ---
def process_organoid_files(directory):
    data = defaultdict(lambda: {"evaluations": [], "quality_scores": []})

    excel_files = [
        f
        for f in glob.glob(os.path.join(directory, "*.xlsx"))
        if (
            "Organoid Classification" in os.path.basename(f)
            or "Image Classification" in os.path.basename(f)
        )
        and "Organoid Classification (Form ABC)" not in os.path.basename(f)
    ]
    print("Excel files found:", excel_files)

    for file in excel_files:
        is_quality_form = "Image Classification" in os.path.basename(file)
        basename = os.path.basename(file)
        try:
            df = pd.read_excel(file)
            for _, row in df.iterrows():
                employee_name = (
                    f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
                    if not is_quality_form
                    else None
                )

                for col in row.index:
                    val = row[col]
                    if (
                        pd.notna(val)
                        and isinstance(val, str)
                        and (
                            "Organoid_" in val
                            or any(x in val for x in ["Ba1", "Ba2", "Ba3", "Ba4", "Dy"])
                        )
                    ):
                        # ---- Core parsing ----
                        original_cell = val
                        parts = [p.strip() for p in val.split(",")]

                        organoid_id = next((p for p in parts if "Organoid_" in p), None)
                        image_id = next(
                            (
                                p
                                for p in parts
                                if any(
                                    x in p for x in ["Ba1", "Ba2", "Ba3", "Ba4", "Dy"]
                                )
                            ),
                            None,
                        )

                        if not organoid_id or not image_id:
                            continue

                        # --- Detect and strip extra tokens (INV, STITCHED, PRE/POST, etc.) ---
                        is_stitched = False

                        if image_id:
                            # detect stitch markers first
                            if re.search(
                                r"stitched|stitch", image_id, flags=re.IGNORECASE
                            ):
                                is_stitched = True

                            # remove common suffix tokens
                            image_id_core = re.sub(
                                r"\b(INV|PRE|POST|STITCH|STITCHED|STCH|Z\d+|REV|BOT|TOP|ROI)\b",
                                "",
                                image_id,
                                flags=re.IGNORECASE,
                            )
                            # also remove parenthetical annotations like "(stitched)"
                            image_id_core = re.sub(r"\(.*?\)", "", image_id_core)
                            image_id_core = re.sub(r"\s+", " ", image_id_core).strip()

                            image_id_clean = clean_id_for_json(image_id_core)
                        else:
                            image_id_clean = None
                            is_stitched = False

                        # --- Strip unwanted suffixes like INV, PRE, POST, STITCH, etc. ---
                        if image_id:
                            image_id_core = re.sub(
                                r"\b(INV|PRE|POST|STITCH|STCH|Z\d+)\b",
                                "",
                                image_id,
                                flags=re.IGNORECASE,
                            )
                            image_id_core = re.sub(
                                r"\s+", " ", image_id_core
                            ).strip()  # normalize whitespace
                            image_id_clean = clean_id_for_json(image_id_core)
                        else:
                            image_id_clean = None

                        parsed_meta = parse_image_id(image_id)

                        # detect split (safe even if none)
                        split_info = (
                            OrganoidNormalizer.extract_split_info(image_id) or {}
                        )
                        split_index = split_info.get("split_index")

                        # --- Construct main_id mirroring all_data naming ---
                        main_id_base = (
                            re.sub(r"\s+", "_", image_id_clean.strip())
                            if image_id_clean
                            else None
                        )

                        if main_id_base:
                            if split_index is not None:
                                main_id = f"{main_id_base}_split{split_index}_{'stitched' if is_stitched else 'nostitch'}"
                            else:
                                main_id = f"{main_id_base}_nosplit_{'stitched' if is_stitched else 'nostitch'}"
                        else:
                            main_id = None

                        # ---- Build entry ----
                        entry = {
                            "original_image_ref": original_cell,  # exact Excel cell text
                            "raw_organoid_id": organoid_id,
                            "image_id": image_id_clean,  # base cleaned form
                            "main_id": main_id,  # used for matching all_data
                            "split_index": split_index,
                            "source_file": basename,
                            **parsed_meta,
                        }

                        # ---- Assign category ----
                        if is_quality_form and any(
                            q in parts for q in ["Good", "Bad", "Reasonable"]
                        ):
                            entry["quality"] = next(
                                p for p in parts if p in ["Good", "Bad", "Reasonable"]
                            )
                            data[organoid_id]["quality_scores"].append(entry)

                        elif not is_quality_form and any(
                            e in parts
                            for e in ["Acceptable", "Not Acceptable", "Not Loaded"]
                        ):
                            entry["evaluation"] = next(
                                p
                                for p in parts
                                if p in ["Acceptable", "Not Acceptable", "Not Loaded"]
                            )
                            entry["employee"] = employee_name
                            data[organoid_id]["evaluations"].append(entry)

                        if parsed_meta == {}:
                            print(
                                f" Could not parse image_id: {image_id} from {organoid_id} in {basename}"
                            )

        except Exception as e:
            print(f" Error processing file {file}: {e}")
            continue

    return data


# --- Run ---
if __name__ == "__main__":
    input_dir = str(SURVEY_RESULTS)
    print("SURVEY_RESULTS =", input_dir)
    result = process_organoid_files(input_dir)
    print(f" Final organoid count: {len(result)}")
    SURVEY_AGGREGATED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SURVEY_AGGREGATED_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f" Wrote: {SURVEY_AGGREGATED_JSON}")
