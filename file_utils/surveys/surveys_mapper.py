import sys, os, json, glob, re
from pathlib import Path
import pandas as pd
from collections import defaultdict

# --- Locate repo root (contains paths.py and .env) ---
HERE = Path(__file__).resolve()
root = next((p for p in HERE.parents if (p / "config.py").exists() and (p / ".env").exists()), None)
if not root:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

sys.path.insert(0, str(root))
os.chdir(str(root))  # optional but helps if config.py uses relative paths

from config import SURVEY_RESULTS, SURVEY_AGGREGATED_JSON
from file_utils.common.organoid_patterns import OrganoidNormalizer, clean_id_for_json, norm_key

input_dir = str(SURVEY_RESULTS)
print("SURVEY_RESULTS =", input_dir)

def parse_image_id(image_id):
    cleaned = re.sub(r"\(.*?\)", "", image_id)       # remove parentheses
    cleaned = re.sub(r"[^A-Za-z0-9\s_]", " ", cleaned)  # replace junk chars with space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()   # normalize whitespace
    parts = cleaned.split()

    try:
        ba_idx = next(i for i, p in enumerate(parts) if re.match(r"Ba\d+", p, re.IGNORECASE))
        ba = parts[ba_idx].upper()
        plate = parts[ba_idx + 1] if ba_idx + 1 < len(parts) and re.match(r"\d+_\d+", parts[ba_idx + 1]) else ""
        dy = next(p for p in parts if re.match(r"Dy\d+", p, re.IGNORECASE))

        # match first valid well-looking thing
        well = next(p for p in parts if re.match(r"^[A-H]\d{1,2}$", p))

        return {
            "BA": f"{ba} {plate}".strip(),
            "dayID": dy,
            "wellID": well
        }
    except (IndexError, StopIteration):
        return {}


def process_organoid_files(directory):
    data = defaultdict(lambda: {"evaluations": [], "quality_scores": []})
    
    excel_files = [f for f in glob.glob(os.path.join(directory, '*.xlsx')) 
                   if ("Organoid Classification" in os.path.basename(f) or "Image Classification" in os.path.basename(f))
                   and "Organoid Classification (Form ABC)" not in f]
    print("Excel files found:", excel_files)

    for file in excel_files:
        is_quality_form = "Image Classification" in os.path.basename(file)
        try:
            df = pd.read_excel(file)

            for _, row in df.iterrows():
                employee_name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip() if not is_quality_form else None

                for col in row.index:
                    val = row[col]
                    if pd.notna(val) and isinstance(val, str) and ('Organoid_' in val or any(x in val for x in ['Ba1', 'Ba2', 'Dy'])):
                        parts = [p.strip() for p in val.split(',')]
                        organoid_id = next((p for p in parts if "Organoid_" in p), None)
                        image_id = next((p for p in parts if any(x in p for x in ['Ba1', 'Ba2', 'Dy'])), None)
                        if image_id:
                            image_id_cleaned = clean_id_for_json(image_id)
                        else:
                            image_id_cleaned = None

                        evaluation = next((p for p in parts if p in ['Acceptable', 'Not Acceptable', 'Not Loaded']), None)
                        quality = next((p for p in parts if p in ['Good', 'Bad', 'Reasonable']), None)
                        parsed_meta = parse_image_id(image_id) if image_id else {}

                        entry = {
                            "image_id": f"{parsed_meta['BA']} {parsed_meta['dayID']} {parsed_meta['wellID']}" if parsed_meta else image_id_cleaned,
                            "source_file": os.path.basename(file),
                            **parsed_meta
                        }

                        if is_quality_form and image_id and quality:
                            entry["quality"] = quality
                            data[organoid_id]["quality_scores"].append(entry)
                        elif not is_quality_form and organoid_id and evaluation and image_id:
                            entry["evaluation"] = evaluation
                            entry["employee"] = employee_name
                            data[organoid_id]["evaluations"].append(entry)

                        # Debug: log missing parsed_meta if needed
                        if parsed_meta == {}:
                            print(f"Unparsed image_id: {image_id_cleaned} from {organoid_id} in {os.path.basename(file)}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    return data

if __name__ == "__main__":
    result = process_organoid_files(input_dir)
    print(f"Final count: {len(result)} image_ids")
    SURVEY_AGGREGATED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SURVEY_AGGREGATED_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {SURVEY_AGGREGATED_JSON}")