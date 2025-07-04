import pandas as pd
import glob
import os
import re
import json
from collections import defaultdict
from dotenv import load_dotenv, find_dotenv

# Load .env with override
load_dotenv(find_dotenv(), override=True)
input_dir = os.getenv("SURVEY_RESULTS")
print("SURVEY_RESULTS =", input_dir)


def parse_image_id(image_id):
    # Normalize: remove parentheses and non-alphanumerics (except underscore), normalize whitespace
    cleaned = re.sub(r"\(.*?\)", "", image_id)         # Remove parenthetical text
    cleaned = re.sub(r"[^A-Za-z0-9\s_]", "", cleaned)  # Remove special chars except underscore
    cleaned = re.sub(r"\s+", " ", cleaned).strip()     # Normalize whitespace

    parts = cleaned.split()

    try:
        # Extract BA (e.g., "Ba2") and make uppercase
        ba_idx = next(i for i, p in enumerate(parts) if re.match(r"Ba\d+", p, re.IGNORECASE))
        ba = parts[ba_idx].upper()

        # Check if next part is a plate number like "96_2"
        plate = ""
        if ba == "BA2" and ba_idx + 1 < len(parts) and re.match(r"\d+_\d+", parts[ba_idx + 1]):
            plate = parts[ba_idx + 1]

        full_ba = f"{ba} {plate}".strip()

        # Extract dayID (e.g., Dy30) and wellID (e.g., H11)
        dy = next(p for p in parts if re.match(r"Dy\d+", p, re.IGNORECASE))
        well = next(p for p in parts if re.match(r"^[A-H]\d{1,2}$", p, re.IGNORECASE))

        return {
            "BA": full_ba,
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
                        evaluation = next((p for p in parts if p in ['Acceptable', 'Not Acceptable', 'Not Loaded']), None)
                        quality = next((p for p in parts if p in ['Good', 'Bad', 'Reasonable']), None)
                        parsed_meta = parse_image_id(image_id) if image_id else {}

                        # Set common info once per organoid_id
                        if parsed_meta:
                            data[organoid_id]["parsed_id"] = parsed_meta
                            data[organoid_id]["image_id"] = image_id

                        # Now just log individual scores or evals
                        if is_quality_form and image_id and quality:
                            data[organoid_id]["quality_scores"].append({
                                "quality": quality,
                                "source_file": os.path.basename(file)
                            })
                        elif not is_quality_form and organoid_id and evaluation and image_id:
                            data[organoid_id]["evaluations"].append({
                                "evaluation": evaluation,
                                "employee": employee_name,
                                "source_file": os.path.basename(file)
                            })


        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    return data

if __name__ == "__main__":
    result = process_organoid_files(input_dir)
    print(f"Final count: {len(result)} organoids")
    with open('organoid_surveys_aggregated.json', 'w') as f:
        json.dump(result, f, indent=2)
