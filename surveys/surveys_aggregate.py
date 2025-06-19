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
    match = re.search(r"(Ba\d+ \d+_\d+) (Dy\d+) ([A-H]\d+)", image_id)
    if match:
        return {
            "BA": match.group(1),
            "dayID": match.group(2),
            "wellID": match.group(3)
        }
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

                        if is_quality_form and image_id and quality:
                            data[organoid_id]["quality_scores"].append({
                                "image_id": image_id,
                                "quality": quality,
                                "source_file": os.path.basename(file),
                                **parsed_meta
                            })
                        elif not is_quality_form and organoid_id and evaluation and image_id:
                            data[organoid_id]["evaluations"].append({
                                "image_id": image_id,
                                "evaluation": evaluation,
                                "employee": employee_name,
                                "source_file": os.path.basename(file),
                                **parsed_meta
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
