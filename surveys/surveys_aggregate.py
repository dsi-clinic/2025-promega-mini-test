import pandas as pd
import glob
import os
from collections import defaultdict
from dotenv import load_dotenv, find_dotenv
import json

# Load environment variables
load_dotenv(find_dotenv(), override=True)
input_dir = os.getenv('SURVEY_RESULTS')
print("SURVEY_RESULTS =", input_dir)

def process_organoid_files(directory):
    organoid_data = defaultdict(list)

    excel_files = [f for f in glob.glob(os.path.join(directory, '*.xlsx')) 
                   if ("Organoid Classification" in os.path.basename(f) or "Image Classification" in os.path.basename(f))
                   and "Organoid Classification (Form ABC)" not in f]

    print("Excel files found:", excel_files)

    for file in excel_files:
        try:
            df = pd.read_excel(file)

            # Determine the name column
            name_col = None
            for col in df.columns:
                if 'Name' in col and 'First' in col and 'Last' in col:
                    name_col = col
                    break

            for _, row in df.iterrows():
                # Prefer structured columns if available
                if name_col:
                    employee_name = row.get(name_col, '').strip()
                else:
                    first = row.get('First Name', '').strip()
                    last = row.get('Last Name', '').strip()
                    employee_name = f"{first} {last}".strip()

                for col in row.index:
                    value = row[col]
                    if pd.notna(value) and isinstance(value, str):
                        if 'Organoid_' in value or 'Acquaintance fase' in value:
                            parts = [p.strip() for p in value.split(',')]

                            organoid_id = None
                            image_id = None
                            evaluation = None
                            quality = None

                            for part in parts:
                                if 'Organoid_' in part:
                                    organoid_id = part
                                elif 'Ba' in part and 'Dy' in part:
                                    image_id = part
                                elif part in ['Acceptable', 'Not Acceptable', 'Not Loaded']:
                                    evaluation = part
                                elif part in ['Good', 'Bad', 'Reasonable']:
                                    quality = part

                            if organoid_id and image_id and (evaluation or quality):
                                organoid_data[organoid_id].append({
                                    'image_id': image_id,
                                    'evaluation': evaluation,
                                    'quality': quality,
                                    'employee': employee_name,
                                    'source_file': os.path.basename(file),
                                    'raw_data': value
                                })
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    return organoid_data

# Execute processing
result = process_organoid_files(input_dir)

# Summary print
print(f"Found {len(result)} unique organoids")
print(f"Total survey responses: {sum(len(v) for v in result.values())}")

# Save to JSON
output_path = 'organoid_classification_results_aggregated.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

output_path
