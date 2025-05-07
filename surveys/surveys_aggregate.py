import pandas as pd
import glob
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def process_organoid_files(directory):
    # Use defaultdict to store lists of results per organoid
    organoid_data = defaultdict(list)
    
    # Get all Excel files in the directory, ignoring temporary files
    excel_files = [f for f in glob.glob(os.path.join(directory, '*.xlsx')) 
                  if not f.startswith(os.path.join(directory, '~$')) and 
                  "Adjusted by Pedro" not in f]
    
    for file in excel_files:
        try:
            # Read the Excel file
            df = pd.read_excel(file)
            
            # Iterate through each row in the dataframe
            for _, row in df.iterrows():
                # Get employee information
                first_name = row.get('First Name', '')
                last_name = row.get('Last Name', '')
                employee_name = f"{first_name} {last_name}".strip()
                
                # Process columns that might contain organoid data
                for col in row.index:
                    if pd.notna(row[col]) and isinstance(row[col], str):
                        if 'Organoid_' in row[col] or any(x in row[col] for x in ['Ba1', 'Ba2', 'Dy30']):
                            # Split the organoid information
                            parts = [p.strip() for p in row[col].split(',')]

                            if "Brum" in employee_name and "291" in str(parts):
                                print(parts)
                            
                            # Find the part that contains the organoid name
                            organoid_name = None
                            for part in parts:
                                if any(x in part for x in ['Ba1', 'Ba2', 'Dy30']):
                                    organoid_name = part
                                    break
                            
                            # Find evaluation and quality if available
                            evaluation = None
                            quality = None
                            
                            for part in parts:
                                part = part.strip()
                                if part in ['Acceptable', 'Not Acceptable', 'Not Loaded']:
                                    evaluation = part
                                elif part in ['Good', 'Bad', 'Reasonable']:
                                    quality = part
                            
                            # If we found all components, add to dictionary
                            if organoid_name and evaluation and quality and employee_name:
                                entry = {
                                    'evaluation': evaluation,
                                    'quality': quality,
                                    'employee': employee_name,
                                    'source_file': os.path.basename(file),
                                    'raw_data': row[col]  # Keep original for debugging
                                }
                                organoid_data[organoid_name].append(entry)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    return organoid_data

if __name__ == "__main__":
    # Directory containing the Excel files
    #input_dir = '/net/projects2/promega/data-analysis/results_surveys/Organoid Classification (Form ABC)/'
    input_dir = os.getenv('SURVEY_RESULTS')
    print(input_dir)
    
    # Process all files
    result = process_organoid_files(input_dir)
    
    # Print summary
    print(f"Found {len(result)} unique organoids")
    print(f"Total survey responses: {sum(len(v) for v in result.values())}")
    
    # Save to a JSON file for later use
    import json
    with open('organoid_classification_results_aggregated.json', 'w') as f:
        json.dump(result, f, indent=2)
    