import json
import os

# Define the paths to the new BA2 mapping files
ba2_mapping_files = [
    '/net/projects2/promega/data-analysis/output/processed_dataset_256x192/BA2_96_1_Dy30/image_mapping_BA2_96_1_Dy30_processed.json',
    '/net/projects2/promega/data-analysis/output/processed_dataset_256x192/BA2_96_2_Dy30/image_mapping_BA2_96_2_Dy30_processed.json'
]

# Load the organoid analysis results
try:
    with open('organoid_analysis_results.json') as f:
        analysis_data = json.load(f)
except FileNotFoundError:
    print("Error: 'organoid_analysis_results.json' not found.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON from 'organoid_analysis_results.json'.")
    exit()

# Load the initial image mapping (for BA1)
image_mapping = {}
ba1_mapping_path = '/net/projects2/promega/data-analysis/output/processed_dataset_256x192/image_mapping_day30_manual_processed_256x192.json'
try:
    with open(ba1_mapping_path) as f:
        image_mapping.update(json.load(f))
except FileNotFoundError:
    print(f"Warning: BA1 image mapping file '{ba1_mapping_path}' not found. Continuing without BA1 mappings.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from BA1 image mapping file '{ba1_mapping_path}'.")
    exit()

# Load the BA2 image mappings
for ba2_file in ba2_mapping_files:
    try:
        with open(ba2_file) as f:
            ba2_mapping = json.load(f)
            # Update the main image mapping with BA2 mappings, adjusting the mask path key
            for key, value in ba2_mapping.items():
                image_mapping[key] = {
                    'img_path': value['img_path'],
                    'seg_map_path': value['mask_path']
                }
    except FileNotFoundError:
        print(f"Warning: BA2 image mapping file '{ba2_file}' not found. Skipping.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from BA2 image mapping file '{ba2_file}'. Skipping.")

# Initialize the new mapping dictionary
labeled_image_mapping = {}

# Iterate through the organoid analysis results
for organoid_name, analysis in analysis_data['analysis'].items():
    if analysis['agreement_level'] == 'complete':
        # Get the classification label
        classifications = analysis['classifications']
        if classifications:
            # Assuming complete agreement means only one classification exists with a non-zero count
            for label, data in classifications.items():
                if data['count'] > 0:
                    label_to_use = label
                    break
            else:
                print(f"Warning: Organoid '{organoid_name}' has complete agreement but no classification label found. Skipping.")
                continue

            # Check if the organoid name exists in the image mapping
            if organoid_name in image_mapping:
                labeled_image_mapping[organoid_name] = {
                    'img_path': image_mapping[organoid_name]['img_path'],
                    'seg_map_path': image_mapping[organoid_name]['seg_map_path'],
                    'label': label_to_use
                }
            else:
                print(f"Warning: Organoid '{organoid_name}' found in analysis but not in image mapping. Skipping.")
        else:
            print(f"Warning: Organoid '{organoid_name}' has complete agreement but no classifications found. Skipping.")
            
# Save the new labeled image mapping to a JSON file
output_file = 'labeled_organoid_mapping_for_classification.json'
with open(output_file, 'w') as f:
    json.dump(labeled_image_mapping, f, indent=2)

print(f"\nLabeled organoid mapping saved to '{output_file}'.")
print(f"Total number of labeled organoids: {len(labeled_image_mapping)}")