import os
import json
import re

# File paths
original_mapping_path = '/net/projects2/promega/data-analysis/output/image_mapping.json'
new_masks_folder = '/net/projects2/promega/data-analysis/manual_masks/Manuais'
new_mapping_path = '/net/projects2/promega/data-analysis/output/image_mapping_day30_manual.json'

# Load the original mapping
with open(original_mapping_path, 'r') as f:
    mapping = json.load(f)

# List the files in the new masks folder
mask_files = os.listdir(new_masks_folder)

new_mapping = {}

# Print all filenames in the mask folder for debugging purposes
print("Files in the new masks folder:")
for mask_file in mask_files:
    print(mask_file)

# Process only BA "Ba1" and day "Dy30" entries
for key, info in mapping.items():
    if info.get('dayID') == 'Dy30' and info.get('BA') == 'Ba1':
        # Use the wellID from the mapping, e.g., "A4"
        well = info.get('wellID')
        dayID = info.get('dayID')  # 'Dy30'
        BA = info.get('BA')  # 'Ba1'

        # Create a relaxed pattern to match the wellID, dayID, and BA in the mask file name
        # Adjust the regex to be more lenient, considering possible variations in naming conventions
        pattern = r'Mask_M.*' + re.escape(BA) + r'.*' + re.escape(dayID) + r'.*' + re.escape(well)

        # Debugging: print the pattern being used and the mask files
        print(f"\nMatching for key: {key} with wellID: {well}, dayID: {dayID}, BA: {BA}")
        print(f"Pattern: {pattern}")
        
        # Look for mask files that match this pattern
        matching_files = [mf for mf in mask_files if re.search(pattern, mf)]
        
        # Debugging: print the matching files
        if matching_files:
            print(f"Found matching mask files: {matching_files}")
        
        if matching_files:
            # If there are multiple matches, you might decide to choose one
            mask_file = matching_files[0]
            # Add a new attribute "Mask Path" with the full path of the mask file
            info['Mask Path'] = os.path.join(new_masks_folder, mask_file)
            new_mapping[key] = info
        else:
            print(f"No matching mask found for key: {key} with wellID: {well} and dayID: {dayID}")

# Save the new mapping JSON
with open(new_mapping_path, 'w') as f:
    json.dump(new_mapping, f, indent=4)

print(f"\nNew mapping created with {len(new_mapping)} entries and saved to {new_mapping_path}")
