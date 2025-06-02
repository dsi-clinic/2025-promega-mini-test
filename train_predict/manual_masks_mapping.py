import os
import json
import re

# File paths
original_mapping_path = '/net/projects2/promega/data-analysis/output/image_mapping.json'
new_masks_folders = [
    '/net/projects2/promega/data-analysis/manual_masks/Manuais',
    '/net/projects2/promega/data-analysis/manual_masks/Treshold'
]
new_mapping_path = '/net/projects2/promega/data-analysis/output/image_mapping_day10_manual.json'

# Load the original mapping
with open(original_mapping_path, 'r') as f:
    mapping = json.load(f)

# List the files in all new masks folders
mask_files = []
for folder in new_masks_folders:
    folder_files = os.listdir(folder)
    mask_files.extend([(folder, f) for f in folder_files])

# Print all filenames in the mask folders for debugging purposes
print("Files in the new masks folders:")
for folder, mask_file in mask_files:
    print(f"{folder}: {mask_file}")

new_mapping = {}

# Process only BA "Ba1" and day "Dy24" entries
for key, info in mapping.items():
    if info.get('dayID') == 'Dy10' and info.get('BA') == 'Ba1':  # Changed Dy30 to Dy24 here
        # Use the wellID from the mapping, e.g., "A4"
        well = info.get('wellID')
        dayID = info.get('dayID')  # 'Dy24'
        BA = info.get('BA')  # 'Ba1'

        # Create patterns to match both Mask_M and Mask_T prefixes
        pattern_manuais = r'Mask_M.*' + re.escape(BA) + r'.*' + re.escape(dayID) + r'.*' + re.escape(well)
        pattern_treshold = r'Mask_T.*' + re.escape(BA) + r'.*' + re.escape(dayID) + r'.*' + re.escape(well)
        
        # Combine patterns with OR condition
        combined_pattern = f'({pattern_manuais}|{pattern_treshold})'

        # Debugging: print the pattern being used
        print(f"\nMatching for key: {key} with wellID: {well}, dayID: {dayID}, BA: {BA}")
        print(f"Combined pattern: {combined_pattern}")
        
        # Look for mask files that match either pattern in any folder
        matching_files = [(folder, mf) for folder, mf in mask_files if re.search(combined_pattern, mf)]
        
        # Debugging: print the matching files
        if matching_files:
            print(f"Found matching mask files: {matching_files}")
        
        if matching_files:
            # If there are multiple matches, choose the first one
            mask_folder, mask_file = matching_files[0]
            # Add a new attribute "Mask Path" with the full path of the mask file
            info['Mask Path'] = os.path.join(mask_folder, mask_file)
            new_mapping[key] = info
        else:
            print(f"No matching mask found for key: {key} with wellID: {well} and dayID: {dayID}")

# Save the new mapping JSON
with open(new_mapping_path, 'w') as f:
    json.dump(new_mapping, f, indent=4)

print(f"\nNew mapping created with {len(new_mapping)} entries and saved to {new_mapping_path}")