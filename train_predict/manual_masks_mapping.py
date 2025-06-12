import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from paths import ORIGINAL_MAPPING, MANUAL_MASK_FOLDERS, MANUAL_MAPPING_OUTPUT_DIR

# === USER VARIABLES ===
TARGET_DAY = 'Dy10'
TARGET_BA = 'Ba1'
OUTPUT_NAME = f"image_mapping_{TARGET_DAY.lower()}_{TARGET_BA.lower()}_manual.json"

# === Setup ===
load_dotenv()

# Paths
original_mapping_path = ORIGINAL_MAPPING
new_masks_folders = MANUAL_MASK_FOLDERS
new_mapping_path = MANUAL_MAPPING_OUTPUT_DIR / OUTPUT_NAME

# === Load base mapping ===
with open(original_mapping_path, 'r') as f:
    mapping = json.load(f)

# === List available masks ===
mask_files = []
for folder in new_masks_folders:
    folder = Path(folder)
    if not folder.exists():
        print(f"Warning: mask folder does not exist: {folder}")
        continue
    mask_files.extend([(folder, f.name) for f in folder.iterdir() if f.is_file()])

print(f"\nFound {len(mask_files)} mask files across {len(new_masks_folders)} folders.\n")

new_mapping = {}

# === Filter and match entries ===
for key, info in mapping.items():
    if info.get('dayID') == TARGET_DAY and info.get('BA') == TARGET_BA:
        well = info.get('wellID')
        if not well:
            continue

        pattern_m = fr"Mask_M.*{re.escape(TARGET_BA)}.*{re.escape(TARGET_DAY)}.*{re.escape(well)}"
        pattern_t = fr"Mask_T.*{re.escape(TARGET_BA)}.*{re.escape(TARGET_DAY)}.*{re.escape(well)}"
        combined_pattern = f"({pattern_m}|{pattern_t})"

        matches = [(folder, mf) for folder, mf in mask_files if re.search(combined_pattern, mf)]

        print(f"\nProcessing: {key} | {TARGET_DAY}, {TARGET_BA}, {well}")
        print(f"  Regex: {combined_pattern}")
        print(f"  Matches found: {len(matches)}")

        if matches:
            mask_folder, mask_file = matches[0]  # use first match
            info['Mask Path'] = str(mask_folder / mask_file)
            new_mapping[key] = info
        else:
            print(f"  -> No matching mask found.")

# === Save new mapping ===
new_mapping_path.parent.mkdir(parents=True, exist_ok=True)
with open(new_mapping_path, 'w') as f:
    json.dump(new_mapping, f, indent=4)

print(f"\n Saved {len(new_mapping)} entries to: {new_mapping_path}")
