import os
import json
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # or adjust as needed
from dotenv import load_dotenv
from paths import ORIGINAL_MAPPING, MANUAL_MAPPING_OUTPUT_DIR, MANUAL_MASK_FOLDERS

# === Setup ===
load_dotenv()

# Output name
OUTPUT_NAME = "image_mapping_thresholded_and_manual.json"
new_mapping_path = MANUAL_MAPPING_OUTPUT_DIR / OUTPUT_NAME

# === Load base mapping ===
with open(ORIGINAL_MAPPING, 'r') as f:
    mapping = json.load(f)

# === List all mask files (manual and thresholded) ===
mask_files = []
for folder in MANUAL_MASK_FOLDERS:
    folder = Path(folder)
    if not folder.exists():
        print(f"Warning: mask folder does not exist: {folder}")
        continue
    mask_files.extend([(folder, f.name) for f in folder.iterdir() if f.is_file()])

print(f"\nFound {len(mask_files)} mask files across {len(MANUAL_MASK_FOLDERS)} folders.\n")

# === Match entries ===
new_mapping = {}

for key, info in mapping.items():
    ba = info.get('BA')
    day = info.get('dayID')
    well = info.get('wellID')

    if not (ba and day and well):
        continue

    # Clean keys
    # Clean the mapping info strings
    ba_clean = ba.replace(" ", "").replace("_", "").lower()
    day_clean = day.strip().lower()
    well_clean = well.strip().lower()

    matches = []
    for folder, mf in mask_files:
        mf_clean = mf.replace(" ", "").replace("_", "").lower()
        if ba_clean in mf_clean and day_clean in mf_clean and well_clean in mf_clean:
            matches.append((folder, mf))



    print(f"\nProcessing: {key} | {day}, {ba}, {well}")
    print(f"  Clean match terms: BA={ba_clean}, Day={day_clean}, Well={well_clean}")
    print(f"  Matches found: {len(matches)}")

    if matches:
        mask_folder, mask_file = matches[0]  # first match
        info['Mask Path'] = str(mask_folder / mask_file)
        new_mapping[key] = {
            "dayID": info.get("dayID"),
            "BA": info.get("BA"),
            "wellID": info.get("wellID"),
            "Best Z Filename": info.get("Best Z Filename"),
            "MT Mask Path": str(mask_folder / mask_file)
        }
        print(f"  -> Using mask: {mask_folder / mask_file}")

    else:
        print(f"  -> No matching mask found.")

# === Save new mapping ===
new_mapping_path.parent.mkdir(parents=True, exist_ok=True)
with open(new_mapping_path, 'w') as f:
    json.dump(new_mapping, f, indent=4)

print(f"\nSaved {len(new_mapping)} entries to: {new_mapping_path}")
