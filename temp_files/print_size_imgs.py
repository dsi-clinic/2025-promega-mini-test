import os
import json
from pathlib import Path
import cv2  # Need cv2 to load images and get shapes


MASKS_FOLDER = "/net/projects2/promega/data-analysis/masks"
JSON_MAPPING_PATH = "/net/projects2/promega/data-analysis/output/image_mapping_day30_manual.json"
# --- Load the JSON mapping file ---
try:
    with open(JSON_MAPPING_PATH, 'r') as f:
        image_mapping = json.load(f)
    print(f"Successfully loaded JSON mapping from: {JSON_MAPPING_PATH}")
except FileNotFoundError:
    print(f"Error: JSON mapping file not found at {JSON_MAPPING_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {JSON_MAPPING_PATH}. Check file format.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading the JSON: {e}")
    exit()

# --- Filter for Dy30 ---
filtered_mapping = {k: v for k, v in image_mapping.items() if v.get('dayID') == 'Dy30'}
print(f"Found {len(filtered_mapping)} entries for dayID='Dy30'")
print("-" * 30) # Separator

# --- Process each entry to print sizes ---
processed_count = 0
skipped_count = 0
for img_id, img_info in filtered_mapping.items():
    print(f"Processing ID: {img_id}")

    # --- Get image path ---
    img_path_str = img_info.get('Best Z Filename')
    if not img_path_str:
        print(f"  -> Skipped: Missing 'Best Z Filename' key in JSON.")
        skipped_count += 1
        continue
    img_path = Path(img_path_str) # Use Path object

    # --- Get or construct mask path ---
    mask_path_str = img_info.get('Mask Path') # Check if 'Mask Path' exists first
    if not mask_path_str:
        # If not, construct it (adjust filename pattern if needed)
        img_base_name = img_path.stem # Gets filename without extension
        mask_filename = f"{img_base_name}_cellpose_mask.png" # Example pattern
        # mask_filename = f"Mask_{img_base_name}.tif" # Alternative pattern
        mask_path = MASKS_FOLDER / mask_filename
        print(f"  Constructed mask path: {mask_path}")
    else:
        mask_path = Path(mask_path_str) # Use Path object
        print(f"  Using mask path from JSON: {mask_path}")

    # --- Validate file existence ---
    if not img_path.exists():
        print(f"  -> Skipped: Image file not found at {img_path}")
        skipped_count += 1
        continue

    if not mask_path.exists():
        print(f"  -> Skipped: Mask file not found at {mask_path}")
        skipped_count += 1
        continue

    # --- Load image and mask to get dimensions ---
    try:
        # Load image
        img = cv2.imread(str(img_path)) # cv2 needs string path
        if img is None:
            print(f"  -> Skipped: OpenCV could not load image (is it corrupted or wrong format?) at {img_path}")
            skipped_count += 1
            continue
        img_shape = img.shape

        # Load mask in grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) # cv2 needs string path
        if mask is None:
            print(f"  -> Skipped: OpenCV could not load mask (is it corrupted or wrong format?) at {mask_path}")
            skipped_count += 1
            continue
        mask_shape = mask.shape

        # --- Print the shapes ---
        print(f"  Image shape: {img_shape}, Mask shape: {mask_shape}")
        processed_count += 1

    except Exception as e:
        print(f"  -> Skipped: Error during image/mask loading for ID {img_id}: {e}")
        skipped_count += 1
        continue # Skip to next iteration on error

print("-" * 30) # Separator
print(f"Finished.")
print(f"Successfully checked dimensions for: {processed_count} entries.")
print(f"Skipped due to missing files or errors: {skipped_count} entries.")