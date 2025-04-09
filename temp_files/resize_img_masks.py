import os
import json
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
import shutil # Used for potentially removing old directories

# --- Configuration ---
# SET YOUR DESIRED OUTPUT DIMENSIONS HERE
TARGET_WIDTH = 256
TARGET_HEIGHT = 192

TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT) # (Width, Height) for OpenCV resize
# Interpolation methods for resizing
IMAGE_INTERPOLATION = cv2.INTER_LINEAR # Common choice for downscaling images
MASK_INTERPOLATION = cv2.INTER_NEAREST  # MUST use nearest for masks

# --- Load Environment Variables ---
load_dotenv()
try:
    # Need original mask folder mainly if mask path isn't directly in JSON
    MASKS_FOLDER = Path("/net/projects2/promega/data-analysis/masks")
    JSON_MAPPING_PATH = Path("/net/projects2/promega/data-analysis/output/image_mapping_day30_manual.json")
    # Define where the NEW processed dataset will be saved
    # (e.g., in the same directory as the original JSON)
    OUTPUT_BASE_FOLDER = JSON_MAPPING_PATH.parent / f"processed_dataset_{TARGET_WIDTH}x{TARGET_HEIGHT}"
except KeyError as e:
    print(f"Error: Environment variable {e} not set. Make sure .env file exists.")
    exit()
except Exception as e:
    print(f"Error setting up paths: {e}")
    exit()

# --- Define Output Paths ---
PROCESSED_IMAGES_FOLDER = OUTPUT_BASE_FOLDER / "images"
PROCESSED_MASKS_FOLDER = OUTPUT_BASE_FOLDER / "masks"
NEW_JSON_MAPPING_FILENAME = f"{JSON_MAPPING_PATH.stem}_processed_{TARGET_WIDTH}x{TARGET_HEIGHT}.json"
NEW_JSON_MAPPING_PATH = OUTPUT_BASE_FOLDER / NEW_JSON_MAPPING_FILENAME

# --- Create Output Directories ---
# Optional: Uncomment to remove existing processed data before starting
# if OUTPUT_BASE_FOLDER.exists():
#     print(f"WARNING: Output folder {OUTPUT_BASE_FOLDER} already exists. Removing it...")
#     shutil.rmtree(OUTPUT_BASE_FOLDER)
#     print("Old output folder removed.")

PROCESSED_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_MASKS_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"Output folders created/ensured:")
print(f"  Base:      {OUTPUT_BASE_FOLDER}")
print(f"  Images:    {PROCESSED_IMAGES_FOLDER}")
print(f"  Masks:     {PROCESSED_MASKS_FOLDER}")
print(f"  New JSON:  {NEW_JSON_MAPPING_PATH}")


# --- Load Original JSON Mapping ---
try:
    with open(JSON_MAPPING_PATH, 'r') as f:
        original_image_mapping = json.load(f)
    print(f"\nLoaded original JSON mapping: {JSON_MAPPING_PATH}")
except Exception as e:
    print(f"Error loading original JSON {JSON_MAPPING_PATH}: {e}")
    exit()

# --- Filter for Dy30 ---
filtered_mapping = {k: v for k, v in original_image_mapping.items() if v.get('dayID') == 'Dy30'}
print(f"Found {len(filtered_mapping)} entries for dayID='Dy30' to process.")
print("-" * 30)

# --- Process images and masks ---
new_mapping_data = {} # Store data for the new JSON
processed_count = 0
skipped_count = 0

for img_id, img_info in filtered_mapping.items():
    print(f"Processing ID: {img_id}")

    # --- Get original paths ---
    img_path_str = img_info.get('Best Z Filename')
    mask_path_str = img_info.get('Mask Path') # Check if 'Mask Path' exists first

    # --- Construct mask path if not present ---
    if not mask_path_str:
        if not img_path_str:
             print(f"  -> Skipped: Cannot construct mask path without 'Best Z Filename'.")
             skipped_count += 1
             continue
        img_base_name = Path(img_path_str).stem
        # *** ADJUST MASK FILENAME PATTERN IF NEEDED ***
        mask_filename = f"{img_base_name}_cellpose_mask.png"
        mask_path = MASKS_FOLDER / mask_filename
    else:
        mask_path = Path(mask_path_str)

    if not img_path_str:
        print(f"  -> Skipped: Missing 'Best Z Filename'.")
        skipped_count += 1
        continue
    img_path = Path(img_path_str)

    # --- Validate existence of originals ---
    if not img_path.exists():
        print(f"  -> Skipped: Original image not found: {img_path}")
        skipped_count += 1
        continue
    if not mask_path.exists():
        print(f"  -> Skipped: Original mask not found: {mask_path}")
        skipped_count += 1
        continue

    try:
        # --- Load original image and mask ---
        original_img = cv2.imread(str(img_path))
        original_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if original_img is None:
            print(f"  -> Skipped: Failed to load original image: {img_path}")
            skipped_count += 1
            continue
        if original_mask is None:
            print(f"  -> Skipped: Failed to load original mask: {mask_path}")
            skipped_count += 1
            continue

        # --- Resize image ---
        resized_img = cv2.resize(original_img, TARGET_SIZE, interpolation=IMAGE_INTERPOLATION)

        # --- Resize mask ---
        resized_mask = cv2.resize(original_mask, TARGET_SIZE, interpolation=MASK_INTERPOLATION)

        # --- Binarize mask (ensure only 0 and 255) ---
        # Check unique values before binarizing (optional debug)
        # print(f"    Unique values in resized mask: {np.unique(resized_mask)}")
        binary_mask = (resized_mask > 0).astype(np.uint8)
        # print(f"    Unique values in binarized mask: {np.unique(binary_mask)}")

        # --- Define output filenames ---
        img_suffix = img_path.suffix if img_path.suffix else '.png' # Keep original suffix or default
        output_img_filename = f"{img_id}_{TARGET_WIDTH}x{TARGET_HEIGHT}{img_suffix}"
        # Always save masks as PNG for lossless quality of 0/255 values
        output_mask_filename = f"{img_id}_mask_{TARGET_WIDTH}x{TARGET_HEIGHT}.png"

        output_img_path = PROCESSED_IMAGES_FOLDER / output_img_filename
        output_mask_path = PROCESSED_MASKS_FOLDER / output_mask_filename

        # --- Save processed files ---
        save_img_success = cv2.imwrite(str(output_img_path), resized_img)
        save_mask_success = cv2.imwrite(str(output_mask_path), binary_mask)

        if not save_img_success:
             print(f"  -> Failed: Could not save processed image to {output_img_path}")
             # Decide if you want to skip or continue if saving fails
             skipped_count += 1
             continue # Skip this entry if image saving failed
        if not save_mask_success:
             print(f"  -> Failed: Could not save processed mask to {output_mask_path}")
             # Decide if you want to skip or continue if saving fails
             skipped_count += 1
             # Clean up already saved image for this entry? Optional.
             # if output_img_path.exists(): output_img_path.unlink()
             continue # Skip this entry if mask saving failed

        # --- Add entry to the new mapping data ---
        # Store absolute paths in the new JSON for robustness
        new_mapping_data[img_id] = {
            'img_path': str(output_img_path.resolve()), # Points to the NEW processed image
            'seg_map_path': str(output_mask_path.resolve()), # Points to the NEW processed mask
            # Keep other relevant info
            'img_id': img_id,
            'dayID': img_info.get('dayID', 'Dy30'),
            'BA': img_info.get('BA'),
            'wellID': img_info.get('wellID'),
            'original_img_path': str(img_path.resolve()), # Keep original path for reference
        }
        processed_count += 1
        # print(f"  Successfully processed and saved.") # Uncomment for verbose output

    except Exception as e:
        print(f"  -> Skipped: UNEXPECTED ERROR during processing for ID {img_id}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for unexpected errors
        skipped_count += 1
        continue

# --- Save the new JSON mapping ---
if not new_mapping_data:
    print("-" * 30)
    print("No entries were successfully processed. New JSON mapping file will not be created.")
else:
    try:
        with open(NEW_JSON_MAPPING_PATH, 'w') as f:
            json.dump(new_mapping_data, f, indent=4) # Use indent for readability
        print("-" * 30)
        print(f"Successfully saved new JSON mapping ({len(new_mapping_data)} entries) to:")
        print(f"  {NEW_JSON_MAPPING_PATH.resolve()}")
    except Exception as e:
        print("-" * 30)
        print(f"Error saving new JSON mapping: {e}")

# --- Final Summary ---
print("-" * 30)
print(f"Preprocessing Summary:")
print(f"  Target Dimensions: {TARGET_WIDTH}x{TARGET_HEIGHT}")
print(f"  Processed Entries: {processed_count}")
print(f"  Skipped Entries:   {skipped_count}")
print(f"  Processed images saved to: {PROCESSED_IMAGES_FOLDER.resolve()}")
print(f"  Processed masks saved to:  {PROCESSED_MASKS_FOLDER.resolve()}")