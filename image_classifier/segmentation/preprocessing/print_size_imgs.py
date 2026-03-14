import json
from pathlib import Path
import cv2
from config import MASKS_FOLDER, DAY30_MAPPING as JSON_MAPPING_PATH


def main():
    try:
        with open(JSON_MAPPING_PATH, "r") as f:
            image_mapping = json.load(f)
        print(f"Successfully loaded JSON mapping from: {JSON_MAPPING_PATH}")
    except FileNotFoundError:
        print(f"Error: JSON mapping file not found at {JSON_MAPPING_PATH}")
        return
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {JSON_MAPPING_PATH}. Check file format."
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred loading the JSON: {e}")
        return

    filtered_mapping = {
        k: v for k, v in image_mapping.items() if v.get("dayID") == "Dy30"
    }
    print(f"Found {len(filtered_mapping)} entries for dayID='Dy30'")
    print("-" * 30)

    processed_count = 0
    skipped_count = 0
    for img_id, img_info in filtered_mapping.items():
        print(f"Processing ID: {img_id}")

        img_path_str = img_info.get("Best Z Filename")
        if not img_path_str:
            print("  -> Skipped: Missing 'Best Z Filename' key in JSON.")
            skipped_count += 1
            continue
        img_path = Path(img_path_str)

        mask_path_str = img_info.get("Mask Path")
        if not mask_path_str:
            img_base_name = img_path.stem
            mask_filename = f"{img_base_name}_cellpose_mask.png"
            mask_path = MASKS_FOLDER / mask_filename
            print(f"  Constructed mask path: {mask_path}")
        else:
            mask_path = Path(mask_path_str)
            print(f"  Using mask path from JSON: {mask_path}")

        if not img_path.exists():
            print(f"  -> Skipped: Image file not found at {img_path}")
            skipped_count += 1
            continue

        if not mask_path.exists():
            print(f"  -> Skipped: Mask file not found at {mask_path}")
            skipped_count += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  -> Skipped: OpenCV could not load image at {img_path}")
                skipped_count += 1
                continue
            img_shape = img.shape

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  -> Skipped: OpenCV could not load mask at {mask_path}")
                skipped_count += 1
                continue
            mask_shape = mask.shape

            print(f"  Image shape: {img_shape}, Mask shape: {mask_shape}")
            processed_count += 1

        except Exception as e:
            print(f"  -> Skipped: Error during image/mask loading for ID {img_id}: {e}")
            skipped_count += 1
            continue

    print("-" * 30)
    print("Finished.")
    print(f"Successfully checked dimensions for: {processed_count} entries.")
    print(f"Skipped due to missing files or errors: {skipped_count} entries.")


if __name__ == "__main__":
    main()
