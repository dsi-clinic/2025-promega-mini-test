# create_batch_mapping.py
import json
from pathlib import Path
import cv2
import argparse

# Configuration
TARGET_SIZE = (256, 192)  # width, height
INTERPOLATION = cv2.INTER_LINEAR

# Paths
ORIGINAL_MAPPING = Path("/net/projects2/promega/data-analysis/output/image_mapping.json")
OUTPUT_DIR = Path("/net/projects2/promega/data-analysis/output/processed_dataset_256x192")

def process_batch(batch_num, day_num=30):
    """Create mapping for a batch with preprocessed images"""
    with open(ORIGINAL_MAPPING, 'r') as f:
        mapping = json.load(f)
    
    # Special handling for Batch2
    if batch_num == 2:
        create_mapping(mapping, "BA2 96_1", day_num)
        create_mapping(mapping, "BA2 96_2", day_num)
    else:
        create_mapping(mapping, f"BA{batch_num}", day_num)

def create_mapping(mapping, batch_id, day_num):
    """Create mapping for specific batch ID"""
    day_id = f"Dy{day_num}"
    output_dir = OUTPUT_DIR / f"{batch_id.replace(' ', '_')}_{day_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = output_dir / f"image_mapping_{batch_id.replace(' ', '_')}_{day_id}_processed.json"
    
    if output_json.exists():
        print(f"Mapping exists: {output_json}")
        return

    new_mapping = {}
    
    for img_id, img_info in mapping.items():
        if img_info.get('dayID') == day_id and img_info.get('BA') == batch_id:
            img_path = Path(img_info.get('Best Z Filename'))
            if not img_path.exists():
                print(f"Skipped: Image not found {img_path}")
                continue
            
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Skipped: Could not read image {img_path}")
                    continue
                
                resized = cv2.resize(img, TARGET_SIZE, interpolation=INTERPOLATION)
                output_path = output_dir / f"{img_id.replace(' ', '_')}.png"
                cv2.imwrite(str(output_path), resized)
                
                # Simplified mapping - just store the image path
                new_mapping[img_id] = {
                    'img_path': str(output_path)
                }
                
            except Exception as e:
                print(f"Error processing {img_id}: {e}")
                continue

    with open(output_json, 'w') as f:
        json.dump(new_mapping, f, indent=2)
    
    print(f"\nCreated mapping for {batch_id} with {len(new_mapping)} images")
    print(f"Images saved to: {output_dir}")
    print(f"Mapping saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True, help='Batch number to process')
    parser.add_argument('--day', type=int, default=30, help='Day number (default: 30)')
    args = parser.parse_args()

    print(f"\nCreating mapping for Batch {args.batch}, Day {args.day}")
    process_batch(args.batch, args.day)
    print("\nDone.")