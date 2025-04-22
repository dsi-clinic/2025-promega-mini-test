# predict_new_batch.py
import torch
from pathlib import Path
import cv2
import numpy as np
import warnings
import json
import random
import argparse

# --- Suppress warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Import MMSegmentation ---
try:
    from mmseg.apis import init_model, inference_model
    from mmengine.model.utils import revert_sync_batchnorm
    print("MMSegmentation API imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    exit()

# ======== USER CONFIG ========
# Model paths
CONFIG_FILE_PATH = '/net/projects2/promega/data-analysis/plots/segformer_masks/20250413_233923/vis_data/config.py'
CHECKPOINT_FILE_PATH = '/net/projects2/promega/data-analysis/plots/segformer_masks/iter_1000.pth'

# Base directory structure
PREPROCESSED_JSON_DIR = '/net/projects2/promega/data-analysis/output/processed_dataset_256x192'
OUTPUT_MASKS_BASE_DIR = '/net/projects2/promega/data-analysis/predictions'

# Collage samples
NUM_SAMPLES_FOR_COLLAGE = 10
# ============================

def get_mapping_paths(batch_number, day_number=30):
    """Get paths to preprocessed JSON mappings, handling BA2 special case."""
    if batch_number == 2:
        # BA2 has two parts - return both paths
        return [
            Path(PREPROCESSED_JSON_DIR) / f"BA2_96_1_Dy{day_number}" / f"image_mapping_BA2_96_1_Dy{day_number}_processed.json",
            Path(PREPROCESSED_JSON_DIR) / f"BA2_96_2_Dy{day_number}" / f"image_mapping_BA2_96_2_Dy{day_number}_processed.json"
        ]
    else:
        # Normal case for other batches
        return [Path(PREPROCESSED_JSON_DIR) / f"Ba{batch_number}_Dy{day_number}" / f"image_mapping_Ba{batch_number}_Dy{day_number}_processed.json"]

def run_inference(batch_number, day_number=30):
    """Run inference on specified batch."""
    mapping_paths = get_mapping_paths(batch_number, day_number)
    
    total_processed = 0
    total_failed = 0
    
    for json_mapping_path in mapping_paths:
        if not json_mapping_path.exists():
            print(f"Warning: Preprocessed JSON not found at {json_mapping_path}")
            continue

        # Determine batch part for output directory
        if batch_number == 2:
            part = "96_1" if "96_1" in str(json_mapping_path) else "96_2"
            output_dir = Path(OUTPUT_MASKS_BASE_DIR) / f"batch{batch_number}_{part}" / f"day{day_number}"
        else:
            output_dir = Path(OUTPUT_MASKS_BASE_DIR) / f"batch{batch_number}" / f"day{day_number}"
            
        masks_dir = output_dir / "predicted_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        collage_path = output_dir / f"inference_collage_batch{batch_number}_{f'part{part}' if batch_number == 2 else ''}_day{day_number}.png"

        # Load mapping
        with open(json_mapping_path, 'r') as f:
            batch_mapping = json.load(f)
        print(f"\nLoaded {len(batch_mapping)} entries from {json_mapping_path.name}")

        # Initialize model (only once)
        if 'model' not in locals():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model = init_model(str(CONFIG_FILE_PATH), str(CHECKPOINT_FILE_PATH), device=device)
            if device == 'cpu':
                model = revert_sync_batchnorm(model)
            print(f"Model loaded on {device}")

        # Select random samples for collage
        sample_ids = random.sample(list(batch_mapping.keys()), 
                                 min(NUM_SAMPLES_FOR_COLLAGE, len(batch_mapping)))
        
        # Process all images
        collage_pairs = []
        processed = failed = 0
        img_h = img_w = None

        print(f"Processing {len(batch_mapping)} images...")
        for img_id, img_info in batch_mapping.items():
            img_path = Path(img_info['img_path'])
            if not img_path.exists():
                print(f"  Skipped: Image not found {img_path}")
                failed += 1
                continue

            try:
                # Inference
                result = inference_model(model, str(img_path))
                pred_mask = (result.pred_sem_seg.data.squeeze().cpu().numpy() * 255).astype(np.uint8)
                
                # Save mask
                mask_path = masks_dir / f"{img_path.stem}_predmask.png"
                cv2.imwrite(str(mask_path), pred_mask)
                processed += 1

                # Prepare collage if selected
                if img_id in sample_ids:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    # Store/resize dimensions
                    if img_h is None:
                        img_h, img_w = img.shape[:2]
                    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    mask_vis = cv2.resize(pred_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                    
                    collage_pairs.append(np.hstack((img, mask_vis)))

            except Exception as e:
                print(f"  Error processing {img_id}: {str(e)}")
                failed += 1

        # Save collage if we have samples
        if collage_pairs:
            cv2.imwrite(str(collage_path), np.vstack(collage_pairs))
            print(f"\nCollage saved to {collage_path}")

        print(f"Finished part. Processed: {processed}, Failed: {failed}")
        total_processed += processed
        total_failed += failed

    print(f"\nTotal processed: {total_processed}, Total failed: {total_failed}")
    return total_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True, help='Batch number to process (2, 3, etc.)')
    parser.add_argument('--day', type=int, default=30, help='Day number (default: 30)')
    args = parser.parse_args()

    print(f"\n{'='*40}")
    print(f"Processing Batch {args.batch}, Day {args.day}")
    print(f"{'='*40}")

    run_inference(args.batch, args.day)
    print("\nDone.")