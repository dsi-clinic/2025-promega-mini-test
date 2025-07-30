# predict_new_batch.py
import torch
from pathlib import Path
import cv2
import numpy as np
import warnings
import json
import random
import argparse
from paths import EARLY_MODEL, LATE_MODEL
from paths import PROCESSED_DATA_DIR, OUTPUT_MASKS_BASE_DIR

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

# Base directory structure
OUTPUT_MASKS_BASE_DIR = '/net/projects2/promega/data-analysis/predictions'

# Collage samples
NUM_SAMPLES_FOR_COLLAGE = 10
# ============================

def get_mapping_paths(batch_number, day_number=30):
    """Get zero-padded lowercase mapping JSON paths, including 96_1 and 96_2 for batch 2."""
    day_str = f"{day_number:02d}"
    paths = []

    if batch_number == 2:
        for part in ["96_1", "96_2"]:
            batch_str = f"ba{batch_number}{part}_Dy{day_str}"
            path = Path(PROCESSED_DATA_DIR) / batch_str / f"image_mapping_{batch_str}_processed.json"
            paths.append(path)
    else:
        batch_str = f"ba{batch_number}96_1_Dy{day_str}"
        path = Path(PROCESSED_DATA_DIR) / batch_str / f"image_mapping_{batch_str}_processed.json"
        paths.append(path)

    return paths


def run_inference(batch_number, day_number=30, model_type="early", overwrite=False):
    """Run inference on specified batch/day."""
    day_str = f"{day_number:02d}"
    mapping_paths = get_mapping_paths(batch_number, day_number)

    total_processed = total_failed = 0

    for json_mapping_path in mapping_paths:
        print(f"Checking for file: {json_mapping_path}")
        print(f"Absolute path exists? {json_mapping_path.resolve()} -> {json_mapping_path.exists()}")

        if not json_mapping_path.exists():
            print(f"Warning: Preprocessed JSON not found at {json_mapping_path}")
            continue

        # Determine output directory, also zero-padded
        if batch_number == 2:
            part = "96_1" if "96_1" in str(json_mapping_path) else "96_2"
            output_dir = Path(OUTPUT_MASKS_BASE_DIR) / f"batch{batch_number}_{part}" / f"day{day_str}"
        else:
            output_dir = Path(OUTPUT_MASKS_BASE_DIR) / f"batch{batch_number}" / f"day{day_str}"
        masks_dir = output_dir / "predicted_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        collage_path = output_dir / f"inference_collage_batch{batch_number}_" \
                                    f"{('part'+part) if batch_number==2 else ''}_" \
                                    f"day{day_str}.png"

        # Load mapping JSON
        with open(json_mapping_path, 'r') as f:
            batch_mapping = json.load(f)
        print(f"\nLoaded {len(batch_mapping)} entries from {json_mapping_path.name}")

        if model_type == 'early':
            model_info = EARLY_MODEL
        else:
            model_info = LATE_MODEL

        CONFIG_FILE_PATH = model_info["config"]
        CHECKPOINT_FILE_PATH = model_info["checkpoint"]

                # Init model once
        if 'model' not in locals():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model = init_model(str(CONFIG_FILE_PATH), str(CHECKPOINT_FILE_PATH), device=device)
            if device == 'cpu':
                model = revert_sync_batchnorm(model)
            print(f"Model loaded on {device}")

        sample_ids = random.sample(list(batch_mapping), 
                                   min(NUM_SAMPLES_FOR_COLLAGE, len(batch_mapping)))

        processed = failed = 0
        collage_pairs = []
        img_h = img_w = None

        for img_id, img_info in batch_mapping.items():
            img_path = Path(img_info['img_path'])
            if not img_path.exists():
                failed += 1
                continue

            try:
                result = inference_model(model, str(img_path))
                pred_mask = (result.pred_sem_seg.data.squeeze().cpu().numpy() * 255).astype(np.uint8)

                mask_path = masks_dir / f"{img_path.stem}_predmask.png"

                if not overwrite and mask_path.exists():
                    processed += 1
                    batch_mapping[img_id]['mask_path'] = str(mask_path)
                    continue

                cv2.imwrite(str(mask_path), pred_mask)
                processed += 1
                batch_mapping[img_id]['mask_path'] = str(mask_path)


                # build collage sample
                if img_id in sample_ids:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    if img_h is None:
                        img_h, img_w = img.shape[:2]
                    img_resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    mask_vis   = cv2.resize(pred_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    mask_vis   = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                    collage_pairs.append(np.hstack((img_resized, mask_vis)))

            except Exception as e:
                failed += 1

        if collage_pairs:
            cv2.imwrite(str(collage_path), np.vstack(collage_pairs))
            print(f"Collage saved to {collage_path}")

        print(f"Finished part. Processed: {processed}, Failed: {failed}")
        total_processed += processed
        total_failed   += failed

        # write back augmented JSON
        with open(json_mapping_path, 'w') as f:
            json.dump(batch_mapping, f, indent=2)
        print(f"Updated mapping with mask paths: {json_mapping_path}")

    print(f"\nTotal processed: {total_processed}, Total failed: {total_failed}")
    return total_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
    '--model_type',
    choices=['early', 'late'],
    required=True,
    help="Choose which model to use: 'early' or 'late'"
    )

    parser.add_argument(
        '--batches',
        type=lambda s: [int(x) for x in s.split(',')],
        required=True,
        help='Comma-separated batch numbers, e.g. 1,2,3'
    )
    parser.add_argument(
        '--days',
        type=lambda s: [int(x) for x in s.split(',')],
        required=True,
        help='Comma-separated day numbers, e.g. 3,6,8'
    )
    parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Force reprocessing even if mask already exists'
    )
    args = parser.parse_args()


    for batch in args.batches:
        for day in args.days:
            print(f"\n{'='*40}")
            print(f"Processing Batch {batch}, Day {day}")
            print(f"{'='*40}")
            run_inference(batch, day, model_type=args.model_type, overwrite=args.overwrite)


    print("\nAll done.")