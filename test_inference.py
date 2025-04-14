# multi_inference_collage_v2.py
import torch
from pathlib import Path
import cv2
import numpy as np
import warnings
import os
import json
import random

# --- Suppress some common warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Try to import MMSegmentation ---
try:
    from mmseg.apis import init_model, inference_model
    from mmengine.model.utils import revert_sync_batchnorm
    print("MMSegmentation API imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure you have mmsegmentation and its dependencies installed")
    print("and that you are running this script from the correct Python environment.")
    exit()

# ===========================================
# === USER: SET THESE PATHS AND PARAMETERS ===
# ===========================================

# 1. Path to the configuration file from your work directory
config_file_path = '/net/projects2/promega/data-analysis/plots/segformer_masks/20250413_233923/vis_data/config.py'

# 2. Path to the checkpoint (.pth) file you want to use
checkpoint_file_path = '/net/projects2/promega/data-analysis/plots/segformer_masks/iter_1000.pth'

# 3. Path to the PROCESSED JSON mapping file (pointing to pre-resized images/masks)
#    This JSON should have been created by the preprocessing script.
processed_json_mapping_path = '/net/projects2/promega/data-analysis/output/processed_dataset_256x192/image_mapping_day30_manual_processed_256x192.json'

# 4. Number of random samples to visualize
num_samples = 10

# 5. Path where you want to save the output collage image
output_collage_save_path = './inference_collage_pairs.png' # Changed filename slightly

# ===========================================
# === END USER SETTINGS ===
# ===========================================

# --- Basic Path Validation ---
config_file = Path(config_file_path)
checkpoint_file = Path(checkpoint_file_path)
json_path = Path(processed_json_mapping_path)
output_path = Path(output_collage_save_path)

if not config_file.is_file():
    print(f"Error: Config file not found at '{config_file}'")
    exit()
if not checkpoint_file.is_file():
    print(f"Error: Checkpoint file not found at '{checkpoint_file}'")
    exit()
if not json_path.is_file():
    print(f"Error: Processed JSON mapping file not found at '{json_path}'")
    exit()

# --- Device Setup ---
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# --- Initialize Model ---
print(f"Initializing model from: {config_file.name}")
print(f"Using checkpoint:      {checkpoint_file.name}")
try:
    model = init_model(str(config_file), str(checkpoint_file), device=device)
    if device == 'cpu' and hasattr(model, 'module'):
         model = model.module
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    print("Model initialized successfully.")
except Exception as e:
    print(f"\nError initializing model: {e}")
    exit()

# --- Load Processed JSON Mapping ---
print(f"Loading processed mapping from: {json_path.name}")
try:
    with open(json_path, 'r') as f:
        processed_mapping = json.load(f)
    print(f"Loaded {len(processed_mapping)} entries.")
except Exception as e:
    print(f"\nError loading processed JSON mapping: {e}")
    exit()

# --- Select Random Samples ---
all_ids = list(processed_mapping.keys())
if len(all_ids) < num_samples:
    print(f"Warning: Requested {num_samples} samples, but only {len(all_ids)} available. Using all available.")
    num_samples = len(all_ids)
    selected_ids = all_ids
else:
    selected_ids = random.sample(all_ids, num_samples)
print(f"Selected {num_samples} random image IDs for inference.")

# --- Run Inference and Collect Images ---
input_images_for_collage = []
predicted_masks_for_collage = []
img_h, img_w = None, None # To store image dimensions

print("\nRunning inference on selected samples...")
for i, img_id in enumerate(selected_ids):
    print(f"  Processing sample {i+1}/{num_samples} (ID: {img_id})...")
    img_info = processed_mapping[img_id]

    input_img_path_str = img_info.get('img_path')
    if not input_img_path_str:
        print(f"    -> Skipped: Missing 'img_path' for ID {img_id}")
        continue
    input_img_path = Path(input_img_path_str)

    if not input_img_path.is_file():
        print(f"    -> Skipped: Processed image file not found at {input_img_path}")
        continue

    try:
        # Run inference
        result = inference_model(model, str(input_img_path))

        # Process prediction mask
        pred_mask_tensor = result.pred_sem_seg.data.squeeze().cpu().numpy()
        pred_mask_vis = (pred_mask_tensor * 255).astype(np.uint8) # 0=black, 1=white

        # Load input image for collage
        input_img_vis = cv2.imread(str(input_img_path))
        if input_img_vis is None:
             print(f"    -> Skipped: Failed to load processed image {input_img_path} for collage.")
             continue

        # Store dimensions from first valid image
        if img_h is None:
            img_h, img_w = input_img_vis.shape[:2]

        # Ensure consistent size
        if input_img_vis.shape[:2] != (img_h, img_w):
            input_img_vis = cv2.resize(input_img_vis, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        if pred_mask_vis.shape[:2] != (img_h, img_w):
             pred_mask_vis = cv2.resize(pred_mask_vis, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Add to lists for collage
        input_images_for_collage.append(input_img_vis)
        # Convert predicted mask to 3 channels (BGR) for stacking
        predicted_masks_for_collage.append(cv2.cvtColor(pred_mask_vis, cv2.COLOR_GRAY2BGR))

    except Exception as e:
        print(f"    -> Skipped: Error during inference or processing for ID {img_id}: {e}")
        continue

# ===========================================
# === NEW COLLAGE CREATION LOGIC ===
# ===========================================
num_collected = len(input_images_for_collage)
if num_collected == 0:
    print("\nNo samples were successfully processed. Cannot create collage.")
else:
    print(f"\nCreating collage from {num_collected} collected samples...")
    all_pairs = []
    for i in range(num_collected):
        input_img = input_images_for_collage[i]
        pred_mask_bgr = predicted_masks_for_collage[i]

        # Stack input image and prediction mask side-by-side
        pair = np.hstack((input_img, pred_mask_bgr))
        all_pairs.append(pair)

    # Stack all the pairs vertically
    try:
        collage = np.vstack(all_pairs)

        # Save the collage
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), collage)
        print(f"\nCollage saved successfully to: {output_path.resolve()}")
        print(f"(Collage layout: {num_collected} rows, each showing [Input Image | Predicted Mask])")

    except Exception as e:
        print(f"\nError creating or saving collage: {e}")
        print("Check if collected images/masks have consistent dimensions.")
# ===========================================
# === END NEW COLLAGE CREATION LOGIC ===
# ===========================================

print("\nScript finished.")