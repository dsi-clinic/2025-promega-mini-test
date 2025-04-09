# quick_inference.py
import torch
from pathlib import Path
import cv2 # To save the output mask
import numpy as np
import warnings
import os

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
    print("and you are running this script from the correct Python environment.")
    exit()

# ===========================================
# === USER: SET THESE PATHS BELOW ===
# ===========================================

# 1. Path to the configuration file from your work directory
config_file_path = '/net/projects2/promega/data-analysis/plots/segformer_masks/20250409_010027/vis_data/config.py'

# 2. Path to the checkpoint (.pth) file you want to use
checkpoint_file_path = '/net/projects2/promega/data-analysis/plots/segformer_masks/iter_1000.pth'

# 3. Path to an example input image you want to segment
#    The pipeline defined in config.py should handle preprocessing like resizing.
input_image_path = '/net/projects2/promega/data-analysis/output/processed_dataset_256x192/images/Ba1 96_1 Dy30 A12_256x192.tif'

# 4. Path where you want to save the output prediction mask image
output_mask_save_path = './inference_prediction.png' # Saves in the current directory

# ===========================================
# === END USER SETTINGS ===
# ===========================================

# --- Basic Path Validation ---
config_file = Path(config_file_path)
checkpoint_file = Path(checkpoint_file_path)
img_path = Path(input_image_path)
output_path = Path(output_mask_save_path)

if not config_file.is_file():
    print(f"Error: Config file not found at '{config_file}'")
    exit()
if not checkpoint_file.is_file():
    print(f"Error: Checkpoint file not found at '{checkpoint_file}'")
    exit()
if not img_path.is_file():
    print(f"Error: Input image file not found at '{img_path}'")
    exit()

# --- Device Setup ---
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

# --- Initialize Model ---
print(f"Initializing model from: {config_file.name}")
print(f"Using checkpoint:      {checkpoint_file.name}")
try:
    # Build the model from the config and checkpoint
    model = init_model(str(config_file), str(checkpoint_file), device=device)
    if device == 'cpu' and hasattr(model, 'module'): # Check if wrapped (e.g., DDP)
         model = model.module
    if device == 'cpu':
        model = revert_sync_batchnorm(model)

    print("Model initialized successfully.")
except Exception as e:
    print(f"\nError initializing model: {e}")
    print("Please check config and checkpoint paths and file integrity.")
    exit()

# --- Perform Inference ---
print(f"\nRunning inference on: {img_path.name}")
try:
    # inference_model handles preprocessing according to the config's pipeline
    result = inference_model(model, str(img_path))
    print("Inference completed.")
except Exception as e:
    print(f"\nError during inference: {e}")
    exit()

# --- Process and Save Output ---
try:
    # The result object (SegDataSample) contains the prediction data
    # Accessing the predicted semantic map (contains class indices)
    # Shape is likely (1, H, W) or (H, W)
    pred_mask_tensor = result.pred_sem_seg.data.squeeze().cpu().numpy()

    # Convert class indices (0, 1, ...) to a visualizable grayscale mask
    # Background (class 0) -> 0 (black)
    # Cell (class 1, if any predicted) -> 255 (white)
    # We expect this model to predict mostly 0s.
    output_mask_vis = (pred_mask_tensor * 255).astype(np.uint8)

    # Ensure output directory exists (if saving to a subdirectory)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the prediction mask using OpenCV
    cv2.imwrite(str(output_path), output_mask_vis)
    print(f"\nPrediction mask saved to: {output_path.resolve()}")


except AttributeError:
    print("\nError: Could not extract prediction ('pred_sem_seg.data') from the result.")
    print("The inference result structure might have changed or inference failed.")
    # print(f"Inference result object: {result}") # Uncomment to inspect result structure
except Exception as e:
    print(f"\nError processing or saving the output mask: {e}")