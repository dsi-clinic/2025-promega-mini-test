"""
Generate mean-filled clipping-mask images that exactly match the training view.
"""

import json, cv2, numpy as np
from pathlib import Path
from skimage.io import imread, imsave
from tqdm import tqdm

# ===== Config =====
DATA_JSON = Path("/net/projects2/promega/data-analysis/output/complete_series_data_no_blanks.json")
GLOBAL_MEAN_PATH = Path("/net/projects2/promega/data-analysis/output/cnn_lstm/global_mean.npy")
OUT_DIR = Path("/net/projects2/promega/data-analysis/output/lstm_ready/clipping_mask_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUR_KERNEL = (15, 15)
DILATE_ITERATIONS = 5

def to_rgb(img):
    return np.stack([img]*3, axis=-1) if img.ndim == 2 else img

def apply_mean_fill(img, mask, global_mean, blur_kernel, dilate_iterations):
    img = img.astype(np.float32)
    if mask.ndim == 3: mask = mask[:, :, 0]
    mask = mask.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    mask = mask.astype(np.float32) / 255.0
    
    mean_rgb = (global_mean * 255.0)[None, None, :]
    filled = img * mask[:, :, None] + mean_rgb * (1 - mask[:, :, None])
    return np.clip(filled / 255.0, 0, 1)

# ===== Load =====
with open(DATA_JSON) as f:
    data = json.load(f)
global_mean = np.load(GLOBAL_MEAN_PATH)

# ===== Process =====
for key, entry in tqdm(data.items(), desc="Generating clipped mean-filled images"):
    proc = entry.get("lstm_processed", {})
    img_path = proc.get("image_path")
    mask_path = proc.get("mask_path")
    if not img_path or not mask_path: 
        continue
    if not Path(img_path).exists() or not Path(mask_path).exists():
        continue

    img = to_rgb(imread(img_path))
    mask = imread(mask_path)
    filled = apply_mean_fill(img, mask, global_mean, BLUR_KERNEL, DILATE_ITERATIONS)
    
    # Save (convert to uint8)
    out_path = OUT_DIR / f"{Path(img_path).stem}_filled.png"
    imsave(out_path, (filled * 255).astype(np.uint8), check_contrast=False)
    entry["lstm_processed"]["clipped_image_path"] = str(out_path)

# ===== Save updated JSON =====
with open(DATA_JSON, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Done. Saved filled images to {OUT_DIR}")
