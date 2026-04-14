"""
Preprocessing for stitched microscopy images.
Removes two common artifacts:
  1. Red scale bar  (remove_red_scalebar)
  2. Black corner boxes  (remove_corner_blackbox)

Original code by Liya Ding.

Usage in dataset __getitem__:
    from image_classifier.preprocessing.stitched_preprocessing import preprocess_stitched

    img = imread(img_path)                         # uint8 numpy
    img = preprocess_stitched(img, img_path)        # no-op if not stitched
"""

from scipy.interpolate import interp1d
from skimage.restoration import inpaint_biharmonic
import numpy as np


def remove_red_scalebar(img):
    """Remove red scale bar by inpainting pixels where R > G."""
    img_out = img.copy()
    diff = img[:, :, 0].astype(np.int16) - img[:, :, 1].astype(np.int16)
    mask = diff > 0
    r_ch = img[:, :, 0]
    out = inpaint_biharmonic(r_ch, mask)
    img_out[:, :, 0] = (out * 255).clip(0, 255).astype(np.uint8)
    img_out[:, :, 1] = (out * 255).clip(0, 255).astype(np.uint8)
    img_out[:, :, 2] = (out * 255).clip(0, 255).astype(np.uint8)
    return img_out


def _fill_edge_line(x, mask_missing):
    """Interpolate missing edge pixels."""
    mask_missing = mask_missing > 0
    t = np.arange(len(x))
    known = ~mask_missing
    if not known.any():
        return x
    f = interp1d(t[known], x[known], kind="linear", fill_value="extrapolate")
    x_filled = x.copy()
    x_filled[mask_missing] = f(t[mask_missing])
    return x_filled


def remove_corner_blackbox(img):
    """Remove black corner boxes by edge interpolation + inpainting."""
    img_out = img.copy()
    mask = (img[:, :, 0] == 0) * 255
    r_ch = img[:, :, 0].copy()

    top_row = img[0, :, 0]
    top_int = top_row[top_row > 0].mean() if (top_row > 0).any() else 128
    bottom_row = img[-1, :, 0]
    bottom_int = bottom_row[bottom_row > 0].mean() if (bottom_row > 0).any() else 128
    left_col = img[:, 0, 0]
    left_int = left_col[left_col > 0].mean() if (left_col > 0).any() else 128
    right_col = img[:, -1, 0]
    right_int = right_col[right_col > 0].mean() if (right_col > 0).any() else 128

    if r_ch[0, 0] == 0:
        r_ch[0, 0] = (top_int + left_int) / 2
    if r_ch[0, -1] == 0:
        r_ch[0, -1] = (top_int + right_int) / 2
    if r_ch[-1, 0] == 0:
        r_ch[-1, 0] = (bottom_int + left_int) / 2
    if r_ch[-1, -1] == 0:
        r_ch[-1, -1] = (bottom_int + right_int) / 2
    mask[0, 0] = 0
    mask[0, -1] = 0
    mask[-1, 0] = 0
    mask[-1, -1] = 0

    for edge_slice in [
        (0, slice(None)),  # top row
        (-1, slice(None)),  # bottom row
        (slice(None), 0),  # left col
        (slice(None), -1),  # right col
    ]:
        edge_mask = mask[edge_slice]
        if edge_mask.max() > 0:
            r_ch[edge_slice] = _fill_edge_line(r_ch[edge_slice], edge_mask)

    mask_bool = mask > 0
    out = inpaint_biharmonic(r_ch, mask_bool)
    img_out[:, :, 0] = (out * 255).clip(0, 255).astype(np.uint8)
    img_out[:, :, 1] = (out * 255).clip(0, 255).astype(np.uint8)
    img_out[:, :, 2] = (out * 255).clip(0, 255).astype(np.uint8)
    return img_out


# ---------------------------------------------------------------------------
# Convenience wrappers used in dataset __getitem__ methods
# ---------------------------------------------------------------------------


def preprocess_stitched(img, img_path):
    """
    Apply stitched-image artifact removal if 'stitched' appears in the path.
    img:      numpy uint8 array (H, W, 3)
    img_path: str or Path – only processed when filename contains 'stitched'
    Returns:  numpy uint8 array (H, W, 3)
    """
    if "stitched" not in str(img_path).lower():
        return img
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = remove_red_scalebar(img)
    img = remove_corner_blackbox(img)
    return img


def preprocess_stitched_pil(pil_img, img_path):
    """
    PIL wrapper: convert to numpy, preprocess, convert back.
    pil_img:  PIL.Image (RGB)
    img_path: str or Path
    Returns:  PIL.Image (RGB)
    """
    if "stitched" not in str(img_path).lower():
        return pil_img
    from PIL import Image as _Image

    arr = np.array(pil_img)
    arr = preprocess_stitched(arr, img_path)
    return _Image.fromarray(arr)
