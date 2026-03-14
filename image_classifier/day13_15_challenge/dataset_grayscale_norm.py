"""
Dataset variant that normalizes overlay (or grayscale) with mean/std computed from
grayscale image pixels only (not from overlay RGB). For use in Day 13/15 challenge.
"""

from pathlib import Path
import sys
import numpy as np
from skimage.io import imread

# Repo root = parent of day13_15_challenge
CHALLENGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHALLENGE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_classifier.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    _draw_outline_overlay_bgr,
)
from image_classifier.preprocessing.stitched_preprocessing import preprocess_stitched


def compute_grayscale_mean_std(organoid_ids, series_metadata, data, max_day):
    """
    Compute mean and std over grayscale pixel values [0,1] from all training
    frames with day <= max_day. Returns (mean, std) each shape (3,) for R=G=B.
    Uses image_path (raw grayscale), not overlay.
    """
    all_pixels = []
    for oid in organoid_ids:
        entry_keys = series_metadata.get(oid, {}).get("entry_keys", [])
        days = series_metadata.get(oid, {}).get("days", [])
        for i, key in enumerate(entry_keys):
            if max_day is not None and i < len(days) and days[i] > max_day:
                continue
            entry = data.get(key, {})
            lp = entry.get("lstm_processed", {})
            img_path = lp.get("image_path") or lp.get("clipped_image_path")
            if not img_path or not Path(img_path).exists():
                continue
            img = imread(img_path)
            if img.ndim == 3:
                img = img[:, :, 0]
            img = img.astype(np.float32) / 255.0
            all_pixels.append(img.ravel())
    if not all_pixels:
        # fallback to ImageNet-like
        return np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
    flat = np.concatenate(all_pixels)
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    if std_val < 1e-8:
        std_val = 1.0
    # same for R, G, B (grayscale)
    mean_3 = np.array([mean_val, mean_val, mean_val], dtype=np.float32)
    std_3 = np.array([std_val, std_val, std_val], dtype=np.float32)
    return mean_3, std_3


class OrganoidTimeSeriesDatasetGrayscaleNorm(OrganoidTimeSeriesDataset):
    """
    Same as OrganoidTimeSeriesDataset but normalizes with custom (mean, std)
    e.g. from grayscale-only stats. custom_mean, custom_std: shape (3,) or (1,3,1,1).
    """

    def __init__(
        self, organoid_ids, series_metadata, data, custom_mean, custom_std, **kwargs
    ):
        super().__init__(organoid_ids, series_metadata, data, **kwargs)
        self.custom_mean = np.asarray(custom_mean, dtype=np.float32)
        self.custom_std = np.asarray(custom_std, dtype=np.float32)
        if self.custom_mean.ndim == 1:
            self.custom_mean = self.custom_mean.reshape(1, 3, 1, 1)
        if self.custom_std.ndim == 1:
            self.custom_std = self.custom_std.reshape(1, 3, 1, 1)

    def __getitem__(self, idx):
        import torch

        organoid_id = self.organoid_ids[idx]
        entry_keys = self.series_metadata[organoid_id]["entry_keys"]
        days = self.series_metadata[organoid_id]["days"]
        images = []
        days_used = []

        for i, key in enumerate(entry_keys):
            if self.max_day is not None and days[i] > self.max_day:
                break
            entry = self.data[key]
            lp = entry["lstm_processed"]
            img_path = lp.get("clipped_image_path", lp["image_path"])
            overlay_path = lp.get("overlay_path", "")
            mask_path = lp.get("mask_path", "")

            if (
                self.image_key == "overlay_path"
                and overlay_path
                and Path(overlay_path).exists()
            ):
                img = imread(overlay_path)
            elif (
                self.image_key == "overlay_path"
                and mask_path
                and Path(mask_path).exists()
            ):
                img = imread(img_path)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                img = preprocess_stitched(img, img_path)
                mask = imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask_bin = (mask > 127).astype(np.uint8)
                img = _draw_outline_overlay_bgr(img.astype(np.uint8), mask_bin)
                img = img.astype(np.float32) / 255.0
            else:
                img = imread(img_path)

            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = preprocess_stitched(img, img_path)
            img = img.astype(np.float32) / 255.0

            if self.use_clipping_mask and not self.input_rgb_mask:
                if mask_path and Path(mask_path).exists():
                    mask = imread(mask_path)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    img = self.apply_mean_fill(img, mask)

            if self.input_rgb_mask and mask_path and Path(mask_path).exists():
                mask = imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask_bin = (mask > 127).astype(np.float32)
                mask_bin = np.expand_dims(mask_bin, axis=-1)
                img = np.concatenate([img, mask_bin], axis=-1)

            if self.transform:
                from PIL import Image

                ch3 = img[:, :, :3] if img.shape[-1] == 4 else img
                img_pil = Image.fromarray(
                    (ch3 * 255).astype(np.uint8)
                    if ch3.max() <= 1
                    else ch3.astype(np.uint8)
                )
                img_pil = self.transform(img_pil)
                ch3 = np.array(img_pil).astype(np.float32) / 255.0
                if self.input_rgb_mask and img.shape[-1] == 4:
                    from scipy.ndimage import zoom

                    m = img[:, :, 3:4]
                    m = zoom(
                        m,
                        (ch3.shape[0] / m.shape[0], ch3.shape[1] / m.shape[1], 1),
                        order=0,
                    )
                    img = np.concatenate([ch3, m], axis=-1)
                else:
                    img = ch3

            images.append(img)
            days_used.append(days[i])

        sequence = np.stack(images)
        sequence = np.transpose(sequence, (0, 3, 1, 2))
        seq = torch.from_numpy(sequence).float()
        C = seq.shape[1]
        if C == 3:
            mean_t = torch.from_numpy(self.custom_mean.reshape(1, 3, 1, 1)).float()
            std_t = torch.from_numpy(self.custom_std.reshape(1, 3, 1, 1)).float()
        else:
            mean_t = torch.tensor([0.485, 0.456, 0.406, 0.5], dtype=torch.float32).view(
                1, 4, 1, 1
            )
            std_t = torch.tensor([0.229, 0.224, 0.225, 0.5], dtype=torch.float32).view(
                1, 4, 1, 1
            )
        seq = (seq - mean_t) / std_t

        meta_entry = self.series_metadata[organoid_id]
        s = str(meta_entry.get("label", "")).strip().lower()
        label = 1 if s in ("good", "acceptable", "accepted") else 0

        days_arr = np.array(days_used, dtype=np.float32)
        dmin = days_arr.min()
        drng = max(days_arr.max() - dmin, 1e-8)
        days_norm = (days_arr - dmin) / drng
        days_norm = torch.from_numpy(days_norm).float()

        n_good = meta_entry.get("n_votes_good", None)
        n_total = meta_entry.get("n_votes_total", None)
        if n_good is not None and n_total and n_total > 0:
            frac = n_good / n_total
            if frac >= 0.9 or frac <= 0.1:
                weight = 1.0
            elif frac >= 0.7 or frac <= 0.3:
                weight = 0.8
            else:
                weight = 0.6
        else:
            weight = 1.0

        return (
            seq,
            days_norm,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            organoid_id,
        )


def _single_day_getitem_grayscale_norm(self, idx):
    """__getitem__ for SingleDayOrganoidDataset that re-normalizes with custom mean/std."""
    import torch
    from skimage.io import imread
    from PIL import Image
    from image_classifier.cnn_lstm.train_base_model import (
        _draw_outline_overlay,
    )

    sample = self.samples[idx]
    if (
        self.image_key == "overlay_path"
        and sample.get("overlay_path")
        and Path(sample["overlay_path"]).exists()
    ):
        img = imread(sample["overlay_path"])
    elif (
        self.image_key == "overlay_path"
        and sample.get("mask_path")
        and Path(sample["mask_path"]).exists()
    ):
        img = imread(sample["img_path"])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = preprocess_stitched(img, sample["img_path"])
        mask = imread(sample["mask_path"])
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_bin = (mask > 127).astype(np.uint8)
        img = _draw_outline_overlay(img.astype(np.uint8), mask_bin)
        img = img.astype(np.float32)
    else:
        img = imread(sample["img_path"])
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = preprocess_stitched(img, sample["img_path"])
    img = img.astype(np.float32) / 255.0
    if (
        self.use_rgb_mask
        and sample.get("mask_path")
        and Path(sample["mask_path"]).exists()
    ):
        mask = imread(sample["mask_path"])
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_bin = (mask > 127).astype(np.float32)
        mask_bin = np.expand_dims(mask_bin, axis=-1)
        img = np.concatenate([img, mask_bin], axis=-1)
    if self.transform:
        ch3 = img[:, :, :3]
        img_pil = Image.fromarray((ch3 * 255).astype(np.uint8))
        img_pil = self.transform(img_pil)
        ch3 = np.array(img_pil).astype(np.float32) / 255.0
        if self.use_rgb_mask and img.shape[-1] == 4:
            from scipy.ndimage import zoom

            m = img[:, :, 3:4]
            h, w = ch3.shape[:2]
            m = zoom(m, (h / m.shape[0], w / m.shape[1], 1), order=0)
            img = np.concatenate([ch3, m], axis=-1)
        else:
            img = ch3
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    C = img.shape[0]
    m = np.asarray(self._custom_mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(self._custom_std, dtype=np.float32).reshape(3, 1, 1)
    mean_t = torch.from_numpy(m).float()
    std_t = torch.from_numpy(s).float()
    if C == 4:
        mean_t = torch.cat([mean_t, torch.tensor([[0.5]], dtype=torch.float32)], dim=0)
        std_t = torch.cat([std_t, torch.tensor([[0.5]], dtype=torch.float32)], dim=0)
    img = (img - mean_t) / std_t
    label = torch.tensor(sample["label"], dtype=torch.float32)
    return img, label, sample["org_id"]


def make_single_day_grayscale_norm_dataset(
    train_ids,
    val_ids,
    test_ids,
    series_metadata,
    data,
    day,
    custom_mean,
    custom_std,
    transform,
    image_key="overlay_path",
    use_rgb_mask=False,
):
    """Build SingleDayOrganoidDataset that normalizes with grayscale-derived mean/std."""
    from image_classifier.cnn_lstm.train_base_model import SingleDayOrganoidDataset

    mean_3 = np.asarray(custom_mean, dtype=np.float32)
    std_3 = np.asarray(custom_std, dtype=np.float32)
    if mean_3.size == 3:
        mean_3 = mean_3.reshape(3)
        std_3 = std_3.reshape(3)

    class _SingleDayGrayscaleNorm(SingleDayOrganoidDataset):
        def __init__(self, *args, custom_mean=None, custom_std=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._custom_mean = custom_mean.reshape(3)
            self._custom_std = custom_std.reshape(3)

        def __getitem__(self, idx):
            return _single_day_getitem_grayscale_norm(self, idx)

    train_ds = _SingleDayGrayscaleNorm(
        train_ids,
        series_metadata,
        data,
        day,
        transform=transform,
        image_key=image_key,
        use_rgb_mask=use_rgb_mask,
        custom_mean=mean_3,
        custom_std=std_3,
    )
    val_ds = _SingleDayGrayscaleNorm(
        val_ids,
        series_metadata,
        data,
        day,
        transform=transform,
        image_key=image_key,
        use_rgb_mask=use_rgb_mask,
        custom_mean=mean_3,
        custom_std=std_3,
    )
    test_ds = _SingleDayGrayscaleNorm(
        test_ids,
        series_metadata,
        data,
        day,
        transform=transform,
        image_key=image_key,
        use_rgb_mask=use_rgb_mask,
        custom_mean=mean_3,
        custom_std=std_3,
    )
    return train_ds, val_ds, test_ds
