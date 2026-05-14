"""
PyTorch Dataset for organoid time-series CNN-LSTM training.

Reads from the runtime ``OrganoidDataset`` (no materialized split JSONs).
Background pixels are mean-filled so the model focuses on organoid texture
rather than the surrounding well.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from pipeline.data_loader import (
    LABEL_TO_INT,
    OrganoidDataset,
    filters_for_mode,
    get_clipped_meanfill_image_path,
    get_clipped_meanfill_mask_path,
    get_day_float,
    get_survey_vote_counts,
    split_organoids,
)
from pipeline.splits import Splits

LABEL_DAY = "Dy30"


def make_idor_series_splits(
    all_data_path: str = "data/all_data.json",
    *,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
):
    """Build the IDOR-series cohort and partition it deterministically.

    Replaces ``load_split_from_json('data_splits/...json')``. Returns
    ``(dataset, train_ids, val_ids, test_ids)`` — pass each id list plus the
    shared ``dataset`` to ``OrganoidTimeSeriesDataset``.
    """
    dataset = OrganoidDataset(
        all_data_path,
        filters=filters_for_mode("series_idor"),
    )
    train_ids, val_ids, test_ids = split_organoids(
        dataset, seed=seed, test_size=test_size, val_size=val_size,
    )
    splits = Splits.from_partition(
        train=train_ids, val=val_ids, test=test_ids,
        name=f"series_idor_seed{seed}",
        provenance=f"split_organoids(seed={seed}, test_size={test_size}, val_size={val_size})",
    )
    dataset.apply_splits(splits)
    print(
        f"IDOR-series cohort: {len(dataset.organoid_ids)} organoids "
        f"-> train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return dataset, train_ids, val_ids, test_ids


def compute_global_mean(dataset: OrganoidDataset, organoid_ids, image_type="clipped"):
    """Compute mean RGB across foreground pixels in the given organoids."""
    print(f"Computing global mean from {len(organoid_ids)} organoids (foreground)...")
    all_means = []
    for oid in organoid_ids:
        for rec in dataset.organoid_records(oid).values():
            img_path = (
                get_clipped_meanfill_image_path(rec)
                if image_type == "clipped"
                else (rec.get("images") or {}).get("img_path")
            )
            mask_path = (
                get_clipped_meanfill_mask_path(rec)
                if image_type == "clipped"
                else (rec.get("images") or {}).get("mask_path")
            )
            if not img_path:
                continue
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if mask_path and Path(mask_path).exists():
                mask = imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                fg = img[(mask > 127).astype(bool)]
                if len(fg) > 0:
                    all_means.append(fg.mean(axis=0))
    global_mean = np.mean(all_means, axis=0)
    print(f"Global mean RGB: {global_mean}")
    return global_mean


def compute_global_mean_from_ids(dataset: OrganoidDataset, organoid_ids, image_type="clipped"):
    """Compute mean RGB across all pixels in the given organoids' images."""
    print(f"Computing global mean from {len(organoid_ids)} organoids (all pixels)...")
    all_means = []
    for oid in organoid_ids:
        for rec in dataset.organoid_records(oid).values():
            img_path = (
                get_clipped_meanfill_image_path(rec)
                if image_type == "clipped"
                else (rec.get("images") or {}).get("img_path")
            )
            if not img_path:
                continue
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            all_means.append(img.reshape(-1, 3).mean(axis=0))
    global_mean = np.mean(all_means, axis=0) / 255.0
    print(f"Global mean RGB: {global_mean}")
    return global_mean


class OrganoidTimeSeriesDataset(Dataset):
    """Loads organoid image sequences from an ``OrganoidDataset``."""

    def __init__(
        self,
        organoid_ids,
        dataset: OrganoidDataset,
        transform=None,
        use_clipping_mask=False,
        global_mean=None,
        max_day=None,
        image_type="clipped",
    ):
        """
        Args:
            organoid_ids: list of organoid_id strings (must be present in ``dataset``).
            dataset:      OrganoidDataset built with ``filters_for_mode("series_idor")``.
            image_type:   'clipped' (575x575 mean-fill) or 'std' (512x384 standard).
        """
        self.organoid_ids = list(organoid_ids)
        self.dataset = dataset
        self.transform = transform
        self.use_clipping_mask = use_clipping_mask
        self.global_mean = global_mean
        self.max_day = max_day
        self.image_type = image_type

    def __len__(self):
        return len(self.organoid_ids)

    def _image_path(self, record):
        if self.image_type == "clipped":
            return get_clipped_meanfill_image_path(record)
        return (record.get("images") or {}).get("img_path")

    def _mask_path(self, record):
        if self.image_type == "clipped":
            return get_clipped_meanfill_mask_path(record)
        return (record.get("images") or {}).get("mask_path")

    def apply_mean_fill(self, img, mask, blur_kernel=(15, 15), dilate_iterations=5):
        """Mean-fill the background using a dilated, feathered mask."""
        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
        if blur_kernel is not None:
            mask = cv2.GaussianBlur(mask, blur_kernel, 0)
        mask = mask.astype(np.float32) / 255.0
        if self.global_mean is not None:
            mean_rgb = (self.global_mean * 255.0)[None, None, :]
        else:
            mean_rgb = img.reshape(-1, 3).mean(axis=0)[None, None, :]
        return img * mask[:, :, None] + mean_rgb * (1.0 - mask[:, :, None])

    def __getitem__(self, idx):
        organoid_id = self.organoid_ids[idx]
        records = self.dataset.organoid_records(organoid_id)

        # Sort timepoints by mdl_day (3.0, 6.0, ..., 20.5, 24.0, 28.0, 30.0)
        timepoints = sorted(
            ((get_day_float(day) or 0.0, day, rec) for day, rec in records.items()),
            key=lambda t: t[0],
        )

        images = []
        days_used = []
        for mdl_day, _day_id, rec in timepoints:
            if self.max_day is not None and mdl_day > self.max_day:
                break

            img_path = self._image_path(rec)
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img.astype(np.float32)

            if self.use_clipping_mask:
                mask_path = self._mask_path(rec)
                if mask_path and Path(mask_path).exists():
                    mask = imread(mask_path)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    img = self.apply_mean_fill(img, mask)

            if self.transform:
                from PIL import Image
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil = self.transform(img_pil)
                img = np.array(img_pil).astype(np.float32) / 255.0

            images.append(img)
            days_used.append(mdl_day)

        sequence = np.stack(images)
        sequence = np.transpose(sequence, (0, 3, 1, 2))
        seq = torch.from_numpy(sequence).float()

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        seq = (seq - imagenet_mean) / imagenet_std

        # Label per LABEL_TO_INT: 1 = Not Acceptable, 0 = Acceptable (rule #9).
        label_str = self.dataset.organoid_label(organoid_id) or ""
        label = LABEL_TO_INT.get(label_str, 0)

        days_arr = np.array(days_used, dtype=np.float32)
        dmin = days_arr.min()
        drng = max(days_arr.max() - dmin, 1e-8)
        days_norm = torch.from_numpy((days_arr - dmin) / drng).float()

        # Sample weight from Dy30 vote agreement
        dy30_rec = records.get(LABEL_DAY)
        n_good, n_total = (0, 0) if dy30_rec is None else get_survey_vote_counts(dy30_rec)
        if n_total > 0:
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
