"""
Dataset class for loading organoid time series data WITH MEAN-FILL MASK
Background is replaced with the mean intensity of the image (keeps brightness consistent).
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import json
import torch
from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import cv2


def compute_global_mean(series_metadata, image_type='clipped'):
    """Compute mean RGB across entire dataset using series split JSON format."""
    print("Computing global dataset mean...")
    all_means = []

    for org_id in series_metadata.keys():
        for tp in series_metadata[org_id]['timepoints']:
            img_path = tp['img_paths'][image_type]
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            mask_path = tp.get('mask_paths', {}).get(image_type)
            if mask_path and Path(mask_path).exists():
                mask = imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = (mask > 127).astype(bool)
                foreground = img[mask]
                if len(foreground) > 0:
                    all_means.append(foreground.mean(axis=0))

    global_mean = np.mean(all_means, axis=0)
    print(f"Global mean RGB: {global_mean}")
    return global_mean

def compute_global_mean_from_ids(organoid_ids, series_metadata, image_type='clipped'):
    """Compute mean RGB across all pixels in training images using series split JSON format."""
    print(f"Computing global mean from {len(organoid_ids)} organoids...")
    all_means = []

    for org_id in organoid_ids:
        for tp in series_metadata[org_id]['timepoints']:
            img_path = tp['img_paths'][image_type]
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img_mean = img.reshape(-1, 3).mean(axis=0)
            all_means.append(img_mean)

    global_mean = np.mean(all_means, axis=0) / 255.0
    print(f"Global mean RGB: {global_mean}")
    return global_mean


class OrganoidTimeSeriesDataset(Dataset):
    """
    Loads organoid image sequences WITH MEAN-FILL MASK for CNN-LSTM training.
    """

    def __init__(self, organoid_ids, series_metadata,
                 transform=None, use_clipping_mask=False,
                 global_mean=None,
                 max_day=None,
                 image_type='clipped'):
        """
        Args:
            organoid_ids:    list of organoid_id strings (keys into series_metadata)
            series_metadata: dict loaded from series_train/val/test.json
                             Each entry has: label, n_votes_good, n_votes_total,
                             base_well, genealogy_type, timepoints (list of
                             {key, mdl_day, img_paths, mask_paths, ...})
            image_type:      which image variant to use — 'clipped' (default, meanfill
                             575x575) or 'std' (512x384 standard resize)
        """
        self.organoid_ids = organoid_ids
        self.series_metadata = series_metadata
        self.transform = transform
        self.use_clipping_mask = use_clipping_mask
        self.global_mean = global_mean
        self.max_day = max_day
        self.image_type = image_type
        
    def __len__(self):
        return len(self.organoid_ids)

    def get_label_from_survey(self, entry):
        """Extract binary label using majority vote from evaluators."""
        if 'survey' not in entry or not entry['survey']:
            return None

        survey = entry['survey']
        if 'evaluations' not in survey or not survey['evaluations']:
            return None

        votes = [ev.get('evaluation') for ev in survey['evaluations']]
        acceptable = votes.count('Acceptable')
        not_acceptable = votes.count('Not Acceptable')

        if acceptable > not_acceptable:
            return 1
        elif not_acceptable > acceptable:
            return 0
        else:
            return None  # Tie


    def apply_mean_fill(self, img, mask, blur_kernel=(15, 15), dilate_iterations=5):
        """
        Applies mean-fill masking with dilation and feathering for more lenient masking
        
        Args:
            img: Input image
            mask: Binary mask (0-255)
            blur_kernel: Gaussian blur kernel size for feathering
            dilate_iterations: Number of dilation iterations to expand mask
        """
        import cv2
        
        # STEP 1: Dilate mask to expand it slightly
        # This gives us a safety margin around the organoid
        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
        
        # STEP 2: Gaussian blur for soft feathering
        # This creates a gradual transition at edges instead of hard cutoff
        if blur_kernel is not None:
            mask = cv2.GaussianBlur(mask, blur_kernel, 0)
        
        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # STEP 3: Apply mean-fill
        # Use global mean if provided, otherwise per-image mean
        if self.global_mean is not None:
            mean_rgb = (self.global_mean * 255.0)[None, None, :]
        else:
            mean_rgb = img.reshape(-1, 3).mean(axis=0)[None, None, :]
        
        # Blend: mask region keeps original pixels, background gets mean
        return img * mask[:, :, None] + mean_rgb * (1.0 - mask[:, :, None])

    def __getitem__(self, idx):
        organoid_id = self.organoid_ids[idx]
        # NEW: read directly from timepoints list in split JSON
        timepoints = self.series_metadata[organoid_id]['timepoints']

        images = []
        days_used = []

        for tp in timepoints:
            # stop at max_day if set
            if self.max_day is not None and tp['mdl_day'] > self.max_day:
                break

            img_path = tp['img_paths'][self.image_type]
            img = imread(img_path)

            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img.astype(np.float32)

            if self.use_clipping_mask:
                mask_path = tp.get('mask_paths', {}).get(self.image_type)
                if mask_path and Path(mask_path).exists():
                    mask = imread(mask_path)
                    if mask.ndim == 3: mask = mask[:, :, 0]
                    img = self.apply_mean_fill(img, mask)

            if self.transform:
                from PIL import Image
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil = self.transform(img_pil)
                img = np.array(img_pil).astype(np.float32) / 255.0

            images.append(img)
            days_used.append(tp['mdl_day'])

        # stack to tensor (T,C,H,W)
        sequence = np.stack(images)
        sequence = np.transpose(sequence, (0, 3, 1, 2))
        seq = torch.from_numpy(sequence).float()

        # ImageNet normalize per frame
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)
        seq = (seq - imagenet_mean) / imagenet_std

        # label from metadata
        meta_entry = self.series_metadata[organoid_id]
        s = str(meta_entry.get("label", "")).strip().lower()
        label = 1 if s in ("good", "acceptable", "accepted") else 0

        # per-sequence day normalization
        days_arr = np.array(days_used, dtype=np.float32)
        dmin = days_arr.min()
        drng = max(days_arr.max() - dmin, 1e-8)
        days_norm = (days_arr - dmin) / drng
        days_norm = torch.from_numpy(days_norm).float()   # (T,)

        # ----- agreement-based weight (0.6/0.8/1.0 as example) -----
        n_good  = meta_entry.get("n_votes_good", None)
        n_total = meta_entry.get("n_votes_total", None)
        if n_good is not None and n_total and n_total > 0:
            frac = n_good / n_total
            if frac >= 0.9 or frac <= 0.1:
                weight = 1.0       # unanimous
            elif frac >= 0.7 or frac <= 0.3:
                weight = 0.8       # strong majority
            else:
                weight = 0.6       # weak majority / tie
        else:
            weight = 1.0

        return (
            seq, 
            days_norm, 
            torch.tensor(label, dtype=torch.float32), 
            torch.tensor(weight, dtype=torch.float32), 
            organoid_id,   # keep the raw string id for FP/FN logs
        )




def resolve_split_path(splits_dir, phase):
    """
    Given a directory and a phase ('train' | 'val' | 'test'), return the path
    to the matching split JSON. Supports both layouts so the trainers can read
    from either:
        new (cohort-style):  <splits_dir>/<phase>.json
        old (data_splits/):  <splits_dir>/<phase>_idor_series.json
    Cohort layout takes precedence when both exist.
    """
    d = Path(splits_dir)
    cohort_style = d / f"{phase}.json"
    legacy_style = d / f"{phase}_idor_series.json"
    if cohort_style.exists():
        return str(cohort_style)
    if legacy_style.exists():
        return str(legacy_style)
    raise FileNotFoundError(
        f"No split file for phase '{phase}' in {d}. Looked for "
        f"{cohort_style.name} and {legacy_style.name}."
    )


def load_split_from_json(split_path):
    """
    Load a pre-made split JSON produced by scripts/split_series_reproducible.py.

    Returns:
        organoid_ids  - list of organoid_id strings
        series_data   - dict {organoid_id: {label, n_votes_good, n_votes_total,
                                            base_well, genealogy_type, n_timepoints,
                                            timepoints: [{key, mdl_day, img_path,
                                                          mask_path, ...}]}}

    Usage:
        train_ids, train_data = load_split_from_json('data_splits/series_train.json')
        val_ids,   val_data   = load_split_from_json('data_splits/series_val.json')
        test_ids,  test_data  = load_split_from_json('data_splits/series_test.json')

        train_dataset = OrganoidTimeSeriesDataset(train_ids, train_data, ...)
    """
    with open(split_path) as f:
        series_data = json.load(f)

    organoid_ids = list(series_data.keys())

    labels = [series_data[oid]['label'] for oid in organoid_ids]
    acc    = labels.count('Acceptable')
    na     = labels.count('Not Acceptable')
    print(f"Loaded {len(organoid_ids)} organoids from {split_path} "
          f"({acc} Acceptable, {na} Not Acceptable)")

    return organoid_ids, series_data
