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


def compute_global_mean(series_metadata, data):
    """Compute mean RGB across entire dataset"""
    print("Computing global dataset mean...")
    all_means = []
    
    for org_id in series_metadata.keys():
        entry_keys = series_metadata[org_id]['entry_keys']
        for key in entry_keys:
            img_path = data[key]['lstm_processed']['image_path']
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            
            # Get mean of foreground pixels only (use mask if available)
            mask_path = data[key]['lstm_processed'].get('mask_path')
            if mask_path and Path(mask_path).exists():
                mask = imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = (mask > 127).astype(bool)  # Binary mask
                
                # Mean of only foreground pixels
                foreground = img[mask]
                if len(foreground) > 0:
                    all_means.append(foreground.mean(axis=0))
    
    global_mean = np.mean(all_means, axis=0)
    print(f"Global mean RGB: {global_mean}")
    return global_mean

def compute_global_mean_from_ids(organoid_ids, series_metadata, data):
    """Compute mean RGB across all pixels in training images (simple!)"""
    print(f"Computing global mean from {len(organoid_ids)} organoids...")
    all_means = []
    
    for org_id in organoid_ids:
        entry_keys = series_metadata[org_id]['entry_keys']
        for key in entry_keys:
            img_path = data[key]['lstm_processed']['image_path']
            img = imread(img_path)
            
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            
            # Just get mean of ENTIRE image (all pixels)
            img_mean = img.reshape(-1, 3).mean(axis=0)
            all_means.append(img_mean)
    
    # Average across all images
    global_mean = np.mean(all_means, axis=0) / 255.0  # Normalize to [0,1]
    print(f"Global mean RGB: {global_mean}")
    return global_mean


class OrganoidTimeSeriesDataset(Dataset):
    """
    Loads organoid image sequences WITH MEAN-FILL MASK for CNN-LSTM training.
    """

    def __init__(self, organoid_ids, series_metadata, data, 
                 transform=None, use_clipping_mask=True, 
                 global_mean=None,
                 max_day=None):
        self.organoid_ids = organoid_ids
        self.series_metadata = series_metadata
        self.data = data
        self.transform = transform
        self.use_clipping_mask = use_clipping_mask
        self.global_mean = global_mean
        self.max_day = max_day
        
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

    def apply_mean_fill(self, img, mask, blur_kernel=(7, 7)):
        """Applies mean-fill masking with Gaussian blur."""
        if blur_kernel is not None:
            mask = cv2.GaussianBlur(mask, blur_kernel, 0)
        
        # Use global mean if provided, otherwise per-image mean
        if self.global_mean is not None:
            mean_rgb = (self.global_mean * 255.0)[None, None, :]
        else:
            mean_rgb = img.reshape(-1, 3).mean(axis=0)[None, None, :]
        
        return img * mask[:, :, None] + mean_rgb * (1.0 - mask[:, :, None])

    def __getitem__(self, idx):
        organoid_id = self.organoid_ids[idx]
        entry_keys = self.series_metadata[organoid_id]['entry_keys']
        days = self.series_metadata[organoid_id]['days']
        images = []

        for i, key in enumerate(entry_keys):
            # Filter by max_day if specified
            if self.max_day is not None and days[i] >= (self.max_day + 1):
                break
            
            entry = self.data[key]
            img_path = entry['lstm_processed']['image_path']
            img = imread(img_path)

            # Convert grayscale → RGB
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img.astype(np.float32)

            if self.use_clipping_mask:
                mask_path = entry['lstm_processed'].get('mask_path')
                if mask_path and Path(mask_path).exists():
                    mask = imread(mask_path)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    mask = mask.astype(np.float32) / 255.0
                    img = self.apply_mean_fill(img, mask)

            img = np.clip(img / 255.0, 0, 1)

            if self.transform:
                img = self.transform(img)

            images.append(img)

        sequence = np.stack(images)  # (T, H, W, C)
        sequence = np.transpose(sequence, (0, 3, 1, 2))  # (T, C, H, W)

        final_entry = self.data[entry_keys[-1]]
        label = self.get_label_from_survey(final_entry)

        return torch.FloatTensor(sequence), torch.LongTensor([label])[0]


def load_data_and_create_splits(series_metadata_path, data_path,
                                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                random_seed=42):
    """Same splitting logic as before."""
    with open(series_metadata_path) as f:
        series_metadata = json.load(f)
    with open(data_path) as f:
        data = json.load(f)

    def get_label(organoid_id):
        entry_keys = series_metadata[organoid_id]['entry_keys']
        final_entry = data[entry_keys[-1]]
        if 'survey' not in final_entry or not final_entry['survey']:
            return None
        survey = final_entry['survey']
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
            return None

    labeled_ids, labels = [], []
    for oid in series_metadata.keys():
        label = get_label(oid)
        if label is not None:
            labeled_ids.append(oid)
            labels.append(label)

    np.random.seed(random_seed)
    idxs = np.random.permutation(len(labeled_ids))
    n_train = int(len(labeled_ids) * train_ratio)
    n_val = int(len(labeled_ids) * val_ratio)

    train_ids = [labeled_ids[i] for i in idxs[:n_train]]
    val_ids = [labeled_ids[i] for i in idxs[n_train:n_train + n_val]]
    test_ids = [labeled_ids[i] for i in idxs[n_train + n_val:]]

    print(f"Total labeled: {len(labeled_ids)} | Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Good: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%) | Bad: {len(labels)-sum(labels)}")

    return train_ids, val_ids, test_ids, series_metadata, data