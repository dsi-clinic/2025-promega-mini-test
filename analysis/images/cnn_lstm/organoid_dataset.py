"""
Dataset class for loading organoid time series data WITH MASKS
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

class OrganoidTimeSeriesDataset(Dataset):
    """
    Loads organoid image sequences WITH MASKS for CNN-LSTM training
    
    Each sample is:
    - Input: Sequence of 11 images (Days 3-30), each with 4 channels (RGB + Mask)
    - Label: Binary (1=Good/Acceptable, 0=Bad/Not Acceptable)
    """
    
    def __init__(self, organoid_ids, series_metadata, data, transform=None, use_masks=True):
        """
        Args:
            organoid_ids: List of organoid IDs to include in this dataset
            series_metadata: Dict mapping organoid_id -> temporal info
            data: Dict with all entry data
            transform: Optional image transformations
            use_masks: Whether to load and use masks (4 channels) or just RGB (3 channels)
        """
        self.organoid_ids = organoid_ids
        self.series_metadata = series_metadata
        self.data = data
        self.transform = transform
        self.use_masks = use_masks
    
    def __len__(self):
        return len(self.organoid_ids)
    
    def get_label_from_survey(self, entry):
        """Extract binary label using majority vote from evaluators"""
        if 'survey' not in entry or not entry['survey']:
            return None
        
        survey = entry['survey']
        if 'evaluations' not in survey or not survey['evaluations']:
            return None
        
        # Count votes
        votes = [ev.get('evaluation') for ev in survey['evaluations']]
        acceptable_count = votes.count('Acceptable')
        not_acceptable_count = votes.count('Not Acceptable')
        
        # Majority vote
        if acceptable_count > not_acceptable_count:
            return 1  # Good
        elif not_acceptable_count > acceptable_count:
            return 0  # Bad
        else:
            return None  # Tie
    
    def __getitem__(self, idx):
        """
        Load one organoid sequence WITH MASKS
        
        Returns:
            images: Tensor of shape (11, 4, H, W) - 11 timepoints, RGBM (4 channels), Height, Width
            label: Tensor with single value (0 or 1)
        """
        organoid_id = self.organoid_ids[idx]
        
        # Get all entry keys for this organoid (in temporal order)
        entry_keys = self.series_metadata[organoid_id]['entry_keys']
        
        # Load all images and masks in the sequence
        images = []
        for key in entry_keys:
            entry = self.data[key]
            
            # Get image path
            img_path = entry['lstm_processed']['image_path']
            
            # Load image
            img = imread(img_path)
            
            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            
            if self.use_masks:
                # Load mask
                mask_path = entry['lstm_processed'].get('mask_path')
                if mask_path and Path(mask_path).exists():
                    mask = imread(mask_path)
                    
                    # Ensure mask is 2D
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]  # Take first channel if RGB
                    
                    # Add channel dimension
                    mask = mask[:, :, np.newaxis]  # (H, W, 1)
                else:
                    # No mask available, use zeros
                    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                
                # Concatenate RGB + Mask = 4 channels
                img = np.concatenate([img, mask], axis=-1)  # (H, W, 4)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Apply transforms if provided
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
        
        # Stack into sequence: (T, H, W, C) -> (T, C, H, W)
        sequence = np.stack(images)  # (11, H, W, 4) or (11, H, W, 3)
        sequence = np.transpose(sequence, (0, 3, 1, 2))  # (11, 4, H, W) or (11, 3, H, W)
        
        # Get label from final timepoint (Day 30)
        final_entry = self.data[entry_keys[-1]]
        label = self.get_label_from_survey(final_entry)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])[0]


def load_data_and_create_splits(series_metadata_path, data_path, 
                                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                random_seed=42):
    """
    Load data and split into train/val/test sets
    
    Returns:
        train_ids, val_ids, test_ids: Lists of organoid IDs
        series_metadata, data: Loaded JSON dicts
    """
    # Load JSONs
    with open(series_metadata_path) as f:
        series_metadata = json.load(f)
    
    with open(data_path) as f:
        data = json.load(f)
    
    # Get label function
    def get_label(organoid_id):
        entry_keys = series_metadata[organoid_id]['entry_keys']
        final_entry = data[entry_keys[-1]]
        
        if 'survey' not in final_entry or not final_entry['survey']:
            return None
        
        survey = final_entry['survey']
        if 'evaluations' not in survey or not survey['evaluations']:
            return None
        
        votes = [ev.get('evaluation') for ev in survey['evaluations']]
        acceptable_count = votes.count('Acceptable')
        not_acceptable_count = votes.count('Not Acceptable')
        
        if acceptable_count > not_acceptable_count:
            return 1
        elif not_acceptable_count > acceptable_count:
            return 0
        else:
            return None
    
    # Filter to only organoids with labels
    labeled_organoids = []
    labels = []
    
    for org_id in series_metadata.keys():
        label = get_label(org_id)
        if label is not None:
            labeled_organoids.append(org_id)
            labels.append(label)
    
    print(f"Total organoids with labels: {len(labeled_organoids)}")
    print(f"  Good: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Bad: {len(labels) - sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
    
    # Shuffle and split
    np.random.seed(random_seed)
    indices = np.random.permutation(len(labeled_organoids))
    
    n_train = int(len(labeled_organoids) * train_ratio)
    n_val = int(len(labeled_organoids) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    train_ids = [labeled_organoids[i] for i in train_indices]
    val_ids = [labeled_organoids[i] for i in val_indices]
    test_ids = [labeled_organoids[i] for i in test_indices]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_ids)} organoids")
    print(f"  Val:   {len(val_ids)} organoids")
    print(f"  Test:  {len(test_ids)} organoids")
    
    return train_ids, val_ids, test_ids, series_metadata, data