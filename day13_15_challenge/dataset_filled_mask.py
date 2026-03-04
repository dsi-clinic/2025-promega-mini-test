"""
Dataset variant: input = (R, G, B) = (grayscale, grayscale, filled_mask) for 3-channel.
Filled mask = binary mask (0/1), not outline. For Day 13/15 challenge (Task 4).
"""
from pathlib import Path
import sys
import numpy as np
from skimage.io import imread

CHALLENGE_DIR = Path(__file__).resolve().parent
ROOT = CHALLENGE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.images.cnn_lstm.organoid_dataset import OrganoidTimeSeriesDataset
from analysis.images.preprocessing.stitched_preprocessing import preprocess_stitched


class OrganoidTimeSeriesDatasetFilledMask(OrganoidTimeSeriesDataset):
    """
    Same as OrganoidTimeSeriesDataset but each frame is 3ch: (gray, gray, filled_mask)
    built from image_path + mask_path (no overlay). Uses ImageNet mean/std.
    """

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
            mask_path = lp.get("mask_path", "")

            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = preprocess_stitched(img, img_path)
            gray = img[:, :, 0].astype(np.float32) / 255.0

            if mask_path and Path(mask_path).exists():
                mask = imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                filled = (mask > 127).astype(np.float32)
            else:
                filled = np.zeros_like(gray, dtype=np.float32)

            # 3ch: (gray, gray, filled_mask)
            img_3ch = np.stack([gray, gray, filled], axis=-1)

            if self.transform:
                from PIL import Image
                img_pil = Image.fromarray((img_3ch[:, :, :3] * 255).astype(np.uint8))
                img_pil = self.transform(img_pil)
                img_3ch = np.array(img_pil).astype(np.float32) / 255.0

            images.append(img_3ch)
            days_used.append(days[i])

        sequence = np.stack(images)
        sequence = np.transpose(sequence, (0, 3, 1, 2))
        seq = torch.from_numpy(sequence).float()
        C = seq.shape[1]
        if C == 3:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        else:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406, 0.5], dtype=torch.float32).view(1, 4, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225, 0.5], dtype=torch.float32).view(1, 4, 1, 1)
        seq = (seq - imagenet_mean) / imagenet_std

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
            weight = 1.0 if (frac >= 0.9 or frac <= 0.1) else (0.8 if (frac >= 0.7 or frac <= 0.3) else 0.6)
        else:
            weight = 1.0

        return (
            seq,
            days_norm,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            organoid_id,
        )


class SingleDayFilledMaskDataset:
    """Single-day dataset returning (gray, gray, filled_mask) 3ch, ImageNet normalized. Same .samples interface as SingleDayOrganoidDataset."""

    def __init__(self, organoid_ids, series_metadata, data, target_day, transform=None):
        self.samples = []
        self.transform = transform
        for org_id in organoid_ids:
            metadata = series_metadata.get(org_id, {})
            label_str = str(metadata.get("label", "")).strip().lower()
            label = 1 if label_str in ("good", "acceptable", "accepted") else 0
            entry_keys = metadata.get("entry_keys", [])
            days = metadata.get("days", [])
            if not entry_keys or not days:
                continue
            best_idx = min(range(len(days)), key=lambda i: abs(days[i] - target_day))
            entry_key = entry_keys[best_idx]
            entry = data.get(entry_key, {})
            processed = entry.get("processed", {})
            img_path = processed.get("img_path")
            mask_path = processed.get("mask_path", "")
            if img_path is None or not Path(img_path).exists():
                continue
            if not mask_path or not Path(mask_path).exists():
                continue
            self.samples.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "label": label,
                "org_id": org_id,
            })
        print(f"  Loaded {len(self.samples)} samples for day ~{target_day} (filled_mask 3ch)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        from PIL import Image
        sample = self.samples[idx]
        img = imread(sample["img_path"])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = preprocess_stitched(img, sample["img_path"])
        gray = img[:, :, 0].astype(np.float32) / 255.0
        mask = imread(sample["mask_path"])
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        filled = (mask > 127).astype(np.float32)
        img_3ch = np.stack([gray, gray, filled], axis=-1)
        if self.transform:
            img_pil = Image.fromarray((img_3ch * 255).astype(np.uint8))
            img_3ch = np.array(self.transform(img_pil)).astype(np.float32) / 255.0
        img_3ch = np.transpose(img_3ch, (2, 0, 1))
        img_t = torch.from_numpy(img_3ch).float()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img_t = (img_t - mean) / std
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return img_t, label, sample["org_id"]
