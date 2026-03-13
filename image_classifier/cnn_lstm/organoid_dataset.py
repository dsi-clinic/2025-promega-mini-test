"""
Dataset class for loading organoid time series data WITH MEAN-FILL MASK
Background is replaced with the mean intensity of the image (keeps brightness consistent).
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import torch
from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import cv2
from image_classifier.preprocessing.stitched_preprocessing import preprocess_stitched


def compute_global_mean(series_metadata, data):
    """Compute mean RGB across entire dataset"""
    print("Computing global dataset mean...")
    all_means = []

    for org_id in series_metadata.keys():
        entry_keys = series_metadata[org_id]["entry_keys"]
        for key in entry_keys:
            img_path = data[key]["lstm_processed"]["image_path"]
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Get mean of foreground pixels only (use mask if available)
            mask_path = data[key]["lstm_processed"].get("mask_path")
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
        entry_keys = series_metadata[org_id]["entry_keys"]
        for key in entry_keys:
            img_path = data[key]["lstm_processed"]["image_path"]
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


def _draw_outline_overlay_bgr(img_bgr, mask_bin, color=(0, 255, 0), thickness=2):
    """Draw mask contour on image (BGR)."""
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img_bgr.copy()
    if contours:
        cv2.drawContours(
            out,
            contours,
            contourIdx=-1,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return out


class OrganoidTimeSeriesDataset(Dataset):
    """
    Loads organoid image sequences. image_key: 'image_path'|'overlay_path'.
    input_rgb_mask: if True, each frame is 4-channel (RGB + mask).
    """

    def __init__(
        self,
        organoid_ids,
        series_metadata,
        data,
        transform=None,
        use_clipping_mask=False,
        global_mean=None,
        max_day=None,
        image_key="image_path",
        input_rgb_mask=False,
    ):
        self.organoid_ids = organoid_ids
        self.series_metadata = series_metadata
        self.data = data
        self.transform = transform
        self.use_clipping_mask = use_clipping_mask
        self.global_mean = global_mean
        self.max_day = max_day
        self.image_key = image_key
        self.input_rgb_mask = input_rgb_mask

    def __len__(self):
        return len(self.organoid_ids)

    def get_label_from_survey(self, entry):
        """Extract binary label using majority vote from evaluators."""
        if "survey" not in entry or not entry["survey"]:
            return None

        survey = entry["survey"]
        if "evaluations" not in survey or not survey["evaluations"]:
            return None

        votes = [ev.get("evaluation") for ev in survey["evaluations"]]
        acceptable = votes.count("Acceptable")
        not_acceptable = votes.count("Not Acceptable")

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
        entry_keys = self.series_metadata[organoid_id]["entry_keys"]
        days = self.series_metadata[organoid_id]["days"]
        # --- build frames + keep matched days ---
        images = []
        days_used = []  # <— track the days that produced frames

        for i, key in enumerate(entry_keys):
            # stop at max_day if set
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
            imagenet_mean = torch.tensor(
                [0.485, 0.456, 0.406], dtype=torch.float32
            ).view(1, 3, 1, 1)
            imagenet_std = torch.tensor(
                [0.229, 0.224, 0.225], dtype=torch.float32
            ).view(1, 3, 1, 1)
        else:
            imagenet_mean = torch.tensor(
                [0.485, 0.456, 0.406, 0.5], dtype=torch.float32
            ).view(1, 4, 1, 1)
            imagenet_std = torch.tensor(
                [0.229, 0.224, 0.225, 0.5], dtype=torch.float32
            ).view(1, 4, 1, 1)
        seq = (seq - imagenet_mean) / imagenet_std

        # label from metadata
        meta_entry = self.series_metadata[organoid_id]
        s = str(meta_entry.get("label", "")).strip().lower()
        label = 1 if s in ("good", "acceptable", "accepted") else 0

        # absolute day normalization (fixed scale; max day is 30)
        days_arr = np.array(days_used, dtype=np.float32)
        days_norm = days_arr / 30.0
        days_norm = torch.from_numpy(days_norm).float()  # (T,)

        # ----- agreement-based weight (0.6/0.8/1.0 as example) -----
        n_good = meta_entry.get("n_votes_good", None)
        n_total = meta_entry.get("n_votes_total", None)
        if n_good is not None and n_total and n_total > 0:
            frac = n_good / n_total
            if frac >= 0.9 or frac <= 0.1:
                weight = 1.0  # unanimous
            elif frac >= 0.7 or frac <= 0.3:
                weight = 0.8  # strong majority
            else:
                weight = 0.6  # weak majority / tie
        else:
            weight = 1.0

        return (
            seq,
            days_norm,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            organoid_id,  # keep the raw string id for FP/FN logs
        )


from collections import defaultdict
import numpy as np
import json


def load_data_and_create_splits(
    series_metadata_path,
    data_path,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    min_good_votes=4,  # keep only confident labels
    min_total_votes=5,
):
    with open(series_metadata_path) as f:
        series_metadata = json.load(f)
    with open(data_path) as f:
        data = json.load(f)

    # ---- filter to only nosplit organoids ----
    nosplit_series_metadata = {}
    for oid, meta in series_metadata.items():
        # Only keep those explicitly marked as nosplit or missing split_genealogy
        sg = str(meta.get("split_genealogy", "")).lower()
        if sg == "nosplit" or sg == "" or sg == "none":
            nosplit_series_metadata[oid] = meta

    print(
        f"Filtered to {len(nosplit_series_metadata)} nosplit organoids out of {len(series_metadata)} total"
    )
    series_metadata = nosplit_series_metadata

    # ---- 1) keep only confidently labeled organoids (optional but recommended) ----
    def hard_label(oid):
        meta = series_metadata[oid]
        s = str(meta.get("label", "")).strip().lower()
        # prefer vote counts if present
        ng = int(meta.get("n_votes_good", -1))
        nt = int(meta.get("n_votes_total", -1))
        if nt >= min_total_votes and ng >= 0:
            if ng >= min_good_votes:
                return 1
            if (nt - ng) >= min_good_votes:
                return 0
        # fallback to coarse label
        if s in ("good", "acceptable", "accepted"):
            return 1
        if s in ("bad", "not acceptable", "rejected", "not_good"):
            return 0
        return None

    labeled = []
    labels = {}
    for oid in series_metadata.keys():
        y = hard_label(oid)
        if y is not None:
            labeled.append(oid)
            labels[oid] = y

    # ---- 2) group organoids by base_well_id (parent + all splits) ----
    groups = defaultdict(list)
    for oid in labeled:
        gid = series_metadata[oid].get("base_well_id", oid)
        groups[gid].append(oid)

    # ---- 3) give each group a "group label" for stratification (majority of its members) ----
    group_ids = list(groups.keys())
    group_labels = []
    for gid in group_ids:
        ys = [labels[oid] for oid in groups[gid] if oid in labels]
        # if mixed, use majority; if tie, just pick 0 to be conservative
        if len(ys) == 0:
            group_labels.append(None)
        else:
            maj = int(np.round(np.mean(ys)))  # >=0.5 → 1
            group_labels.append(maj)

    # keep only groups that have at least one labeled member
    kept = [(g, y) for g, y in zip(group_ids, group_labels) if y is not None]
    group_ids, group_labels = zip(*kept) if kept else ([], [])

    # ---- 4) stratified shuffle split on groups ----
    rng = np.random.RandomState(random_seed)
    idx = np.arange(len(group_ids))
    # stratify by label: shuffle positives and negatives separately, then interleave
    pos_idx = idx[np.array(group_labels) == 1]
    neg_idx = idx[np.array(group_labels) == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    mixed = np.concatenate([pos_idx, neg_idx])
    # simple shuffle that approximately preserves class balance
    rng.shuffle(mixed)

    n = len(mixed)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_g = [group_ids[i] for i in mixed[:n_train]]
    val_g = [group_ids[i] for i in mixed[n_train : n_train + n_val]]
    test_g = [group_ids[i] for i in mixed[n_train + n_val :]]

    # ---- 5) expand group → organoid ids for each split ----
    train_ids = [oid for g in train_g for oid in groups[g]]
    val_ids = [oid for g in val_g for oid in groups[g]]
    test_ids = [oid for g in test_g for oid in groups[g]]

    # sanity: no leakage
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    # quick stats
    def split_stats(ids):
        ys = [labels[i] for i in ids if i in labels]
        if len(ys) == 0:
            return (0, 0, 0.0)
        return (sum(ys), len(ys) - sum(ys), sum(ys) / len(ys))

    gtr, btr, prtr = split_stats(train_ids)
    gva, bva, prva = split_stats(val_ids)
    gte, bte, prte = split_stats(test_ids)
    print(
        f"Groups: {len(group_ids)}  | Train groups {len(train_g)}, Val {len(val_g)}, Test {len(test_g)}"
    )
    print(f"Train: {len(train_ids)} ids (good {gtr}, bad {btr}, pos {prtr:.2f})")
    print(f"Val:   {len(val_ids)} ids (good {gva}, bad {bva}, pos {prva:.2f})")
    print(f"Test:  {len(test_ids)} ids (good {gte}, bad {bte}, pos {prte:.2f})")

    return train_ids, val_ids, test_ids, series_metadata, data
