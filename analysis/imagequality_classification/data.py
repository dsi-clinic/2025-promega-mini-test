#!/usr/bin/env python3
"""Data loading + filtering for the image-quality classifier.

ImagePathDataset wraps a list of image paths + labels (and optional mask
paths) into a torch Dataset. ImageClassifierEmitter builds a per-day view
of records from all_data.json suitable for training.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from pipeline.common.json_views import BaseViewEmitter
from pipeline.merge.normalized_records import OrganoidRecord

SCHEMA_DICT = Dict[str, Any]


class ImageClassifierEmitter(BaseViewEmitter):
    """Build view payload for the image classifier training script."""

    name = "image_classifier"

    def __init__(self):
        self._records_by_day: Dict[str, List[SCHEMA_DICT]] = defaultdict(list)
        self._skipped_records_by_day: Dict[str, List[str]] = defaultdict(list)

    def process(self, record: OrganoidRecord) -> None:
        label = record.get("label", {}).get("acceptance_flag")
        img_path = record.get("images", {}).get("img_path")
        mask_path = record.get("images", {}).get("mask_path")
        overlay_path = record.get("images", {}).get("overlay_path")

        if (
            label not in self.label_list
            or not img_path
            or not mask_path
            or not record.get("day", {}).get("id")
        ):
            self._skipped_records_by_day[record.get("day", {}).get("id")].append(record.get("id"))
            return

        self._records_by_day[record.get("day", {}).get("id")].append({
            "id": record.get("id"),
            "img_path": img_path,
            "label": label,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
        })


class ImagePathDataset(Dataset):
    """torch Dataset wrapping a list of image paths + labels.

    Optionally returns a mask tensor alongside the image. Renamed from
    OrganoidDataset to avoid clashing with pipeline.data_loader.OrganoidDataset
    (a different concept).
    """

    def __init__(self, img_paths, labels, target_size, mask_paths=None,
                 augment=False, use_mask=False, normalize=False):
        self.img_paths = img_paths
        self.labels = labels
        self.mask_paths = mask_paths
        self.augment = augment
        self.use_mask = use_mask
        t = [T.Resize(target_size)]
        if augment:
            t += [T.RandomHorizontalFlip(0.5), T.ColorJitter(0.2, 0.2, 0.2, 0.1)]
        t += [T.ToTensor()]
        if normalize:
            t += [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.t_img = T.Compose(t)
        if self.use_mask:
            if self.mask_paths is None:
                raise ValueError("mask_paths must be provided when use_mask=True")
            self.t_mask = T.Compose([
                T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.t_img(Image.open(self.img_paths[idx]).convert("RGB"))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.use_mask:
            mask = self.t_mask(Image.open(self.mask_paths[idx]).convert("L"))
            return img, mask, label
        return img, label


def load_image_classifier_views(all_data_json: Path):
    """Load image classifier views from all_data.json by replaying the emitter."""
    import json
    with open(all_data_json) as f:
        records = json.load(f)
    emitter = ImageClassifierEmitter()
    for record in records.values():
        emitter.process(record)
    return emitter.finalize()


def filter_missing_files(day, labels, imgs, masks, backbone_key, cfg):
    """Drop entries whose image (or mask, if use_mask) files don't exist on disk.

    Records the dropped entries to a CSV under cfg.out_dir/<backbone_key>/.
    Returns (filtered_imgs, filtered_labels, filtered_masks_or_None) or None
    if nothing remains.
    """
    filtered_imgs, filtered_labels = [], []
    filtered_masks = [] if cfg.use_mask else None
    missing = []
    for idx, img_path in enumerate(imgs):
        img_path = Path(str(img_path))
        mask_path = Path(str(masks[idx])) if (cfg.use_mask and masks is not None) else None
        if not img_path.exists():
            missing.append({"img_path": str(img_path), "mask_path": str(mask_path or ""), "reason": "missing_image"})
            continue
        if cfg.use_mask and (mask_path is None or not mask_path.exists()):
            missing.append({"img_path": str(img_path), "mask_path": str(mask_path or ""), "reason": "missing_mask"})
            continue
        filtered_imgs.append(str(img_path))
        filtered_labels.append(labels[idx])
        if cfg.use_mask:
            filtered_masks.append(str(mask_path))

    if missing:
        _write_missing_csv(day, cfg, backbone_key, missing)

    if not filtered_labels:
        print(f"⚠ Skipping {day} — no valid samples after filtering missing files")
        return None

    return filtered_imgs, filtered_labels, filtered_masks


def _write_missing_csv(day, cfg, backbone_key, missing_records):
    log_dir = cfg.out_dir / backbone_key
    log_dir.mkdir(parents=True, exist_ok=True)
    missing_csv = log_dir / "missing_files.csv"
    fieldnames = ["img_path", "mask_path", "reason"]

    rows = []
    if missing_csv.exists():
        with missing_csv.open("r", newline="") as fh:
            rows.extend(csv.DictReader(fh))
    rows.extend(missing_records)

    with missing_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"⚠ {day}: skipped {len(missing_records)} entries due to missing files (details → {missing_csv})")


def make_loader(imgs, labels, augment, batch_size, cfg, mask_paths=None):
    ds = ImagePathDataset(
        imgs, labels, target_size=cfg.target_size,
        mask_paths=mask_paths, augment=augment, use_mask=cfg.use_mask,
    )
    if cfg.deterministic:
        gen = torch.Generator()
        gen.manual_seed(cfg.seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=cfg.num_workers, generator=gen)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=cfg.num_workers)
