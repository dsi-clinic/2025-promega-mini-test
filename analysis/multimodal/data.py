#!/usr/bin/env python3
"""Multimodal dataset + data loading.

MultimodalRowDataset wraps a DataFrame of (organoid, day) rows. Metabolite
zero-padding (for the day-conditional MalateGlo column) happens here at
construction time — the model branches see fixed-width vectors.

load_and_prepare_data builds train/val/test DataFrames from all_data.json
via pipeline.data_loader.OrganoidDataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from pipeline.data_loader import (
    LABEL_TO_INT,
    filters_for_mode,
    get_day_int_floor,
)
from pipeline.data_loader import (
    OrganoidDataset as PipelineOrganoidDataset,
)
from pipeline.splits import Splits

# Metabolite features — match the legacy multimodal trainer + paper.
# NEVER read *_initial_concentration here.
BASE_MET_FEATURES = [
    "GlucoseGlo_concentration_uM",
    "GlutamateGlo_concentration_uM",
    "LactateGlo_concentration_uM",
    "PyruvateGlo_concentration_uM",
]
MALATE_FEATURE = "MalateGlo_concentration_uM"  # included on all days (0 if missing)
META_DIM = len(BASE_MET_FEATURES) + 1  # 5: base + Malate, padded to this width


def day_to_int(day_str: str) -> int:
    n = get_day_int_floor(day_str)
    return -1 if n is None else n


def get_transforms(config: dict, augment: bool = False):
    t: list = [T.Resize(config["target_size"])]
    if augment and config["use_augmentation"]:
        t.extend([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)])
    t.extend([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    return T.Compose(t)


class MultimodalRowDataset(Dataset):
    """Torch Dataset wrapping a DataFrame of (organoid, day) rows.

    Renamed from OrganoidDataset to avoid clashing with
    pipeline.data_loader.OrganoidDataset (a different concept). Pads
    metabolite vectors to META_DIM at construction time so model branches
    expect fixed-width inputs.
    """

    def __init__(self, df, config, transform=None,
                 scaler: StandardScaler | None = None, fit_scaler: bool = False):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.label_map = LABEL_TO_INT
        self.img_key = "overlay_path" if "overlay" in config["input_mode"] else "img_path"
        self.use_mask = "mask" in config["input_mode"]
        self.use_metabolites = config["use_metabolites"]

        valid = []
        for i in range(len(self.df)):
            if config["use_images"]:
                img = self.df.iloc[i][self.img_key]
                if pd.isna(img) or not Path(img).exists():
                    continue
                if self.use_mask:
                    mask = self.df.iloc[i]["mask_path"]
                    if pd.isna(mask) or not Path(mask).exists():
                        continue
            valid.append(i)
        self.df = self.df.iloc[valid].reset_index(drop=True)

        if self.use_metabolites:
            features = self._extract_metabolite_features_padded()
            if fit_scaler:
                self.scaler = StandardScaler()
                self.meta_features = self.scaler.fit_transform(features)
            elif scaler is not None:
                self.scaler = scaler
                self.meta_features = self.scaler.transform(features)
            else:
                self.scaler = None
                self.meta_features = features.astype(np.float32)
            self.meta_features = self.meta_features.astype(np.float32)
        else:
            self.scaler = scaler
            self.meta_features = None

    def _extract_metabolite_features_padded(self) -> np.ndarray:
        """Return (n, META_DIM) array. Malate is included on all days (0 if missing)."""
        rows = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            feat = []
            for col in BASE_MET_FEATURES:
                val = row.get(col, np.nan)
                feat.append(0.0 if pd.isna(val) else float(val))
            malate = row.get(MALATE_FEATURE, np.nan)
            feat.append(0.0 if pd.isna(malate) else float(malate))
            rows.append(feat)
        return np.array(rows, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(self.label_map.get(row["label"], 0), dtype=torch.float32)

        items = []
        if self.config["use_images"]:
            img = Image.open(row[self.img_key]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            items.append(img)
            if self.use_mask:
                mask = Image.open(row["mask_path"]).convert("L")
                mask = T.Compose([T.Resize(self.config["target_size"]), T.ToTensor()])(mask)
                items.append(mask)

        if self.use_metabolites:
            items.append(torch.tensor(self.meta_features[idx], dtype=torch.float32))

        items.append(label)
        return tuple(items)


def load_and_prepare_data(config: dict):
    """Build train/val/test DataFrames from all_data.json via OrganoidDataset.

    One row per (organoid, day). Reads the normalized schema directly:
    images.* paths, metabolite.<name>.concentration_uM, label.value.
    """
    ds = PipelineOrganoidDataset(
        config["all_data_path"],
        splits=Splits.from_csv(config["splits_csv"]),
        filters=filters_for_mode(config.get("mode", "base"), modality="both"),
    )

    metabolite_cols = BASE_MET_FEATURES + [MALATE_FEATURE]
    metabolite_keys = [c.replace("_concentration_uM", "") for c in metabolite_cols]

    def split_to_df(split: str) -> pd.DataFrame:
        rows = []
        for org_id, info in ds.get_split(split).items():
            label_str = info["label"]
            for day, rec in info["records"].items():
                imgs = rec.get("images", {}) or {}
                mets = rec.get("metabolite", {}) or {}
                row = {
                    "org_id": org_id,
                    "label": label_str,
                    "day": day,
                    "day_num": day_to_int(day),
                    "img_path": imgs.get("img_path"),
                    "mask_path": imgs.get("mask_path"),
                    "overlay_path": imgs.get("overlay_path"),
                }
                for col, key in zip(metabolite_cols, metabolite_keys):
                    val = (mets.get(key) or {}).get("concentration_uM")
                    row[col] = np.nan if val is None else float(val)
                rows.append(row)
        return pd.DataFrame(rows)

    return split_to_df("train"), split_to_df("val"), split_to_df("test")
