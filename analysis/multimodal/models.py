#!/usr/bin/env python3
"""Multimodal classifier: image backbone + optional mask + optional metabolite branches."""

import numpy as np
import timm
import torch
import torch.nn as nn

from .data import META_DIM

BACKBONE_MODELS = {
    "vit": "vit_base_patch16_224",
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0",
}


class MaskBranch(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(32 * 16, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.encoder(x)


class MetaboliteBranch(nn.Module):
    """MLP for metabolite vectors. Expects fixed-width input (META_DIM).

    Padding for day-conditional features (Malate=0 for days ≤10) is the
    dataset's responsibility — see MultimodalRowDataset.
    """

    def __init__(self, input_dim: int = META_DIM, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class MultimodalClassifier(nn.Module):
    """Image + (optional) mask + (optional) metabolite classifier with configurable fusion."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.use_mask = "mask" in config["input_mode"]
        self.use_metabolites = config["use_metabolites"]
        self.fusion_strategy = config.get("fusion_strategy", "concat")

        backbone_name = BACKBONE_MODELS[config["backbone"]]
        extra = {"img_size": config["target_size"]} if "vit" in backbone_name else {}
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, **extra)

        img_dim = self.backbone.num_features
        if self.use_mask:
            self.mask_branch = MaskBranch(64)
            img_dim += self.mask_branch.out_dim
        else:
            self.mask_branch = None

        if self.use_metabolites:
            self.meta_branch = MetaboliteBranch(input_dim=META_DIM, hidden_dim=64)
            meta_dim = self.meta_branch.out_dim
            if self.fusion_strategy == "gated":
                self.gate = nn.Sequential(nn.Linear(meta_dim, img_dim), nn.Sigmoid())
                fused_dim = img_dim
            else:
                fused_dim = img_dim + meta_dim
        else:
            fused_dim = img_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, *args):
        if self.config["use_images"] and self.use_metabolites:
            if self.use_mask:
                img, mask, meta = args[:3]
            else:
                img, meta = args[:2]
        elif self.config["use_images"]:
            if self.use_mask:
                img, mask = args[:2]
            else:
                img = args[0]
        elif self.use_metabolites:
            meta = args[0]
        else:
            raise ValueError("Must use either images or metabolites")

        if self.config["use_images"]:
            img_feats = self.backbone(img)
            if self.use_mask:
                img_feats = torch.cat([img_feats, self.mask_branch(mask)], 1)
        if self.use_metabolites:
            meta_feats = self.meta_branch(meta)

        if self.config["use_images"] and self.use_metabolites:
            if self.fusion_strategy == "gated":
                fused = img_feats * self.gate(meta_feats)
            else:
                fused = torch.cat([img_feats, meta_feats], 1)
        elif self.config["use_images"]:
            fused = img_feats
        else:
            fused = meta_feats

        return self.head(fused).squeeze(1)


class EarlyStopping:
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best = -np.inf
        self.counter = 0

    def __call__(self, score):
        if score > self.best + 1e-4:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
