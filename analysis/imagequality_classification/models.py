#!/usr/bin/env python3
"""Model classes and training-loop helpers for the image quality classifier.

Includes EarlyStopping, the small CNN backbone, the optional mask branch,
and the top-level ImageOnlyClassifier head. The DEVICE constant and the
BACKBONES registry live here too since they're closely tied to model
construction.
"""

import os
import re

# Deterministic CUDA workspace (must be set before torch import)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BACKBONES = {
    "vit": "vit_base_patch16_224",   # img_size set to (H, W) at create_model time
    "resnet": "resnet50",
    "cnn": "cnn",                    # internal SmallCNNBackbone, not a timm name
}

BACKBONES_DINOV2 = {
    "dinov2": "facebook/dinov2-base",
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0",
}


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.bad = 0

    def step(self, score):
        if score > self.best + self.min_delta:
            self.best = score
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


class SmallCNNBackbone(nn.Module):
    """Simple CNN feature extractor used when backbone_key == 'cnn'."""

    def __init__(self, out_dim=256, deterministic=False):
        super().__init__()
        # Input size after 3 conv layers with stride=2: (384/8, 512/8) = (48, 64)
        # AdaptiveAvgPool2d isn't deterministic; AvgPool2d with explicit kernel is.
        avg_pool = (
            nn.AdaptiveAvgPool2d((4, 4))
            if not deterministic
            else nn.AvgPool2d(kernel_size=(12, 16), stride=(12, 16))
        )
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            avg_pool,
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.proj(self.features(x))


class MaskBranch(nn.Module):
    """Compact branch encoding binary masks into a feature vector."""

    def __init__(self, out_dim=64, deterministic=False):
        super().__init__()
        avg_pool = (
            nn.AdaptiveAvgPool2d((4, 4))
            if not deterministic
            else nn.AvgPool2d(kernel_size=(12, 16), stride=(12, 16))
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            avg_pool,
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, mask):
        return self.encoder(mask)


class ImageOnlyClassifier(nn.Module):
    """Backbone + optional mask branch + small MLP head.

    Supports three backbone families:
      * cnn      — internal SmallCNNBackbone
      * dinov2   — HuggingFace AutoModel (CLS token from last_hidden_state)
      * timm     — anything else (vit_*, resnet*, efficientnet_*)

    Phase-1 training freezes the backbone; phase-2 calls unfreeze_backbone()
    to release the deeper transformer/conv blocks.
    """

    def __init__(self, backbone_key, backbone_name, target_size, use_mask=False, deterministic=False):
        super().__init__()
        self.use_mask = use_mask
        self.backbone_key = backbone_key
        self._is_dinov2 = backbone_key == "dinov2"
        self._is_cnn = backbone_key == "cnn"
        self._is_timm = not (self._is_dinov2 or self._is_cnn)

        if self._is_cnn:
            self.backbone = SmallCNNBackbone(deterministic=deterministic)
            out_dim = self.backbone.out_dim
        elif self._is_dinov2:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(backbone_name)
            out_dim = self.backbone.config.hidden_size
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            extra_args = {}
            if "vit" in backbone_name:
                extra_args["img_size"] = target_size  # (H, W)
            self.backbone = timm.create_model(
                backbone_name, pretrained=True, num_classes=0,
                global_pool="avg", **extra_args,
            )
            out_dim = self.backbone.num_features
            for p in self.backbone.parameters():
                p.requires_grad = False  # frozen for phase 1

        if self.use_mask:
            self.mask_branch = MaskBranch(out_dim=64, deterministic=deterministic)
            head_in = out_dim + self.mask_branch.out_dim
        else:
            self.mask_branch = None
            head_in = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def unfreeze_backbone(self):
        if self._is_cnn:
            return
        if self._is_dinov2:
            for name, p in self.backbone.named_parameters():
                m = re.search(r"layer\.(\d+)", name)
                if m and int(m.group(1)) >= 8:
                    p.requires_grad = True
            return
        for name, p in self.backbone.named_parameters():
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img, mask=None):
        if self._is_dinov2:
            f = self.backbone(img).last_hidden_state[:, 0, :]
        else:
            f = self.backbone(img)
        if self.use_mask:
            if mask is None:
                raise ValueError("mask tensor must be provided when use_mask=True")
            f = torch.cat([f, self.mask_branch(mask)], dim=1)
        return self.classifier(f).squeeze(1)
