"""
Temporal-Change EfficientNet: captures morphological trajectory, not just snapshots.

Key insight: good organoids change dramatically over time, bad ones stay static.
The original attention-pooling collapses all frames into one weighted average,
destroying the change signal. This model explicitly computes:
  1. Attention-pooled features          (what the organoid looks like overall)
  2. Feature variance over time         (how much it changed)
  3. Last-minus-first feature delta     (direction and magnitude of change)
  4. Mean consecutive-frame differences (average rate of change)

The classifier sees all four signals concatenated, so it can learn that
"high variance + large delta = good organoid" directly.
"""

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 100
WARMUP_EPOCHS = 1
LR_HEAD = 5e-4
LR_CNN_UNFREEZE = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 15
ATTN_DROPOUT = 0.4
SEED = 42

DAY_RANGES = [8, 10, 13, 15, 17, 20.5, 24, 30]


def set_seed(seed=SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class TemporalAttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, feats, mask=None):
        w = self.attn(feats).squeeze(-1)
        if mask is not None:
            w = w.masked_fill(~mask, float("-inf"))
        a = torch.softmax(w, dim=1).unsqueeze(-1)
        pooled = (a * feats).sum(dim=1)
        return pooled, a.squeeze(-1)


def _patch_effnet_first_conv_4ch(backbone):
    first_conv = backbone.features[0][0]
    if first_conv.in_channels == 4:
        return
    assert first_conv.in_channels == 3
    new_conv = nn.Conv2d(
        4,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = first_conv.weight
        new_conv.weight[:, 3] = 0.0
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)
    backbone.features[0][0] = new_conv


class OrganoidCNN_TChange(nn.Module):
    """
    EfficientNet + Temporal-Change pooling.

    Classifier input = concat of:
      - attention-pooled features   (d_cnn)     : what it looks like
      - feature variance over time  (d_cnn)     : how much it changed
      - last - first frame delta    (d_cnn)     : direction of change
      - mean consecutive deltas     (d_cnn)     : average rate of change

    Total classifier input: 4 * d_cnn = 5120
    """

    def __init__(self, d_cnn=1280, attn_dropout=0.4, in_channels=3):
        super().__init__()
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        eff.classifier = nn.Identity()
        if in_channels == 4:
            _patch_effnet_first_conv_4ch(eff)
        self.cnn = eff

        for p in self.cnn.parameters():
            p.requires_grad = False

        self.time_proj = nn.Sequential(
            nn.Linear(1, d_cnn // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(d_cnn // 2, d_cnn),
            nn.LayerNorm(d_cnn),
        )

        self.temporal = TemporalAttentionPool(d_cnn)

        # Projection to compress the 4 temporal signals before classification
        self.temporal_compress = nn.Sequential(
            nn.LayerNorm(d_cnn * 4),
            nn.Linear(d_cnn * 4, d_cnn),
            nn.ReLU(),
            nn.Dropout(attn_dropout * 0.5),
        )

        self.head = nn.Sequential(
            nn.Dropout(attn_dropout),
            nn.Linear(d_cnn, 128),
            nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(128, 1),
        )

    def unfreeze_last_blocks(self):
        for name, p in self.cnn.named_parameters():
            if "features.6" in name or "features.7" in name:
                p.requires_grad = True

    def forward(self, x, days_norm):
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.cnn(x[:, t])
            dt = days_norm[:, t].unsqueeze(1).to(f.device)
            f = f + self.time_proj(dt)
            feats.append(f)
        feats = torch.stack(feats, dim=1)  # (B, T, d_cnn)

        mask = (days_norm > 0).to(feats.device)  # (B, T)
        n_real = mask.sum(dim=1).clamp(min=1)  # (B,)

        # 1) Attention-pooled (what it looks like overall)
        pooled, attn = self.temporal(feats, mask=mask)  # (B, d_cnn)

        # 2) Feature variance over time (how much it changed)
        mask_3d = mask.unsqueeze(-1)  # (B, T, 1)
        feat_mean = (feats * mask_3d).sum(dim=1) / n_real.unsqueeze(1)  # (B, d_cnn)
        feat_var = ((feats - feat_mean.unsqueeze(1)) ** 2 * mask_3d).sum(
            dim=1
        ) / n_real.unsqueeze(1)  # (B, d_cnn)

        # 3) Last real frame minus first frame (overall trajectory direction)
        first_feat = feats[:, 0, :]  # (B, d_cnn) -- always real
        last_indices = (n_real.long() - 1).clamp(min=0)  # (B,)
        last_feat = feats[
            torch.arange(B, device=feats.device), last_indices
        ]  # (B, d_cnn)
        delta_first_last = last_feat - first_feat  # (B, d_cnn)

        # 4) Mean consecutive-frame differences (rate of change)
        diffs = feats[:, 1:, :] - feats[:, :-1, :]  # (B, T-1, d_cnn)
        diff_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)  # (B, T-1, 1)
        n_diffs = diff_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_diffs = (diffs.abs() * diff_mask).sum(dim=1) / n_diffs  # (B, d_cnn)

        # Concatenate all 4 signals
        combined = torch.cat(
            [pooled, feat_var, delta_first_last, mean_diffs], dim=-1
        )  # (B, 4*d_cnn)
        compressed = self.temporal_compress(combined)  # (B, d_cnn)
        logit = self.head(compressed).squeeze(1)  # (B,)
        return logit, attn


@torch.no_grad()
def evaluate_binary(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels, losses = [], [], []
    false_pos, false_neg = [], []

    for seqs, days, labels, weights, ids in loader:
        seqs = seqs.to(device)
        days = days.to(device).float()
        labels = labels.float().to(device)

        logits, _ = model(seqs, days)
        loss_raw = criterion(logits, labels)
        losses.append(loss_raw.mean().item())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu()
        labels_cpu = labels.int().cpu()

        for oid, pred, true in zip(ids, preds, labels_cpu):
            if pred == 1 and true == 0:
                false_pos.append(oid)
            elif pred == 0 and true == 1:
                false_neg.append(oid)

        all_probs.append(probs.cpu())
        all_labels.append(labels_cpu)

    if len(all_probs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false_pos, false_neg

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = (probs > 0.5).int()

    acc = (preds == labels.int()).float().mean().item()

    from sklearn.metrics import (
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
    )

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels.numpy(), preds.numpy(), average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(labels.numpy(), probs.numpy())
    except ValueError:
        ap = float("nan")

    return (
        float(np.mean(losses)),
        acc,
        float(prec),
        float(rec),
        float(f1),
        float(auc),
        float(ap),
        false_pos,
        false_neg,
    )
