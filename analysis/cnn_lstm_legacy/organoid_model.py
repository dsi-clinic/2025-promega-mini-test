"""
CNN-LSTM model for organoid time series classification
Uses EfficientNet-B0 for feature extraction
"""
import torch
import torch.nn as nn
from torchvision import models

class TemporalAttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1)
        )
    def forward(self, feats):  # feats: (B, T, D)
        w = self.attn(feats).squeeze(-1)           # (B, T)
        a = torch.softmax(w, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (a * feats).sum(dim=1)            # (B, D)
        return pooled, a.squeeze(-1)               # return weights too for analysis

class OrganoidCNN_TAtt(nn.Module):
    def __init__(self, d_cnn=1280):
        super().__init__()
        eff = models.efficientnet_b0(pretrained=True)
        eff.classifier = nn.Identity()
        self.cnn = eff
        for p in self.cnn.parameters():  # start frozen
            p.requires_grad = False

        self.temporal = TemporalAttentionPool(d_cnn)
        self.head = nn.Sequential(
            nn.Linear(d_cnn, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)  # BCEWithLogits
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.cnn(x[:, t])       # (B, 1280)
            feats.append(f)
        feats = torch.stack(feats, dim=1)  # (B, T, 1280)
        pooled, attn = self.temporal(feats)
        logit = self.head(pooled).squeeze(1)
        return logit, attn
