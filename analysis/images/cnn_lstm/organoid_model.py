"""
CNN-LSTM model for organoid time series classification
Uses EfficientNet-B0 for feature extraction
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


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


class OrganoidCNN_LSTM(nn.Module):
    def __init__(self, d_cnn=1280, hidden_size=256, num_layers=1, bidirectional=False):
        super().__init__()
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        eff.classifier = nn.Identity()
        self.cnn = eff
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.time_proj = nn.Sequential(
            nn.Linear(1, d_cnn // 2),
            nn.ReLU(),
            nn.Linear(d_cnn // 2, d_cnn),
        )

        self.lstm = nn.LSTM(
            input_size=d_cnn,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.5),
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def unfreeze_last_blocks(self, n_blocks: int = 2):
        """Unfreeze last EfficientNet feature blocks for fine-tuning."""
        feats = getattr(self.cnn, "features", None)
        if feats is None:
            return
        start = max(0, len(feats) - n_blocks)
        for i in range(start, len(feats)):
            for p in feats[i].parameters():
                p.requires_grad = True

    def forward(self, x, days_norm):  # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.cnn(x[:, t])
            dt = days_norm[:, t].unsqueeze(1).to(f.device)
            f = f + self.time_proj(dt)
            feats.append(f)
        feats = torch.stack(feats, dim=1)

        lstm_out, _ = self.lstm(feats)
        last_hidden = lstm_out[:, -1, :]
        logit = self.head(last_hidden).squeeze(1)
        return logit
