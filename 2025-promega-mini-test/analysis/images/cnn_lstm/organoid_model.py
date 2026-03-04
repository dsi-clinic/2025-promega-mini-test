"""
CNN-LSTM model for organoid time series classification
Uses EfficientNet-B0 for feature extraction
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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


def _patch_effnet_first_conv_4ch(backbone):
    """Replace first conv 3->32 with 4->32; copy pretrained for first 3 ch, zero-init 4th."""
    first_conv = backbone.features[0][0]
    if first_conv.in_channels == 4:
        return
    assert first_conv.in_channels == 3
    new_conv = nn.Conv2d(4, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias is not None)
    with torch.no_grad():
        new_conv.weight[:, :3] = first_conv.weight
        new_conv.weight[:, 3] = 0.0
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)
    backbone.features[0][0] = new_conv


class OrganoidCNN_LSTM(nn.Module):
    """
    CNN-LSTM for organoid quality prediction using EfficientNet-B0.
    in_channels=3 (RGB) or 4 (RGB+mask).
    """
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=2, in_channels=3):
        super().__init__()
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        eff.classifier = nn.Identity()
        if in_channels == 4:
            _patch_effnet_first_conv_4ch(eff)
        self.cnn = eff
        for p in self.cnn.parameters():
            p.requires_grad = False
        
        cnn_feature_size = 1280  # EfficientNet-B0 output
        
        # LSTM to process temporal sequence
        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.5 if lstm_layers > 1 else 0
        )
        
        # Final classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, num_timepoints, channels, height, width)
        
        Returns:
            logits: Tensor (batch_size, num_classes)
        """
        batch_size, num_timepoints, C, H, W = x.size()
        
        # Process each timepoint through CNN
        cnn_features = []
        for t in range(num_timepoints):
            img_t = x[:, t, :, :, :]  # (batch_size, 3, H, W)
            feat_t = self.cnn(img_t)  # (batch_size, 1280)
            feat_t = feat_t.view(batch_size, -1)  # Flatten
            cnn_features.append(feat_t)
        
        # Stack features into sequence
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, T, 1280)
        
        # Process sequence through LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use final hidden state for classification
        final_hidden = h_n[-1]  # (batch_size, lstm_hidden)
        
        # Dropout and classification
        x = self.dropout(final_hidden)
        logits = self.fc(x)  # (batch_size, num_classes)
        
        return logits
