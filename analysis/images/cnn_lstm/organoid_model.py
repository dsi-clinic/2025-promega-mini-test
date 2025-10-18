"""
CNN-LSTM model for organoid time series classification
Uses EfficientNet-B0 for feature extraction
"""
import torch
import torch.nn as nn
from torchvision import models


class OrganoidCNN_LSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=2):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Remove the final classification layer
        efficientnet.classifier = nn.Identity()
        self.cnn = efficientnet
        
        # EfficientNet-B0 outputs 1280 features
        cnn_out_features = 1280
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, time, channels, height, width)
        batch_size, timesteps, C, H, W = x.size()
        
        # Process each frame through CNN
        cnn_out = []
        for t in range(timesteps):
            features = self.cnn(x[:, t, :, :, :])
            
            # Flatten if needed (EfficientNet returns (batch, 1280))
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            cnn_out.append(features)
        
        # Stack into sequence
        cnn_out = torch.stack(cnn_out, dim=1)  # (batch, time, 1280)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        
        # Classification
        output = self.classifier(final_hidden)
        
        return output