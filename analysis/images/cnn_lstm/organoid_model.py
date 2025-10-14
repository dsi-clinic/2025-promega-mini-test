"""
CNN-LSTM model for organoid time series classification
STANDARD 3-CHANNEL INPUT (RGB)
"""
import torch
import torch.nn as nn
from torchvision import models

class OrganoidCNN_LSTM(nn.Module):
    """
    CNN-LSTM for organoid quality prediction
    
    Architecture:
    1. CNN (ResNet50) extracts features from each timepoint independently
    2. LSTM processes the sequence of features over time
    3. Final linear layer outputs binary classification
    """
    
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=2, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes (2 for binary)
            lstm_hidden: Hidden size of LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(OrganoidCNN_LSTM, self).__init__()
        
        # CNN Feature Extractor - Use pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove final classification layer - we just want features
        # ResNet50 outputs 2048 features
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early CNN layers (optional - speeds up training)
        # We'll fine-tune only later layers
        for param in list(self.cnn.parameters())[:-20]:
            param.requires_grad = False
        
        cnn_feature_size = 2048
        
        # LSTM to process temporal sequence
        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Final classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_timepoints, channels, height, width)
               e.g., (32, 11, 3, 768, 768)
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, num_timepoints, C, H, W = x.size()
        
        # Process each timepoint through CNN
        cnn_features = []
        for t in range(num_timepoints):
            # Get image at time t
            img_t = x[:, t, :, :, :]  # (batch_size, 3, 768, 768)
            
            # Extract features
            feat_t = self.cnn(img_t)  # (batch_size, 2048, 1, 1)
            feat_t = feat_t.view(batch_size, -1)  # (batch_size, 2048)
            
            cnn_features.append(feat_t)
        
        # Stack features into sequence
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, 11, 2048)
        
        # Process sequence through LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use final hidden state for prediction
        final_hidden = h_n[-1]  # (batch_size, lstm_hidden)
        
        # Dropout and classification
        x = self.dropout(final_hidden)
        logits = self.fc(x)  # (batch_size, num_classes)
        
        return logits