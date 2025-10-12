"""
CNN-LSTM model for organoid time series classification
NOW SUPPORTS 4-CHANNEL INPUT (RGB + Mask)
"""
import torch
import torch.nn as nn
from torchvision import models

class OrganoidCNN_LSTM(nn.Module):
    """
    CNN-LSTM for organoid quality prediction with mask input
    
    Architecture:
    1. CNN (ResNet50 modified for 4 channels) extracts features from each timepoint
    2. LSTM processes the sequence of features over time
    3. Final linear layer outputs binary classification
    """
    
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=2, dropout=0.5, 
                 input_channels=4):
        """
        Args:
            num_classes: Number of output classes (2 for binary)
            lstm_hidden: Hidden size of LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            input_channels: 4 for RGB+Mask, 3 for RGB only
        """
        super(OrganoidCNN_LSTM, self).__init__()
        
        # CNN Feature Extractor - Use pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # MODIFICATION: Change first conv layer to accept 4 channels instead of 3
        if input_channels == 4:
            original_first_conv = resnet.conv1
            
            # Create new conv layer with 4 input channels
            resnet.conv1 = nn.Conv2d(
                in_channels=4,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            
            # Initialize weights: Copy pretrained RGB weights, initialize mask channel
            with torch.no_grad():
                # Copy RGB channel weights
                resnet.conv1.weight[:, :3, :, :] = original_first_conv.weight
                
                # Initialize mask channel as average of RGB channels
                resnet.conv1.weight[:, 3:, :, :] = original_first_conv.weight.mean(dim=1, keepdim=True)
        
        # Remove final classification layer - we just want features
        # ResNet50 outputs 2048 features
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early CNN layers (optional - speeds up training)
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
               e.g., (32, 11, 4, 768, 768) for RGB+Mask
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, num_timepoints, C, H, W = x.size()
        
        # Process each timepoint through CNN
        cnn_features = []
        for t in range(num_timepoints):
            # Get image at time t
            img_t = x[:, t, :, :, :]  # (batch_size, 4, 768, 768)
            
            # Extract features
            feat_t = self.cnn(img_t)  # (batch_size, 2048, 1, 1)
            feat_t = feat_t.view(batch_size, -1)  # (batch_size, 2048)
            
            cnn_features.append(feat_t)
        
        # Stack features into sequence
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, 11, 2048)
        
        # Process sequence through LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use final hidden state for classification
        final_hidden = h_n[-1]  # (batch_size, lstm_hidden)
        
        # Dropout and classification
        x = self.dropout(final_hidden)
        logits = self.fc(x)  # (batch_size, num_classes)
        
        return logits