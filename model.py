"""
gru_xnet: CNN-BiGRU-Self Attention Network for EEG Emotion Recognition
Modified from original paper to use BiGRU instead of BiLSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class ChannelIndependentCNN(nn.Module):
    """
    Channel-independent CNN module
    Each EEG channel has its own independent CNN for spatial feature extraction
    """
    
    def __init__(self, n_freq_bins: int):
        """
        Args:
            n_freq_bins: Number of frequency bins from STFT
        """
        super().__init__()
        
        # Independent CNN for each channel
        # Input: (batch, 1, n_freq_bins, n_time_bins)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, n_freq_bins, n_time_bins)
            
        Returns:
            features: (batch, 128, freq_reduced, time_reduced)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.dropout(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_k)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output


class gru_xnet(nn.Module):
    """
    gru_xnet: CNN-BiGRU-Self Attention Network for EEG Emotion Recognition
    
    Architecture:
    1. STFT transformation (done in preprocessing)
    2. Channel-independent CNNs for spatial feature extraction
    3. BiGRU for temporal modeling
    4. Multi-head self-attention for feature enhancement
    5. Classification head
    """
    
    def __init__(
        self,
        n_channels: int,
        n_freq_bins: int,
        n_time_bins: int,
        n_classes: int,
        gru_hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.5
    ):
        """
        Args:
            n_channels: Number of EEG channels
            n_freq_bins: Number of frequency bins from STFT
            n_time_bins: Number of time bins from STFT
            n_classes: Number of output classes
            gru_hidden_size: Hidden size for BiGRU
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_freq_bins = n_freq_bins
        self.n_time_bins = n_time_bins
        self.n_classes = n_classes
        self.gru_hidden_size = gru_hidden_size
        
        # Channel-independent CNNs (one per channel)
        self.channel_cnns = nn.ModuleList([
            ChannelIndependentCNN(n_freq_bins) for _ in range(n_channels)
        ])
        
        # Calculate feature dimensions after CNN
        # After 3 pooling layers with stride 2: divide by 8
        self.freq_reduced = n_freq_bins // 8
        self.time_reduced = n_time_bins // 8
        
        # Flatten CNN output for each channel
        self.cnn_output_dim = 128 * self.freq_reduced * self.time_reduced
        
        # Feature fusion: concatenate all channel features
        self.feature_fusion = nn.Linear(n_channels * self.cnn_output_dim, gru_hidden_size * 2)
        
        # BiGRU for temporal modeling
        self.bigru = nn.GRU(
            input_size=gru_hidden_size * 2,
            hidden_size=gru_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(
            d_model=gru_hidden_size * 2,  # *2 for bidirectional
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x_stft):
        """
        Args:
            x_stft: STFT features (batch, n_channels, n_freq_bins, n_time_bins)
            
        Returns:
            output: Class logits (batch, n_classes)
        """
        batch_size = x_stft.shape[0]
        
        # Apply channel-independent CNNs
        channel_features = []
        for i in range(self.n_channels):
            # Extract single channel: (batch, 1, n_freq_bins, n_time_bins)
            channel_input = x_stft[:, i:i+1, :, :]
            
            # Apply CNN
            channel_feat = self.channel_cnns[i](channel_input)  # (batch, 128, freq_reduced, time_reduced)
            
            # Flatten
            channel_feat = channel_feat.view(batch_size, -1)  # (batch, cnn_output_dim)
            channel_features.append(channel_feat)
        
        # Concatenate all channel features
        concat_features = torch.cat(channel_features, dim=1)  # (batch, n_channels * cnn_output_dim)
        
        # Feature fusion
        fused_features = self.feature_fusion(concat_features)  # (batch, gru_hidden_size * 2)
        
        # Reshape for BiGRU: treat as sequence of length 1 initially
        # Then we'll expand it to create a proper sequence
        gru_input = fused_features.unsqueeze(1)  # (batch, 1, gru_hidden_size * 2)
        
        # For better temporal modeling, we can repeat the features to create a sequence
        # This simulates temporal evolution
        seq_len = 10  # Sequence length for GRU
        gru_input = gru_input.repeat(1, seq_len, 1)  # (batch, seq_len, gru_hidden_size * 2)
        
        # BiGRU
        gru_output, _ = self.bigru(gru_input)  # (batch, seq_len, gru_hidden_size * 2)
        
        # Multi-head self-attention
        attn_output = self.attention(gru_output)  # (batch, seq_len, gru_hidden_size * 2)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(attn_output, dim=1)  # (batch, gru_hidden_size * 2)
        
        # Classification
        output = self.classifier(pooled_output)  # (batch, n_classes)
        
        return output


class gru_xnetDynamic(nn.Module):
    """
    Dynamic version of gru_xnet that handles variable-length sequences
    Uses actual temporal dimension from CNN output
    """
    
    def __init__(
        self,
        n_channels: int,
        n_freq_bins: int,
        n_time_bins: int,
        n_classes: int,
        gru_hidden_size: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.5
    ):
        """
        Args:
            n_channels: Number of EEG channels
            n_freq_bins: Number of frequency bins from STFT
            n_time_bins: Number of time bins from STFT
            n_classes: Number of output classes
            gru_hidden_size: Hidden size for BiGRU
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_freq_bins = n_freq_bins
        self.n_time_bins = n_time_bins
        self.n_classes = n_classes
        self.gru_hidden_size = gru_hidden_size
        
        # Channel-independent CNNs
        self.channel_cnns = nn.ModuleList([
            ChannelIndependentCNN(n_freq_bins) for _ in range(n_channels)
        ])
        
        # Calculate dimensions after CNN pooling
        self.freq_reduced = n_freq_bins // 8
        self.time_reduced = n_time_bins // 8
        
        # Feature dimension per time step per channel
        self.feature_dim_per_channel = 128 * self.freq_reduced
        
        # BiGRU for temporal modeling
        # Input: features from all channels at each time step
        self.bigru = nn.GRU(
            input_size=n_channels * self.feature_dim_per_channel,
            hidden_size=gru_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(
            d_model=gru_hidden_size * 2,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x_stft):
        """
        Args:
            x_stft: STFT features (batch, n_channels, n_freq_bins, n_time_bins)
            
        Returns:
            output: Class logits (batch, n_classes)
        """
        batch_size = x_stft.shape[0]
        actual_channels = x_stft.shape[1]  # Get actual number of channels in input
        
        # Apply channel-independent CNNs and preserve temporal structure
        channel_features = []
        for i in range(min(self.n_channels, actual_channels)):  # Only process actual channels
            channel_input = x_stft[:, i:i+1, :, :]
            channel_feat = self.channel_cnns[i](channel_input)  # (batch, 128, freq_reduced, time_reduced)
            
            # Reshape to (batch, time_reduced, 128 * freq_reduced)
            channel_feat = channel_feat.permute(0, 3, 1, 2)  # (batch, time_reduced, 128, freq_reduced)
            channel_feat = channel_feat.reshape(batch_size, self.time_reduced, -1)
            channel_features.append(channel_feat)
        
        # If we have fewer channels than expected, pad with zeros
        if actual_channels < self.n_channels:
            # Create zero padding for missing channels
            feature_dim_per_channel = 128 * self.freq_reduced
            padding_features = torch.zeros(
                batch_size, self.time_reduced, 
                (self.n_channels - actual_channels) * feature_dim_per_channel,
                device=x_stft.device, dtype=x_stft.dtype
            )
            channel_features.append(padding_features)
        
        # Concatenate features from all channels along feature dimension
        # (batch, time_reduced, n_channels * feature_dim_per_channel)
        temporal_features = torch.cat(channel_features, dim=2)
        
        # BiGRU
        gru_output, _ = self.bigru(temporal_features)  # (batch, time_reduced, gru_hidden_size * 2)
        
        # Multi-head self-attention
        attn_output = self.attention(gru_output)  # (batch, time_reduced, gru_hidden_size * 2)
        
        # Global average pooling over temporal dimension
        pooled_output = torch.mean(attn_output, dim=1)  # (batch, gru_hidden_size * 2)
        
        # Classification
        output = self.classifier(pooled_output)  # (batch, n_classes)
        
        return output


def create_gru_xnet_model(
    n_channels: int,
    n_freq_bins: int,
    n_time_bins: int,
    n_classes: int,
    model_type: str = 'dynamic',
    **kwargs
) -> nn.Module:
    """
    Factory function to create gru_xnet model
    
    Args:
        n_channels: Number of EEG channels
        n_freq_bins: Number of frequency bins from STFT
        n_time_bins: Number of time bins from STFT  
        n_classes: Number of output classes
        model_type: 'standard' or 'dynamic'
        **kwargs: Additional arguments for model
        
    Returns:
        gru_xnet model
    """
    if model_type == 'dynamic':
        return gru_xnetDynamic(
            n_channels=n_channels,
            n_freq_bins=n_freq_bins,
            n_time_bins=n_time_bins,
            n_classes=n_classes,
            **kwargs
        )
    else:
        return gru_xnet(
            n_channels=n_channels,
            n_freq_bins=n_freq_bins,
            n_time_bins=n_time_bins,
            n_classes=n_classes,
            **kwargs
        )


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    n_channels = 32
    n_freq_bins = 129  # Typical for STFT
    n_time_bins = 126
    n_classes = 2
    
    # Create model
    model = create_gru_xnet_model(
        n_channels=n_channels,
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
        n_classes=n_classes,
        model_type='dynamic',
        gru_hidden_size=128,
        num_attention_heads=4,
        dropout=0.5
    )
    
    # Test forward pass
    x = torch.randn(batch_size, n_channels, n_freq_bins, n_time_bins)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
