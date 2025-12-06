"""
LSTM and BiLSTM Models for EEG-based Emotion Recognition
Implements architectures from the paper for binary and multiclass classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLSTMClassifier(nn.Module):
    """
    LSTM classifier for binary emotion classification
    Architecture: Input -> LSTM(128) -> Flatten -> BatchNorm -> Dropout -> Dense(2)
    """
    
    def __init__(self, input_size, hidden_size=128, dropout=0.5):
        """
        Initialize Binary LSTM classifier
        
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units (default: 128)
            dropout: Dropout rate (default: 0.5)
        """
        super(BinaryLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,  # Process features sequentially
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 2)
        """
        # Reshape input to (batch_size, seq_len, 1)
        x = x.unsqueeze(2)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state
        out = h_n[-1]  # Shape: (batch_size, hidden_size)
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Output layer
        out = self.fc(out)
        
        return out


class BinaryBiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for binary emotion classification
    Architecture: Input -> BiLSTM(128) -> Flatten -> BatchNorm -> Dropout -> Dense(2)
    """
    
    def __init__(self, input_size, hidden_size=128, dropout=0.5):
        """
        Initialize Binary BiLSTM classifier
        
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units (default: 128)
            dropout: Dropout rate (default: 0.5)
        """
        super(BinaryBiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Batch normalization (2*hidden_size because of bidirectional)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 2)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 2)
        """
        # Reshape input to (batch_size, seq_len, 1)
        x = x.unsqueeze(2)
        
        # BiLSTM layer
        lstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Concatenate forward and backward hidden states
        # h_n has shape (2, batch_size, hidden_size)
        out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch_size, hidden_size*2)
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Output layer
        out = self.fc(out)
        
        return out


class MulticlassLSTMClassifier(nn.Module):
    """
    LSTM classifier for multiclass emotion classification
    Architecture: Input -> LSTM(128) -> LSTM(64) -> Flatten -> BatchNorm -> Dropout -> Dense(4)
    """
    
    def __init__(self, input_size, hidden_size_1=128, hidden_size_2=64, dropout=0.5):
        """
        Initialize Multiclass LSTM classifier
        
        Args:
            input_size: Number of input features
            hidden_size_1: Number of hidden units in first LSTM layer (default: 128)
            hidden_size_2: Number of hidden units in second LSTM layer (default: 64)
            dropout: Dropout rate (default: 0.5)
        """
        super(MulticlassLSTMClassifier, self).__init__()
        
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.input_size = input_size
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size_2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size_2, 4)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 4)
        """
        # Reshape input to (batch_size, seq_len, 1)
        x = x.unsqueeze(2)
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        
        # Second LSTM layer
        lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)
        
        # Take the last hidden state
        out = h_n[-1]  # Shape: (batch_size, hidden_size_2)
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Output layer
        out = self.fc(out)
        
        return out


class MulticlassBiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for multiclass emotion classification
    Architecture: Input -> BiLSTM(128) -> BiLSTM(64) -> Flatten -> BatchNorm -> Dropout -> Dense(4)
    """
    
    def __init__(self, input_size, hidden_size_1=128, hidden_size_2=64, dropout=0.5):
        """
        Initialize Multiclass BiLSTM classifier
        
        Args:
            input_size: Number of input features
            hidden_size_1: Number of hidden units in first BiLSTM layer (default: 128)
            hidden_size_2: Number of hidden units in second BiLSTM layer (default: 64)
            dropout: Dropout rate (default: 0.5)
        """
        super(MulticlassBiLSTMClassifier, self).__init__()
        
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.input_size = input_size
        
        # First BiLSTM layer
        self.bilstm1 = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Second BiLSTM layer
        self.bilstm2 = nn.LSTM(
            input_size=hidden_size_1 * 2,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size_2 * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size_2 * 2, 4)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 4)
        """
        # Reshape input to (batch_size, seq_len, 1)
        x = x.unsqueeze(2)
        
        # First BiLSTM layer
        bilstm1_out, _ = self.bilstm1(x)
        
        # Second BiLSTM layer
        bilstm2_out, (h_n, c_n) = self.bilstm2(bilstm1_out)
        
        # Concatenate forward and backward hidden states
        out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch_size, hidden_size_2*2)
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Output layer
        out = self.fc(out)
        
        return out


def create_model(model_type, input_size, device='cpu'):
    """
    Factory function to create models
    
    Args:
        model_type: One of ['binary_lstm', 'binary_bilstm', 'multiclass_lstm', 'multiclass_bilstm']
        input_size: Number of input features
        device: Device to place the model on
        
    Returns:
        Model instance
    """
    models = {
        'binary_lstm': BinaryLSTMClassifier(input_size=input_size),
        'binary_bilstm': BinaryBiLSTMClassifier(input_size=input_size),
        'multiclass_lstm': MulticlassLSTMClassifier(input_size=input_size),
        'multiclass_bilstm': MulticlassBiLSTMClassifier(input_size=input_size)
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = models[model_type]
    model = model.to(device)
    
    return model
