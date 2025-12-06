"""
AccurateEEG - EEG-based Emotion Recognition using LSTM and BiLSTM Networks

Implementation of the paper:
"Accurate EEG-Based Emotion Recognition using LSTM and BiLSTM Networks"
(IEEE ICSIMA 2024)

Modules:
    - config: Configuration settings
    - feature_extraction: Feature extraction from EEG signals
    - data_loader: Custom data loader for GAMEEMO dataset
    - models: LSTM and BiLSTM model architectures
    - train: Training functions
    - evaluate: Evaluation functions
    - utils: Utility functions
"""

__version__ = '1.0.0'
__author__ = 'Based on Yaacob et al. (2024)'

__all__ = [
    'config',
    'feature_extraction',
    'data_loader',
    'models',
    'train',
    'evaluate',
    'utils'
]
