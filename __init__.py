"""
gru_xnet: CNN-BiGRU-Self Attention Network for EEG Emotion Recognition
Modified from original paper to use BiGRU instead of BiLSTM

This package implements the gru_xnet architecture for multi-dataset EEG emotion recognition.
"""

# Keep the historical exports lazy: importing the repository must not execute
# the archived trainer or require its missing parent-directory data pipeline.
# The maintained standalone API is the gruxnet package.
from importlib import import_module

_EXPORTS = {
    "model": ("gru_xnet", "gru_xnetDynamic", "create_gru_xnet_model", "ChannelIndependentCNN", "MultiHeadSelfAttention"),
    "preprocessing": ("STFTPreprocessor", "MultiDatasetSTFTPreprocessor", "EEGNormalizer", "create_dataset_stft_configs"),
    "config": ("ExperimentConfig", "STFTConfig", "ModelConfig", "TrainingConfig", "DataConfig", "get_full_training_config", "get_quick_test_config", "get_loso_config"),
    "data_loader": ("gru_xnetDataset", "load_combined_dataset", "create_data_loaders"),
    "train": ("Trainer",),
    "utils": ("set_seed", "get_device", "EarlyStopping", "MetricTracker", "save_checkpoint", "load_checkpoint", "plot_training_history", "plot_confusion_matrix", "print_classification_report"),
}


def __getattr__(name):
    for module, exports in _EXPORTS.items():
        if name in exports:
            prefix = f"{__package__}." if __package__ else ""
            value = getattr(import_module(prefix + module), name)
            globals()[name] = value
            return value
    raise AttributeError(name)

__version__ = '1.0.0'
__author__ = 'gru_xnet Team'

__all__ = [
    # Model
    'gru_xnet',
    'gru_xnetDynamic',
    'create_gru_xnet_model',
    'ChannelIndependentCNN',
    'MultiHeadSelfAttention',
    
    # Preprocessing
    'STFTPreprocessor',
    'MultiDatasetSTFTPreprocessor',
    'EEGNormalizer',
    'create_dataset_stft_configs',
    
    # Configuration
    'ExperimentConfig',
    'STFTConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'get_full_training_config',
    'get_quick_test_config',
    'get_loso_config',
    
    # Data loading
    'gru_xnetDataset',
    'load_combined_dataset',
    'create_data_loaders',
    
    # Training
    'Trainer',
    
    # Utils
    'set_seed',
    'get_device',
    'EarlyStopping',
    'MetricTracker',
    'save_checkpoint',
    'load_checkpoint',
    'plot_training_history',
    'plot_confusion_matrix',
    'print_classification_report',
]
