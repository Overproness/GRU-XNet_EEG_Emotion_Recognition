"""
gru_xnet: CNN-BiGRU-Self Attention Network for EEG Emotion Recognition
Modified from original paper to use BiGRU instead of BiLSTM

This package implements the gru_xnet architecture for multi-dataset EEG emotion recognition.
"""

from .model import (
    gru_xnet,
    gru_xnetDynamic,
    create_gru_xnet_model,
    ChannelIndependentCNN,
    MultiHeadSelfAttention
)

from .preprocessing import (
    STFTPreprocessor,
    MultiDatasetSTFTPreprocessor,
    EEGNormalizer,
    create_dataset_stft_configs
)

from .config import (
    ExperimentConfig,
    STFTConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    get_full_training_config,
    get_quick_test_config,
    get_loso_config
)

from .data_loader import (
    gru_xnetDataset,
    load_combined_dataset,
    create_data_loaders
)

from .train import Trainer

from .utils import (
    set_seed,
    get_device,
    EarlyStopping,
    MetricTracker,
    save_checkpoint,
    load_checkpoint,
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report
)

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
