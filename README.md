# GRU-XNet: CNN-BiGRU-Self Attention for EEG Emotion Recognition

Deep learning architecture for multi-dataset EEG-based emotion recognition using bidirectional GRU networks with self-attention mechanisms.

## Authors

**Muhammad Wasif Shakeel** - [GitHub](https://github.com/mwasifshkeel)  
**Muhammad Muntazar** - [GitHub](https://github.com/overproness)

National University of Sciences and Technology (NUST), Pakistan  
Deep Learning Course Project - Fall 2025

## Paper

The complete research paper detailing methodology, experiments, and results is available in the [paper/](paper/) directory:
- [DL-Report.pdf](paper/DL-Report.pdf) - Full research paper

## Pre-trained Model

Google Drive: [Download Trained Model](https://drive.google.com/file/d/1em6OdJllEMgycVeKM01s0ckQxPDpU_7Y/view?usp=sharing)

## Overview

GRU-XNet is a hybrid architecture combining convolutional neural networks, bidirectional gated recurrent units, and multi-head self-attention for EEG emotion recognition. The model processes time-frequency representations of EEG signals through channel-independent CNNs, captures temporal dependencies with BiGRU layers, and applies attention mechanisms for enhanced feature extraction.

### Key Components

- **STFT Preprocessing**: Time-frequency transformation of raw EEG signals
- **Channel-Independent CNNs**: Separate convolutional pathways for each EEG electrode
- **Bidirectional GRU**: Temporal sequence modeling with forward and backward context
- **Multi-Head Self-Attention**: Weighted feature aggregation across time steps
- **Multi-Dataset Training**: Unified training across DEAP, GAMEEMO, and SEEDIV datasets

## Features

- Multi-dataset support: DEAP (32 channels), GAMEEMO (14 channels), SEEDIV (62 channels)
- Comprehensive data augmentation pipeline with 1:2 original-to-augmented ratio
- Short-Time Fourier Transform (STFT) preprocessing with adaptive parameters
- Mixed precision training with automatic gradient scaling
- Early stopping with configurable patience
- Leave-One-Subject-Out (LOSO) cross-validation support
- Automatic checkpointing and training resume capability

## Architecture

```
Input: Raw EEG (n_channels, n_timepoints)
   ↓
STFT Transform → (n_channels, n_freq_bins, n_time_bins)
   ↓
Channel-Independent CNNs (one per channel)
   ↓
Feature Fusion
   ↓
BiGRU (bidirectional, 2 layers)
   ↓
Multi-Head Self-Attention (4 heads)
   ↓
Classification Head
   ↓
Output: Class logits
```

### 1. Installation and Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --config full

# Reduced training for testing
python train.py --config quick
```

### 2. Python API

```python
from config import get_full_training_config
from train import Trainer

config = get_full_training_config()
config.training.batch_size = 64
config.training.num_epochs = 50

trainer = Trainer(config)
trainer.train()
test_results = trainer.test()

print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
```

### 3. Using Components

```python
import numpy as np
import torch
from model import create_gru_xnet_model
from preprocessing import STFTPreprocessor

stft_processor = STFTPreprocessor(
    sampling_rate=128,
    nperseg=256,
    freq_range=(0.5, 50.0)
)

eeg_data = np.random.randn(32, 8064)
stft_features = stft_processor.transform(eeg_data)

model = create_gru_xnet_model(
    n_channels=32,
    n_freq_bins=stft_features.shape[1],
    n_time_bins=stft_features.shape[2],
    n_classes=2,
    model_type='dynamic'
)

x = torch.from_numpy(stft_features).unsqueeze(0).float()
output = model(x)
```

## Configuration

Configuration is managed through dataclasses in [config.py](config.py):

```python
from config import ExperimentConfig

config = ExperimentConfig(experiment_name="my_experiment")

# Model settings
config.model.gru_hidden_size = 256
config.model.num_attention_heads = 8
config.model.dropout = 0.3

# Training settings
config.training.learning_rate = 0.0005
config.training.batch_size = 256
config.training.num_epochs = 100

# Data settings
config.data.datasets = ['DEAP', 'GAMEEMO']
config.data.balance_classes = True
config.data.cache_stft = True
```

## Key Configuration Options

### Model Configuration

- `model_type`: 'standard' or 'dynamic' (dynamic preserves temporal structure)
- `gru_hidden_size`: Hidden size for BiGRU (default: 128)
- `num_attention_heads`: Number of attention heads (default: 4)
- `dropout`: Dropout rate (default: 0.5)

### Training Configuration

- `learning_rate`: Adam learning rate (default: 0.001)
- `batch_size`: Batch size (default: 128)
- `num_epochs`: Number of epochs (default: 30)
- `use_scheduler`: Enable cosine annealing LR schedule
- `early_stopping`: Enable early stopping (patience: 10)
- `use_amp`: Enable mixed precision training

### Data Configuration

- `datasets`: List of datasets to use ['DEAP', 'GAMEEMO', 'SEEDIV']
- `use_augmented`: Use augmented data (default: True)
- `augmentation_ratio`: Ratio of augmented to original (default: 2.0)
- `balance_classes`: Balance class distribution in training
- `cache_stft`: Cache STFT features for faster loading
- `train_ratio/val_ratio/test_ratio`: Data split ratios

## Repository Structure

```
GRU-XNet_EEG_Emotion_Recognition/
├── model.py                    # GRU-XNet architecture implementation
├── preprocessing.py            # STFT transformation and normalization
├── config.py                   # Experiment configuration dataclasses
├── data_loader.py              # PyTorch dataset and dataloader implementations
├── train.py                    # Training pipeline with checkpointing
├── utils.py                    # Helper functions and metrics tracking
├── visualize_model.py          # Model performance visualization tools
├── visualize_architecture.py   # Architecture diagram generation
├── gru_xnet_Training_Kaggle.ipynb  # Kaggle training notebook
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── augmentation_pipeline/      # Data augmentation framework
│   ├── base_augmentation.py       # Base augmentation interface
│   ├── beginner_augmentations.py  # Basic augmentation techniques
│   ├── intermediate_augmentations.py  # Advanced augmentation methods
│   ├── advanced_augmentations.py  # Complex augmentations (VAE, GAN)
│   ├── augmentation_pipeline.py   # Pipeline orchestration
│   ├── augment_all_datasets.py    # Batch augmentation script
│   ├── augmentation_config.py     # Augmentation parameters
│   ├── dataset_loaders.py         # Dataset loading utilities
│   ├── quality_validation.py      # Augmentation quality checks
│   └── requirements.txt
│
├── LitReview/                  # Literature review implementations
│   ├── carnn.ipynb                # CA-RNN baseline
│   ├── cbsatt.ipynb               # CBS-Attention baseline
│   ├── EffectiveConnectivity.ipynb  # Connectivity analysis
│   └── AccurateEEG/               # Baseline LSTM implementations
│       ├── models.py                 # BiLSTM/LSTM architectures
│       ├── train.py                  # Baseline training
│       ├── data_loader.py            # Data preparation
│       ├── feature_extraction.py     # Feature engineering
│       └── outputs/                  # Baseline results
│
├── outputs/                    # Training outputs (generated)
│   ├── checkpoints/               # Model checkpoints
│   ├── figures/                   # Training visualizations
│   ├── config.json                # Saved configuration
│   ├── training_history.json      # Epoch-wise metrics
│   ├── test_results.json          # Final evaluation results
│   └── classification_report.txt  # Detailed classification metrics
│
└── paper/                      # Research paper
    └── DL-Report.pdf              # Complete project report
```

### Directory Details

**Root Directory**: Core model implementation, training, and evaluation scripts.

**augmentation_pipeline/**: Modular data augmentation framework with multiple augmentation strategies including time-domain transformations, frequency-domain modifications, and noise injection techniques.

**LitReview/**: Baseline implementations from existing literature for comparison, including CNN-RNN hybrids, attention-based models, and traditional LSTM approaches.

**outputs/**: Auto-generated directory containing model checkpoints, training history, evaluation metrics, and visualization plots.

**paper/**: Research documentation including methodology, experimental setup, results analysis, and conclusions.

## Model Details

### Channel-Independent CNN

- 3 convolutional blocks
- 3×3 kernels, 2×2 max pooling
- BatchNorm + ReLU activation
- Channels: 32 → 64 → 128

### BiGRU

- 2-layer bidirectional GRU
- Hidden size: 128 (256 total with bidirectional)
- Dropout between layers

### Multi-Head Self-Attention

- 4 attention heads
- Scaled dot-product attention
- Residual connections + Layer normalization

### Classification Head

- 3 fully connected layers
- Dimensions: (gru_hidden\*2) → 256 → 128 → n_classes
- ReLU activation + Dropout

## Multi-Dataset Handling

The implementation handles datasets with different characteristics:

| Dataset | Channels | Sampling Rate | Time Length | Classes           |
| ------- | -------- | ------------- | ----------- | ----------------- |
| DEAP    | 32       | 128 Hz        | 8064        | 2 (binary)        |
| GAMEEMO | 14       | 128 Hz        | 640         | 2 (binary)        |
| SEEDIV  | 62       | 200 Hz        | 28000       | 4 → 2 (converted) |

**Key Features**:

- Dataset-specific STFT parameters
- Standardized output dimensions
- Balanced sampling across datasets
- Subject-wise splitting

## Training Recommendations

1. **Initial Testing**: Use `--config quick` to verify setup before full training runs
2. **STFT Caching**: Enable `cache_stft=True` in configuration to cache preprocessed features for faster subsequent epochs
3. **GPU Memory Management**: If encountering out-of-memory errors, reduce batch size proportionally
4. **Mixed Precision**: Automatic mixed precisto cache preprocessed features
3. **GPU Memory**: Reduce batch size if encountering out-of-memory errors
4. **Mixed Precision**: AMP is enabled by default for faster training
5. **Learning Rate**: Default is 0.001; reduce to 0.0005 or 0.0001 if training is unstable

## Output Structure

```
outputs/gru_xnet/
├── checkpoints/
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── figures/
│   └── training_curves.png
├── config.json
├── training_history.json
└── test_results.json
```

### Memory Issues

```python
config.training.batch_size = 32  # Reduce batch size
config.data.num_workers = 0      # Reduce workers
config.training.use_amp = True   # Enable mixed precision
```

### Slow Training

```python
config.data.cache_stft = True    # Cache STFT features
**Memory Issues**:
```python
config.training.batch_size = 32
config.data.num_workers = 0
config.training.use_amp = True
```

**Slow Training**:
```python
config.data.cache_stft = True
config.data.num_workers = 4
config.training.use_amp = True
```

**Poor Performance**:
```python
config.training.learning_rate = 0.0005
config.model.dropout = 0.3
config.training.num_epochs = 50
config.data.balance_classes = True
  type={Deep Learning Course Project}
}
```

## Acknowledgments

Datasets used in this research:
- DEAP: Database for Emotion Analysis using Physiological signals
- GAMEEMO: Game-based EEG emotion dataset
- SEED-IV: SJTU Emotion EEG Dataset (4 classes)

This work builds upon various deep learning techniques for physiological signal processing and emotion recognition.

## License

[MIT License](LICENSE)