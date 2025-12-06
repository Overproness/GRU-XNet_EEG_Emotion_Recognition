# gru_xnet: CNN-BiGRU-Self Attention for EEG Emotion Recognition

Implementation of the gru_xnet architecture for multi-dataset EEG emotion recognition, modified to use BiGRU instead of BiLSTM.

## Trained Checkpoint Link:

Google Drive: [gru_xnet Trained Model](https://drive.google.com/file/d/1em6OdJllEMgycVeKM01s0ckQxPDpU_7Y/view?usp=sharing)

## Overview

gru_xnet combines:

- **STFT preprocessing**: Time-frequency transformation of EEG signals
- **Channel-independent CNNs**: Separate CNN for each EEG channel
- **BiGRU**: Bidirectional GRU for temporal modeling (replacing BiLSTM from original paper)
- **Multi-head Self-Attention**: Dynamic feature weighting
- **Multi-dataset training**: Combined training on DEAP, GAMEEMO, and SEEDIV datasets

## Features

**Multi-Dataset Support**: DEAP (32ch), GAMEEMO (14ch), SEEDIV (62ch)  
**Data Augmentation**: 1:2 ratio of original to augmented data  
**STFT Preprocessing**: Adaptive time-frequency transformation  
**Mixed Precision Training**: Faster training with AMP  
**Early Stopping**: Prevent overfitting  
**LOSO Cross-Validation**: Leave-One-Subject-Out support

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

## Quick Start

### 1. Basic Training

```bash
# Full training
python train.py --config full

# Quick test (reduced data/epochs)
python train.py --config quick
```

### 2. Python API

```python
from config import get_full_training_config
from train import Trainer

# Create configuration
config = get_full_training_config()

# Customize if needed
config.training.batch_size = 64
config.training.num_epochs = 50

# Train
trainer = Trainer(config)
trainer.train()
test_results = trainer.test()

print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
```

### 3. Using Individual Components

```python
import numpy as np
import torch
from model import create_gru_xnet_model
from preprocessing import STFTPreprocessor

# Create STFT preprocessor
stft_processor = STFTPreprocessor(
    sampling_rate=128,
    nperseg=256,
    freq_range=(0.5, 50.0)
)

# Transform EEG data
eeg_data = np.random.randn(32, 8064)  # (channels, timepoints)
stft_features = stft_processor.transform(eeg_data)
print(f"STFT shape: {stft_features.shape}")  # (32, n_freq, n_time)

# Create model
model = create_gru_xnet_model(
    n_channels=32,
    n_freq_bins=stft_features.shape[1],
    n_time_bins=stft_features.shape[2],
    n_classes=2,
    model_type='dynamic'
)

# Forward pass
x = torch.from_numpy(stft_features).unsqueeze(0).float()
output = model(x)
print(f"Output shape: {output.shape}")  # (1, 2)
```

## Configuration

Edit `config.py` or create custom configs:

```python
from config import ExperimentConfig, ModelConfig, TrainingConfig

config = ExperimentConfig(
    experiment_name="my_experiment"
)

# Model settings
config.model.gru_hidden_size = 256
config.model.num_attention_heads = 8
config.model.dropout = 0.3

# Training settings
config.training.learning_rate = 0.0005
config.training.batch_size = 256
config.training.num_epochs = 100

# Data settings
config.data.datasets = ['DEAP', 'GAMEEMO']  # Use only 2 datasets
config.data.balance_classes = True
config.data.cache_stft = True  # Cache STFT for faster loading
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

## File Structure

```
gru_xnet/
├── __init__.py              # Package initialization
├── model.py                 # gru_xnet model architecture
├── preprocessing.py         # STFT preprocessing
├── config.py                # Configuration classes
├── data_loader.py          # Data loading pipeline
├── train.py                 # Training script
├── utils.py                 # Utility functions
├── README.md                # This file
└── requirements.txt         # Dependencies
```

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

## Training Tips

1. **Start with quick test**: Use `--config quick` to verify everything works
2. **Enable caching**: Set `cache_stft=True` to cache STFT features
3. **Monitor GPU memory**: Reduce batch size if OOM errors occur
4. **Use mixed precision**: Enabled by default with `use_amp=True`
5. **Adjust learning rate**: Try 0.0005 or 0.0001 if training is unstable

## Outputs

Training produces:

```
outputs/gru_xnet/
├── checkpoints/
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── figures/
│   └── training_curves.png
├── logs/
├── config.json
├── training_history.json
└── test_results.json
```

## Performance Monitoring

During training:

- Loss and accuracy printed every epoch
- Best model saved automatically
- Training curves plotted at end
- Test set evaluation with confusion matrix

## Troubleshooting

### Memory Issues

```python
config.training.batch_size = 32  # Reduce batch size
config.data.num_workers = 0      # Reduce workers
config.training.use_amp = True   # Enable mixed precision
```

### Slow Training

```python
config.data.cache_stft = True    # Cache STFT features
config.data.num_workers = 4      # Increase workers
config.training.use_amp = True   # Use mixed precision
```

### Poor Performance

```python
config.training.learning_rate = 0.0005  # Lower LR
config.model.dropout = 0.3              # Reduce dropout
config.training.num_epochs = 50         # Train longer
config.data.balance_classes = True      # Balance classes
```

## Citation

If you use this implementation, please cite the original gru_xnet paper and acknowledge the modifications:

```bibtex
@article{gru_xnet,
  title={gru_xnet: Channel-Balanced Self-Attention for Emotion Recognition},
  author={Original Authors},
  journal={Journal Name},
  year={2023}
}
```

**Modifications**: BiGRU instead of BiLSTM, multi-dataset support, PyTorch implementation

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact the team.
