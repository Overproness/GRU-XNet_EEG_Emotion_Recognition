# GRU-XNet: CNN-BiGRU-Self Attention for EEG Emotion Recognition

Deep learning architecture for multi-dataset EEG-based emotion recognition using bidirectional GRU networks with self-attention mechanisms.

## Publication revision (6 October 2026)

**Completed 9 October 2026:** all 4,080 training trajectories, 2,040 selected neural cases and 340 context cells in the [fold-local tuning protocol](docs/publication/Heldout_Tuning_Protocol_2026-10-06.md) are complete and verified. [Full findings](results/development/heldout_tuning_2026-10-06/FINDINGS.md) and [completion/recovery record](docs/publication/GRU-XNet_Heldout_Tuning_Status_2026-10-06.md) retain every grouping, initialization and contrast. On familiar DEAP videos, EEG-plus-context gives 76.09% balanced accuracy versus 78.05% for calibrated context alone; EEGNet gives 50.85% on unseen videos. This phase establishes no reliable EEG gain over both context controls. It reuses existing cohorts and leaves the manuscript/research question unchanged.

Use the maintained [publication pipeline](PUBLICATION.md) for new experiments:

```powershell
conda activate pytorch
python -m gruxnet train --cache ../publication_runs/cache_common14 --output ../publication_runs/my_subject_run --device cuda
```

The course-project scripts, pretrained model, paper PDF, and scores below are historical. Their pooled sample split and augmentation provenance do not establish unseen-subject or unseen-dataset generalization. The new pipeline verifies labels against source metadata, separates subjects before training, augments only training batches, and provides true leave-one-dataset-out evaluation. Its compact model is a distinct variant and requires new results.

**Data finding:** this extracted DEAP mirror has 439 valence/arousal ratings replaced by `9 - original`. The new loader uses the included participant-rating spreadsheet, joined by participant and video ID, and verifies dominance/liking to check alignment. Downloaded files remain untouched.

**Source verification:** the altered labels are already present in the supplied upstream DEAP Kaggle source. All 304 training input/metadata files match the three upstream archives by size and CRC32; six representative files also match SHA-256. [dataset_sources.json](dataset_sources.json) records the author's bundle, upstream URLs, pinned versions, and the limits of this verification. New audits and caches accept `--provenance dataset_sources.json` to bind this declaration.

Joint training on all three datasets is the historical research question. Alternative questions may be explored; adopting one requires showing the findings and receiving the author's approval. The [readiness record](docs/publication/GRU-XNet_Publication_Readiness_2026-10-05.md) tracks unresolved concerns. The completed [DEAP-only learning control](docs/publication/GRU-XNet_DEAP_Control_2026-10-05.md) is a diagnostic: raw EEGNet with training-only normalization reaches 46.25% held-out trial balanced accuracy, so this configuration supplies no reliable generalization gain. Its source, normalization, partitions, and predictions were verified using the commands in PUBLICATION.md. The [archived manuscript](docs/paper_archive/2026-10-05-pre-exploration/README.md) preserves the current source for backtracking.

**Exploratory findings:** matched per-window normalization gives 46.67% DEAP balanced accuracy, also near chance. A five-fold SEED-IV classical control reaches 67.13% binary trial balanced accuracy with 14 electrodes. [Initial findings](docs/publication/GRU-XNet_Exploration_Findings_2026-10-05.md). The completed 165-run feature-MLP investigation covers all three targets, both compute/exposure budgets and separate binary heads. Shared-joint versus target-only neural participant intervals include zero on every target/budget, while linear pooling penalties occur on SEED-IV/GAMEEMO. Simple heads do not consistently help. These separate target-centered development studies do not establish one jointly selected checkpoint's performance, a novel method, or submission readiness. [All-target findings and remaining work](docs/publication/GRU-XNet_Multitarget_Transfer_Findings_2026-10-05.md), [SEED-IV source controls and mdJPT prior work](docs/publication/GRU-XNet_Neural_Transfer_Investigation_2026-10-05.md).

**Transformer/native-label controls (6 October):** 120 additional neural runs compare a temporal transformer with an MLP of nearly identical parameter count, absolute/relative bandpower, and coarse/native supervision on identical fixed 40-second SEED-IV inputs. Absolute/coarse three-class BA is 44.24% transformer versus 42.00% MLP; its unadjusted paired interval is +0.14 to +4.47 percentage points. All eight native-minus-coarse label intervals include zero. The logistic binary mean (64.63%) exceeds every neural binary mean here. All 120 selected checkpoints and 29,160 neural/linear probability rows replay. These single-corpus development experiments do not test mitigation of cross-dataset negative transfer. [Findings and uncertainty](docs/publication/GRU-XNet_Transformer_Native_Label_Findings_2026-10-06.md), [protocol](docs/publication/Temporal_Native_Control_Protocol_2026-10-06.md), [additional primary prior work](docs/publication/GRU-XNet_Transformer_Research_Update_2026-10-06.md). That phase completed with **37 passing tests**; the manuscript and research question remain unchanged.

**Participant/session and frozen-pretraining controls (6 October):** 90 further neural fits and 70 selected linear models are verified, covering 34,560 test probability rows. Three-class BA drops from 45.64% familiar-session to 39.90% unseen-session for the small transformer; its MLP advantage disappears under the latter protocol. Frozen official REVE gives 39.14% versus 33.33% for the same randomly initialized frozen architecture on unseen sessions. Pretraining helps within this adapter, but does not establish a new method or resolve session transfer. Crossed participant/material uncertainty is reported; original video hashes and independently certified full pretraining-corpus exclusion remain unavailable. [Complete findings and limitations](docs/publication/GRU-XNet_Session_Pretraining_Findings_2026-10-06.md), [source-session protocol](docs/publication/Session_Stimulus_Control_Protocol_2026-10-06.md), [audited REVE protocol](docs/publication/Frozen_REVE_Control_Protocol_2026-10-06.md). That phase completed with **43 passing tests**. The manuscript and research question remain unchanged; the paper is still not submission-ready.

## Matched material controls (6 October 2026)

**Matched materials within a session (6 October).** The preceding participant/session phase had 43 tests; this phase completed with **47 passing tests**. All 540 new neural fits and 360 selected classical heads are complete. Every checkpoint and all 21,600 predictions replay; all 1,440 classical candidates independently refit with exact selected coefficients. Transformer three-class BA changes from 43.35% with shared materials to 39.92% with unseen materials, difference -3.44 pp with crossed participant/material interval [-7.88,+0.56]. All twelve exposure contrasts span zero: neither equivalence nor a strong new contribution is established. There is no clear unseen-material transformer advantage. [Full findings](docs/publication/GRU-XNet_Within_Session_Material_Findings_2026-10-06.md), [predeclared protocol](docs/publication/Within_Session_Material_Control_Protocol_2026-10-06.md), [primary literature and cross-corpus feasibility](docs/publication/GRU-XNet_Material_Generalization_Research_Update_2026-10-06.md). The manuscript and research question remain unchanged; the paper is still not submission-ready.

## Repeated groupings and DEAP control (6 October 2026)

**Repeated feature-control phase:** 680 new neural fits, 880 selected linear heads, all 3,520 independently refitted linear candidates and 160 video-prior diagnostics are complete. Two additional SEED-IV participant/video groupings and two compatible DEAP groupings use one fixed initialization; SEED also reuses the original initialization42. The suite has **53 passing tests**. Repeats reuse the same people/videos and are sensitivity checks. New fits produce 46,144 test predictions; every selected checkpoint and all analysis contrasts replay. [Full findings and limits](docs/publication/GRU-XNet_Repeated_Material_Findings_2026-10-06.md), [predeclared protocol](docs/publication/Repeated_Material_Control_Protocol_2026-10-06.md).

SEED transformer three-class BA changes from 43.27% shared to 39.07% unseen videos, difference -4.20 pp, crossed interval [-7.39,-1.07]. DEAP transformer binary BA is 52.28% versus 50.66%, difference -1.61 pp, interval [-4.97,+1.73]. All DEAP EEG-model exposure intervals span zero; neither corpus has a clear unseen-video transformer advantage over the MLP. A source-label-only DEAP video prior reaches 77.54% with shared videos and 49.67% unseen, without EEG. This shows contextual label predictability, not a proven neural mechanism. DEAP preparation and full source replay reproduce all 1,264 waveforms/features exactly; first-party signal authentication remains outstanding. The research question/manuscript are unchanged, and full GRU-XNet ablations, unseen-corpus validation and a justified new contribution remain unfinished.

## Full-width EEG and contextual controls (verified first pass)

All **680 full-width neural fits**, **170 context calibration cells** and **680 independently refitted regularization candidates** are complete. Every selected checkpoint and all 96 primary/168 within-video contrasts replay. The suite has **64 passing tests**. The [full findings](docs/publication/GRU-XNet_Full_Context_Findings_2026-10-06.md) retain every model, probability, selection history and limitation; the [protocol](docs/publication/Full_Context_Control_Protocol_2026-10-06.md) declares one participant/video grouping, one base initialization scheme and 200 updates with common14/40-second input adaptations.

Unseen-video balanced accuracy is 36.11% GRU, 36.91% matched BiLSTM and 38.77% local CBSAtt on SEED-IV (three classes); it is 46.54%, 48.34% and 51.10% on DEAP (binary valence). All primary reference-versus-GRU accuracy/log-loss intervals include zero. On shared DEAP videos, EEG-plus-context reaches 78.10% versus 77.36% for the raw video prior, difference +0.74 pp with exploratory crossed range [-0.94,+3.11]. All DEAP EEG-plus-context versus context-only and neural within-video alignment intervals include zero. This first pass establishes neither a GRU advantage nor useful incremental EEG prediction; it does not rule out information under stronger optimization or representations.

Training priors exclude the receiving participant's labels; test priors use source labels only. The local reference's pooling/dropout behavior is checked in training mode. Fifteen partial revision-1 fits are preserved and excluded. The full-width architecture retains the original CNN/recurrent widths with declared input adaptations; local CBSAtt is not authenticated author code or a reproduction of published scores. [Additional prior work and reference audit](docs/publication/GRU-XNet_Context_Alignment_Research_Update_2026-10-06.md). A public checkout can recompute the probability-based findings using `python scripts/verify_full_context_export.py`, without EEG or checkpoints.

The manuscript and main research question remain unchanged. The source-only learning phase below is complete; independent full-model confirmation, joint/LODO validation and final novelty assessment remain necessary. The paper is not submission-ready.

## Source-only learning diagnostics (verified, 6 October 2026)

All **96 declared fits** are complete and replayed: 16 real/permuted-label memorization checks and 80 source learning curves through 1,200 updates, retaining both learning rates. All **192 final/selected states**, **688 probability metric sets** and **12 exact older prefixes** pass. The suite has **71 passing tests**. GRU/EEGNet repeat two groupings and two initializations on partial source panels; outer-test rows are excluded from model/scaler/selection access. [Findings and limitations](docs/publication/GRU-XNet_Learning_Control_Findings_2026-10-06.md), [frozen protocol](docs/publication/Learning_Control_Protocol_2026-10-06.md).

Longer training improves training fit without reliable validation gains. At LR 0.0003, mean GRU training BA changes 76.85% → 94.37% on SEED-IV and 59.66% → 89.27% on DEAP between 200/1,200 updates; validation changes 34.72% → 29.17% and 49.64% → 46.39%. Both post-hoc source-normalization supplements are also verified: all 160 full-population and sixteen tiny-batch cases retain learned weights and reconstruct moments exactly. Tiny strict capacity passes change 14/16 → 16/16, including both originally failed EEGNet SEED inference states. This establishes capacity/evaluation-state behavior, without physiological-information or convergence claims.

The EEGNet port matches the pinned authors' executed TensorFlow function, including training-mode BatchNorm and constraints; published scores/optimizer trajectories are not reproduced. [Authentication and overlapping prior work](docs/publication/GRU-XNet_Learning_Control_Research_Update_2026-10-06.md). All original results and fitting sources remain frozen. Broader held-out confirmation requires per-outer-fold source selection and complete population coverage. A public checkout can recompute metrics, normalization contrasts and report means using the four verification commands in [PUBLICATION.md](PUBLICATION.md). The manuscript/research question remain unchanged.

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
