"""
Configuration file for EEG-based Emotion Recognition System
Based on: Accurate EEG-Based Emotion Recognition using LSTM and BiLSTM Networks
"""

import os

# Dataset Configuration
DATASET_PATH = r"YOUR GAMEEMO DATASET PATH HERE"  # Update with actual path
NUM_SUBJECTS = 28
NUM_CHANNELS = 14
NUM_GAMES = 4
SAMPLING_RATE = 128  # Hz

# Windowing Configuration for Data Segmentation
WINDOW_SIZE_SEC = 4  # Window size in seconds
WINDOW_OVERLAP = 0.5  # 50% overlap between consecutive windows
WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_SEC * SAMPLING_RATE)  # 512 samples
WINDOW_STEP_SAMPLES = int(WINDOW_SIZE_SAMPLES * (1 - WINDOW_OVERLAP))  # 256 samples

# Channel names for Emotiv EPOC+ 14-Channel headset
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# Emotion Labels
# Binary: Positive (1), Negative (0)
BINARY_LABELS = {
    'positive': 1,
    'negative': 0
}

# Multiclass: LANV (0), LAPV (1), HANV (2), HAPV (3)
# L/H = Low/High Arousal, N/P = Negative/Positive Valence
MULTICLASS_LABELS = {
    'LANV': 0,  # Low Arousal, Negative Valence
    'LAPV': 1,  # Low Arousal, Positive Valence
    'HANV': 2,  # High Arousal, Negative Valence
    'HAPV': 3   # High Arousal, Positive Valence
}

# Game to emotion mapping (based on paper: boring, calm, horror, funny)
GAME_TO_EMOTION = {
    'G1': 'LANV',  # Boring game -> Low Arousal, Negative Valence
    'G2': 'LAPV',  # Calm game -> Low Arousal, Positive Valence
    'G3': 'HANV',  # Horror game -> High Arousal, Negative Valence
    'G4': 'HAPV'   # Funny game -> High Arousal, Positive Valence
}

# Binary mapping
GAME_TO_BINARY = {
    'G1': 'negative',  # Boring
    'G2': 'positive',  # Calm
    'G3': 'negative',  # Horror
    'G4': 'positive'   # Funny
}

# Feature Extraction Configuration
WAVELET_NAME = 'db4'  # Daubechies 4 wavelet
WAVELET_LEVEL = 4  # Decomposition level

# Frequency bands for EEG
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 64)
}

# Model Configuration
# Binary Classification LSTM
BINARY_LSTM_CONFIG = {
    'input_size': None,  # Will be set based on features
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.5,
    'bidirectional': False,
    'num_classes': 2
}

# Binary Classification BiLSTM
BINARY_BILSTM_CONFIG = {
    'input_size': None,
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.5,
    'bidirectional': True,
    'num_classes': 2
}

# Multiclass LSTM
MULTICLASS_LSTM_CONFIG = {
    'input_size': None,
    'hidden_size_1': 128,
    'hidden_size_2': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'bidirectional': False,
    'num_classes': 4
}

# Multiclass BiLSTM
MULTICLASS_BILSTM_CONFIG = {
    'input_size': None,
    'hidden_size_1': 128,
    'hidden_size_2': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'bidirectional': True,
    'num_classes': 4
}

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
TRAIN_VAL_SPLIT = 0.8

# Device configuration
DEVICE = 'cuda'  # Will be set to 'cuda' if available, else 'cpu'

# Output directories
OUTPUT_DIR = r"YOUR OUTPUT DIRECTORY PATH HERE"  # Update with actual path
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Random seed for reproducibility
RANDOM_SEED = 42
