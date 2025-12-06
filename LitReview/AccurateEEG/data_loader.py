"""
Custom Data Loader for GAMEEMO Dataset
Handles loading preprocessed EEG data and extracting features
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from config import *
from feature_extraction import FeatureExtractor


class GAMEEMODataset(Dataset):
    """PyTorch Dataset for GAMEEMO EEG data"""
    
    def __init__(self, features, labels, transform=None):
        """
        Initialize dataset
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
            transform: Optional transform to apply to features
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label


class GAMEEMODataLoader:
    """Data loader for GAMEEMO dataset with feature extraction"""
    
    def __init__(self, dataset_path, classification_type='binary', 
                 use_cached_features=False, cache_dir='./cache'):
        """
        Initialize GAMEEMO data loader
        
        Args:
            dataset_path: Path to GAMEEMO dataset directory
            classification_type: 'binary' or 'multiclass'
            use_cached_features: Whether to use cached extracted features
            cache_dir: Directory to save/load cached features
        """
        self.dataset_path = dataset_path
        self.classification_type = classification_type
        self.use_cached_features = use_cached_features
        self.cache_dir = cache_dir
        self.feature_extractor = FeatureExtractor(sampling_rate=SAMPLING_RATE)
        self.scaler = StandardScaler()
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Cache file name
        self.cache_file = os.path.join(cache_dir, 
                                      f'features_{classification_type}.pkl')
    
    def load_subject_data(self, subject_id, game_id):
        """
        Load preprocessed EEG data for a specific subject and game
        
        Args:
            subject_id: Subject ID (1-28)
            game_id: Game ID (1-4)
            
        Returns:
            numpy array of shape (n_channels, n_samples)
        """
        subject_folder = f'(S{subject_id:02d})'
        csv_file = f'S{subject_id:02d}G{game_id}AllChannels.csv'
        file_path = os.path.join(self.dataset_path, subject_folder, 
                                'Preprocessed EEG Data', '.csv format', csv_file)
        
        try:
            # Load CSV data
            data = pd.read_csv(file_path)
            
            # Extract only the channel columns (should be 14 channels)
            # Assuming the CSV has columns for each channel
            channel_data = data.values.T  # Transpose to (channels, samples)
            
            return channel_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def segment_signal(self, signal_data):
        """
        Segment continuous EEG signal into overlapping windows
        
        Args:
            signal_data: numpy array of shape (n_channels, n_samples)
            
        Returns:
            List of signal segments, each of shape (n_channels, window_size)
        """
        n_channels, n_samples = signal_data.shape
        window_size = WINDOW_SIZE_SAMPLES
        step_size = WINDOW_STEP_SAMPLES
        
        segments = []
        start_idx = 0
        
        while start_idx + window_size <= n_samples:
            segment = signal_data[:, start_idx:start_idx + window_size]
            segments.append(segment)
            start_idx += step_size
        
        return segments

    
    def get_label(self, game_id):
        """
        Get emotion label for a game
        
        Args:
            game_id: Game ID (1-4)
            
        Returns:
            Label value
        """
        game_key = f'G{game_id}'
        
        if self.classification_type == 'binary':
            emotion = GAME_TO_BINARY[game_key]
            return BINARY_LABELS[emotion]
        else:  # multiclass
            emotion = GAME_TO_EMOTION[game_key]
            return MULTICLASS_LABELS[emotion]
    
    def extract_features_from_signal(self, signal_data):
        """
        Extract features from multi-channel EEG signal
        
        Args:
            signal_data: numpy array of shape (n_channels, n_samples)
            
        Returns:
            1D numpy array of features
        """
        return self.feature_extractor.extract_features_multi_channel(signal_data)
    
    def load_all_data(self):
        """
        Load all data and extract features
        
        Returns:
            Tuple of (features, labels)
        """
        # Check if cached features exist
        if self.use_cached_features and os.path.exists(self.cache_file):
            print(f"Loading cached features from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data['features'], cache_data['labels']
        
        print("Extracting features from raw data...")
        print(f"Using {WINDOW_SIZE_SEC}s windows with {WINDOW_OVERLAP*100:.0f}% overlap")
        features_list = []
        labels_list = []
        
        # Iterate through all subjects and games
        for subject_id in range(1, NUM_SUBJECTS + 1):
            print(f"Processing Subject {subject_id}/{NUM_SUBJECTS}")
            
            for game_id in range(1, NUM_GAMES + 1):
                # Load EEG data
                signal_data = self.load_subject_data(subject_id, game_id)
                
                if signal_data is None:
                    continue
                
                # Segment signal into windows
                segments = self.segment_signal(signal_data)
                
                # Extract features from each segment
                label = self.get_label(game_id)
                
                for segment in segments:
                    features = self.extract_features_from_signal(segment)
                    features_list.append(features)
                    labels_list.append(label)
        
        # Convert to numpy arrays
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        print(f"Total segments extracted: {len(features)}")
        
        # Handle NaN and Inf values
        print("Checking for invalid values...")
        nan_mask = np.isnan(features)
        inf_mask = np.isinf(features)
        
        if nan_mask.any():
            print(f"  Found {nan_mask.sum()} NaN values - replacing with 0")
            features[nan_mask] = 0
        
        if inf_mask.any():
            print(f"  Found {inf_mask.sum()} Inf values - replacing with 0")
            features[inf_mask] = 0
        
        # Cache the features
        print(f"Caching features to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels}, f)
        
        return features, labels
    
    def prepare_dataloaders(self, batch_size=32, train_split=0.8, random_seed=42):
        """
        Prepare train and validation dataloaders
        
        Args:
            batch_size: Batch size for dataloaders
            train_split: Proportion of data for training
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader, feature_dim)
        """
        # Load all data
        features, labels = self.load_all_data()
        
        print(f"Total samples: {len(labels)}")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, train_size=train_split, 
            random_state=random_seed, stratify=labels
        )
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Handle any NaN/Inf values that may have appeared during normalization
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"After normalization - Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"After normalization - Val range: [{X_val.min():.4f}, {X_val.max():.4f}]")
        
        # Create datasets
        train_dataset = GAMEEMODataset(X_train, y_train)
        val_dataset = GAMEEMODataset(X_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=0
        )
        
        return train_loader, val_loader, features.shape[1]


def get_dataloaders(classification_type='binary', batch_size=32, 
                   use_cached=True, random_seed=42):
    """
    Convenience function to get dataloaders
    
    Args:
        classification_type: 'binary' or 'multiclass'
        batch_size: Batch size
        use_cached: Whether to use cached features
        random_seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, feature_dim)
    """
    data_loader = GAMEEMODataLoader(
        dataset_path=DATASET_PATH,
        classification_type=classification_type,
        use_cached_features=use_cached,
        cache_dir=os.path.join(OUTPUT_DIR, 'cache')
    )
    
    return data_loader.prepare_dataloaders(
        batch_size=batch_size,
        train_split=TRAIN_VAL_SPLIT,
        random_seed=random_seed
    )
