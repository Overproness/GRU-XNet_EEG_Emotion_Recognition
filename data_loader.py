"""
Data loader for gru_xnet that integrates with existing data loading pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, List, Dict, Optional
import pickle
from tqdm import tqdm
from pathlib import Path

from config import ExperimentConfig
from preprocessing import MultiDatasetSTFTPreprocessor, create_dataset_stft_configs

# Import from parent directory
from data_loading_pipeline import (
    DEAPAugmentedDataset,
    GAMEEMOAugmentedDataset,
    SEEDIVAugmentedDataset
)


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable channel counts across datasets
    
    Since different datasets have different channel counts (DEAP: 32, GAMEEMO: 14, SEEDIV: 62),
    we pad smaller ones to match the largest (62 channels for SEEDIV).
    
    Args:
        batch: List of (stft_features, label) tuples
        
    Returns:
        stft_features: Stacked tensor with shape (batch_size, 62, n_freq_bins, n_time_bins)
        labels: Stacked tensor with shape (batch_size,)
    """
    stft_features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    
    # Find max channel count and dimensions
    max_channels = max([x.shape[0] for x in stft_features_list])
    n_freq_bins = stft_features_list[0].shape[1]
    n_time_bins = stft_features_list[0].shape[2]
    
    # Pad all to max_channels
    padded_features = []
    for features in stft_features_list:
        n_channels = features.shape[0]
        if n_channels < max_channels:
            # Zero-pad to max_channels
            padding = torch.zeros(max_channels - n_channels, n_freq_bins, n_time_bins, dtype=features.dtype)
            features = torch.cat([features, padding], dim=0)
        padded_features.append(features)
    
    # Stack into batch
    stft_features = torch.stack(padded_features, dim=0)
    
    return stft_features, labels


class gru_xnetDataset(Dataset):
    """
    PyTorch Dataset for gru_xnet that handles STFT transformation
    """
    
    def __init__(
        self,
        data: List[np.ndarray],  # Changed to List to handle variable channel counts
        labels: np.ndarray,
        dataset_names: np.ndarray,
        subject_ids: np.ndarray,  # Added subject_ids parameter
        stft_preprocessor: MultiDatasetSTFTPreprocessor,
        cache_dir: Optional[str] = None,
        split_name: str = 'train'
    ):
        """
        Args:
            data: List of EEG samples (n_samples,) each with (n_channels, n_timepoints)
            labels: (n_samples,)
            dataset_names: (n_samples,) - dataset identifier for each sample
            subject_ids: (n_samples,) - subject ID for each sample
            stft_preprocessor: STFT preprocessor
            cache_dir: Directory to cache STFT features
            split_name: 'train', 'val', or 'test'
        """
        self.data = data
        self.labels = labels
        self.dataset_names = dataset_names
        self.subject_ids = subject_ids
        self.stft_preprocessor = stft_preprocessor
        self.cache_dir = cache_dir
        self.split_name = split_name
        
        # Create cache if enabled
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f'{split_name}_stft.pkl')
            self.stft_cache = self._load_or_compute_stft()
        else:
            self.stft_cache = None
            if split_name == 'train':
                print(f"STFT caching disabled - computing on-the-fly during training")
    
    def _load_or_compute_stft(self):
        """Load cached STFT or compute and cache"""
        if os.path.exists(self.cache_file):
            print(f"Loading cached STFT from {self.cache_file}...")
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Computing STFT for {len(self.data)} samples...")
        stft_features = []
        
        for i in tqdm(range(len(self.data)), desc=f"Computing STFT ({self.split_name})"):
            eeg_sample = self.data[i]
            dataset_name = self.dataset_names[i]
            
            # Apply STFT
            stft_feat = self.stft_preprocessor.transform(
                eeg_sample,
                dataset_name,
                standardize=True
            )
            stft_features.append(stft_feat)
        
        # Keep as list to handle variable channel counts across datasets
        # Don't convert to numpy array as shapes differ (DEAP:32ch, GAMEEMO:14ch, SEEDIV:62ch)
        
        # Cache for future use
        print(f"Caching STFT to {self.cache_file}...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(stft_features, f)
        
        return stft_features
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns:
            stft_features: (n_channels, n_freq_bins, n_time_bins)
            label: scalar
        """
        if self.stft_cache is not None:
            # Get from cache (already a numpy array)
            stft_features = self.stft_cache[idx]
        else:
            # Compute on-the-fly
            eeg_sample = self.data[idx]
            dataset_name = self.dataset_names[idx]
            stft_features = self.stft_preprocessor.transform(
                eeg_sample,
                dataset_name,
                standardize=True
            )
        
        # Convert to torch tensors
        # stft_features is already numpy array from either cache or transform
        stft_features = torch.from_numpy(stft_features).float()
        label = torch.tensor(self.labels[idx]).long()
        
        return stft_features, label


def load_combined_dataset(
    config: ExperimentConfig,
    verbose: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and combine all datasets with augmented data using data_loading_pipeline
    
    Args:
        config: Experiment configuration
        verbose: Whether to print loading progress
        
    Returns:
        data: List of arrays - (n_samples, n_channels, n_timepoints) - variable n_channels per dataset
        labels: (n_samples,)
        dataset_names: (n_samples,) - 'DEAP', 'GAMEEMO', or 'SEEDIV'
        subject_ids: (n_samples,)
    """
    all_data = []
    all_labels = []
    all_dataset_names = []
    all_subject_ids = []
    
    base_dir = config.data.base_dir
    augmented_dir = os.path.join(base_dir, config.data.augmented_dir)
    
    # Load each dataset
    for dataset_name in config.data.datasets:
        if verbose:
            print(f"\nLoading {dataset_name}...")
        
        if dataset_name == 'DEAP':
            dataset = DEAPAugmentedDataset(augmented_dir=augmented_dir)
            data_orig, labels_orig, subjects_orig = dataset.load_data()
        elif dataset_name == 'GAMEEMO':
            dataset = GAMEEMOAugmentedDataset(augmented_dir=augmented_dir)
            data_orig, labels_orig, subjects_orig = dataset.load_data()
        elif dataset_name == 'SEEDIV':
            dataset = SEEDIVAugmentedDataset(augmented_dir=augmented_dir)
            data_orig, labels_orig, subjects_orig = dataset.load_data()
            
            # Handle SEEDIV labels (4-class to binary if needed)
            if not config.data.seediv_keep_multiclass:
                if verbose:
                    print(f"Converting SEEDIV from 4-class to binary...")
                labels_orig = np.array([
                    config.data.seediv_binary_mapping.get(int(l), int(l))
                    for l in labels_orig
                ])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Store with dataset identifier
        all_data.append(data_orig)
        all_labels.append(labels_orig)
        all_dataset_names.extend([dataset_name] * len(labels_orig))
        all_subject_ids.append(subjects_orig)
        
        if verbose:
            print(f"  Loaded {len(labels_orig)} samples")
            print(f"  Shape: {data_orig.shape}")
            print(f"  Label distribution: {np.bincount(labels_orig.astype(int))}")
    
    # Combine labels, dataset_names, and subject_ids
    labels = np.concatenate(all_labels)
    dataset_names = np.array(all_dataset_names)
    subject_ids = np.concatenate(all_subject_ids)
    
    if verbose:
        print(f"\nTotal combined dataset:")
        print(f"  Total samples: {len(labels)}")
        print(f"  Datasets: {np.unique(dataset_names, return_counts=True)}")
        print(f"  Label distribution: {np.bincount(labels.astype(int))}")
    
    return all_data, labels, dataset_names, subject_ids


def split_data(
    data: List[np.ndarray],
    labels: np.ndarray,
    dataset_names: np.ndarray,
    subject_ids: np.ndarray,
    config: ExperimentConfig,
    random_state: int = 42
) -> Tuple:
    """
    Split data into train/val/test sets
    
    Args:
        data: List of arrays for each dataset
        labels: Combined labels
        dataset_names: Dataset identifiers
        subject_ids: Subject IDs
        config: Experiment configuration
        random_state: Random seed
        
    Returns:
        (train_data, train_labels, train_datasets, train_subjects),
        (val_data, val_labels, val_datasets, val_subjects),
        (test_data, test_labels, test_datasets, test_subjects)
    """
    np.random.seed(random_state)
    
    # Concatenate data for splitting
    # We need to handle variable channel counts
    n_samples_cumsum = [0]
    for d in data:
        n_samples_cumsum.append(n_samples_cumsum[-1] + len(d))
    
    # Create indices for each sample
    indices = np.arange(len(labels))
    
    if config.data.use_loso_cv:
        # Leave-One-Subject-Out
        test_mask = (subject_ids == config.data.cv_subject)
        train_val_mask = ~test_mask
        
        train_val_indices = indices[train_val_mask]
        test_indices = indices[test_mask]
        
        # Further split train into train and val
        n_train_val = len(train_val_indices)
        n_val = int(n_train_val * config.data.val_ratio / (1 - config.data.test_ratio))
        
        np.random.shuffle(train_val_indices)
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]
    else:
        # Random split
        np.random.shuffle(indices)
        
        n_samples = len(indices)
        n_train = int(n_samples * config.data.train_ratio)
        n_val = int(n_samples * config.data.val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
    
    # Extract splits
    def extract_split(split_indices):
        split_labels = labels[split_indices]
        split_datasets = dataset_names[split_indices]
        split_subjects = subject_ids[split_indices]
        
        # Extract data for each sample (keep as list for variable channel counts)
        split_data = []
        for idx in split_indices:
            # Find which dataset this sample belongs to
            for i in range(len(config.data.datasets)):
                if idx < n_samples_cumsum[i + 1]:
                    # Get local index within this dataset
                    local_idx = idx - n_samples_cumsum[i]
                    split_data.append(data[i][local_idx])
                    break
        
        return split_data, split_labels, split_datasets, split_subjects
    
    train_split = extract_split(train_indices)
    val_split = extract_split(val_indices)
    test_split = extract_split(test_indices)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_split[1])} samples")
    print(f"  Val: {len(val_split[1])} samples")
    print(f"  Test: {len(test_split[1])} samples")
    
    return train_split, val_split, test_split


def create_data_loaders(
    config: ExperimentConfig,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training
    
    Args:
        config: Experiment configuration
        verbose: Whether to print progress
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load combined dataset
    data, labels, dataset_names, subject_ids = load_combined_dataset(config, verbose)
    
    # Split data
    train_split, val_split, test_split = split_data(
        data, labels, dataset_names, subject_ids, config
    )
    
    # Create STFT preprocessor
    stft_configs = create_dataset_stft_configs(
        target_freq_range=config.stft.freq_range
    )
    stft_preprocessor = MultiDatasetSTFTPreprocessor(
        dataset_configs=stft_configs,
        target_freq_bins=config.stft.target_freq_bins,
        target_time_bins=config.stft.target_time_bins
    )
    
    # Create cache directory
    cache_dir = os.path.join(config.data.base_dir, config.data.cache_dir) if config.data.cache_stft else None
    
    # Create datasets - unpack the 4-element tuples correctly
    train_dataset = gru_xnetDataset(
        data=train_split[0],
        labels=train_split[1],
        dataset_names=train_split[2],
        subject_ids=train_split[3],
        stft_preprocessor=stft_preprocessor,
        cache_dir=cache_dir,
        split_name='train'
    )
    
    val_dataset = gru_xnetDataset(
        data=val_split[0],
        labels=val_split[1],
        dataset_names=val_split[2],
        subject_ids=val_split[3],
        stft_preprocessor=stft_preprocessor,
        cache_dir=cache_dir,
        split_name='val'
    )
    
    test_dataset = gru_xnetDataset(
        data=test_split[0],
        labels=test_split[1],
        dataset_names=test_split[2],
        subject_ids=test_split[3],
        stft_preprocessor=stft_preprocessor,
        cache_dir=cache_dir,
        split_name='test'
    )
    
    # Create samplers for balanced training
    train_sampler = None
    if config.data.balance_classes:
        class_counts = np.bincount(train_split[1].astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_split[1].astype(int)]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    print("Testing gru_xnet data loader...")
    
    from config import get_quick_test_config
    
    # Use quick test config
    config = get_quick_test_config()
    config.data.cache_stft = False  # Disable caching for quick test
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, verbose=True)
    
    # Test iteration
    print("\nTesting data iteration...")
    for batch_idx, (stft_features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  STFT shape: {stft_features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label values: {labels.unique()}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("\nData loader test passed!")
