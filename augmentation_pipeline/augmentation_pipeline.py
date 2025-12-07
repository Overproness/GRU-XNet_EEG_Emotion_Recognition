"""
Main augmentation pipeline orchestrator
Coordinates all augmentation techniques to achieve 1:2 ratio (original:augmented)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

from augmentation_config import AugmentationConfig, DEFAULT_CONFIG
from base_augmentation import BaseAugmentation
from beginner_augmentations import create_beginner_augmentations
from intermediate_augmentations import create_intermediate_augmentations
from advanced_augmentations import create_advanced_augmentations


class AugmentationPipeline:
    """
    Main pipeline for EEG data augmentation
    
    Applies multiple augmentation techniques in the correct proportions
    to achieve target augmentation ratio (default 1:2)
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Augmentation configuration (uses default if None)
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        
        # Initialize statistics first (needed by _init_augmentations)
        self.stats = {
            'n_original': 0,
            'n_augmented': 0,
            'augmentation_counts': {},
        }
        
        # Validate configuration
        if not self.config.validate_contributions():
            print("Warning: Augmentation contributions don't match target ratio")
            print("Adjusting contributions automatically...")
            self._auto_adjust_contributions()
        
        # Initialize augmentation techniques
        self.augmentations = {}
        self._init_augmentations()
    
    def _init_augmentations(self):
        """Initialize all enabled augmentation techniques"""
        # Beginner augmentations
        beginner_augs = create_beginner_augmentations(random_seed=self.config.random_seed)
        if self.config.use_gaussian_noise:
            self.augmentations['gaussian_noise'] = (
                beginner_augs['gaussian_noise'], 
                self.config.gaussian_noise_contribution
            )
        if self.config.use_time_shift:
            self.augmentations['time_shift'] = (
                beginner_augs['time_shift'],
                self.config.time_shift_contribution
            )
        if self.config.use_window_slicing:
            self.augmentations['window_slicing'] = (
                beginner_augs['window_slicing'],
                self.config.window_slicing_contribution
            )
        if self.config.use_amplitude_scaling:
            self.augmentations['amplitude_scaling'] = (
                beginner_augs['amplitude_scaling'],
                self.config.amplitude_scaling_contribution
            )
        if self.config.use_channel_dropout:
            self.augmentations['channel_dropout'] = (
                beginner_augs['channel_dropout'],
                self.config.channel_dropout_contribution
            )
        
        # Intermediate augmentations
        sampling_rate = self.config.dataset_sampling_rates.get('DEAP', 128)
        intermediate_augs = create_intermediate_augmentations(
            sampling_rate=sampling_rate,
            random_seed=self.config.random_seed
        )
        if self.config.use_frequency_filtering:
            self.augmentations['frequency_filtering'] = (
                intermediate_augs['frequency_filtering'],
                self.config.frequency_filtering_contribution
            )
        if self.config.use_timefreq_augmentation:
            self.augmentations['timefreq_augmentation'] = (
                intermediate_augs['timefreq_augmentation'],
                self.config.timefreq_augmentation_contribution
            )
        if self.config.use_mixup:
            self.augmentations['mixup'] = (
                intermediate_augs['mixup'],
                self.config.mixup_contribution
            )
        if self.config.use_cutmix:
            self.augmentations['cutmix'] = (
                intermediate_augs['cutmix'],
                self.config.cutmix_contribution
            )
        
        # Advanced augmentations
        advanced_augs = create_advanced_augmentations(random_seed=self.config.random_seed)
        if self.config.use_smote:
            self.augmentations['smote'] = (
                advanced_augs['smote'],
                self.config.smote_contribution
            )
        
        # Initialize stats
        for name in self.augmentations.keys():
            self.stats['augmentation_counts'][name] = 0
    
    def _auto_adjust_contributions(self):
        """Automatically adjust contributions to match target ratio"""
        contributions = self.config.get_contribution_summary()
        total = sum(contributions.values())
        
        if total == 0:
            print("Error: No augmentations enabled!")
            return
        
        # Scale all contributions proportionally
        scale_factor = self.config.target_augmentation_ratio / total
        
        for attr_name in dir(self.config):
            if attr_name.endswith('_contribution') and not attr_name.startswith('_'):
                current_value = getattr(self.config, attr_name)
                setattr(self.config, attr_name, current_value * scale_factor)
    
    def augment_dataset(self, 
                       data: np.ndarray,
                       labels: np.ndarray,
                       return_original: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment entire dataset
        
        Args:
            data: Original data, shape (n_samples, n_channels, n_timepoints)
            labels: Original labels, shape (n_samples,)
            return_original: If True, include original data in output
            
        Returns:
            augmented_data: Augmented + (optionally) original data
            augmented_labels: Corresponding labels
        """
        n_original = len(data)
        self.stats['n_original'] = n_original
        
        # Calculate number of augmented samples needed per technique
        n_augmented_per_technique = {}
        for name, (aug_obj, contribution) in self.augmentations.items():
            n_augmented_per_technique[name] = int(n_original * contribution)
        
        print(f"\nAugmenting {n_original} original samples...")
        print(f"Target: {int(n_original * self.config.target_augmentation_ratio)} augmented samples")
        print("-" * 80)
        
        # Store augmented samples
        all_augmented_data = []
        all_augmented_labels = []
        
        if return_original:
            all_augmented_data.append(data)
            all_augmented_labels.append(labels)
        
        # Apply each augmentation technique
        for name, (aug_obj, contribution) in self.augmentations.items():
            n_needed = n_augmented_per_technique[name]
            if n_needed == 0:
                continue
            
            print(f"Applying {name}: generating {n_needed} samples...")
            
            augmented_batch = []
            augmented_labels_batch = []
            
            # Special handling for techniques that require pairs
            if name in ['mixup', 'cutmix']:
                augmented_batch, augmented_labels_batch = self._apply_mixing_augmentation(
                    data, labels, aug_obj, n_needed
                )
            # SMOTE needs to be fitted first
            elif name == 'smote':
                aug_obj.fit(data, labels)
                for i in range(n_needed):
                    # Randomly select a sample
                    idx = np.random.randint(0, n_original)
                    aug_data, aug_label = aug_obj.augment(data[idx], labels[idx])
                    augmented_batch.append(aug_data)
                    augmented_labels_batch.append(aug_label)
            else:
                # Standard augmentation
                for i in range(n_needed):
                    # Randomly select a sample to augment
                    idx = np.random.randint(0, n_original)
                    aug_data, aug_label = aug_obj.augment(data[idx], labels[idx])
                    augmented_batch.append(aug_data)
                    augmented_labels_batch.append(aug_label)
            
            # Add to collection
            all_augmented_data.append(np.array(augmented_batch))
            all_augmented_labels.append(np.array(augmented_labels_batch))
            
            # Update stats
            self.stats['augmentation_counts'][name] = n_needed
        
        # Concatenate all augmented data
        final_data = np.concatenate(all_augmented_data, axis=0)
        final_labels = np.concatenate(all_augmented_labels, axis=0)
        
        self.stats['n_augmented'] = len(final_data) - (n_original if return_original else 0)
        
        print("-" * 80)
        print(f"Total augmented samples: {self.stats['n_augmented']}")
        print(f"Final dataset size: {len(final_data)}")
        print(f"Actual ratio (original:augmented): 1:{self.stats['n_augmented']/n_original:.2f}")
        
        return final_data, final_labels
    
    def _apply_mixing_augmentation(self,
                                   data: np.ndarray,
                                   labels: np.ndarray,
                                   aug_obj: BaseAugmentation,
                                   n_needed: int) -> Tuple[List, List]:
        """Apply mixing-based augmentation (mixup, cutmix)"""
        augmented_batch = []
        augmented_labels_batch = []
        
        n_original = len(data)
        
        for i in range(n_needed):
            # Randomly select two samples
            idx1 = np.random.randint(0, n_original)
            idx2 = np.random.randint(0, n_original)
            
            # Apply augmentation
            aug_data, aug_label = aug_obj.augment(
                data[idx1], labels[idx1],
                data2=data[idx2], label2=labels[idx2]
            )
            
            augmented_batch.append(aug_data)
            
            # Handle mixed labels
            if isinstance(aug_label, dict):
                # Use hard label from first sample for simplicity
                # In real training, you'd use soft labels
                augmented_labels_batch.append(aug_label['hard1'])
            else:
                augmented_labels_batch.append(aug_label)
        
        return augmented_batch, augmented_labels_batch
    
    def print_statistics(self):
        """Print augmentation statistics"""
        print("\n" + "=" * 80)
        print("AUGMENTATION STATISTICS")
        print("=" * 80)
        print(f"Original samples: {self.stats['n_original']}")
        print(f"Augmented samples: {self.stats['n_augmented']}")
        print(f"Ratio: 1:{self.stats['n_augmented'] / max(1, self.stats['n_original']):.2f}")
        print("\nPer-technique breakdown:")
        print("-" * 80)
        
        for name, count in self.stats['augmentation_counts'].items():
            if count > 0:
                pct = (count / self.stats['n_augmented']) * 100 if self.stats['n_augmented'] > 0 else 0
                print(f"  {name:30s}: {count:6d} samples ({pct:5.1f}%)")
        
        print("=" * 80)
    
    def save_augmented_dataset(self, 
                              data: np.ndarray,
                              labels: np.ndarray,
                              output_dir: str,
                              dataset_name: str):
        """
        Save augmented dataset to disk
        
        Args:
            data: Augmented data
            labels: Augmented labels
            output_dir: Output directory
            dataset_name: Name for the saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_file = output_path / f"{dataset_name}_augmented_data.npy"
        labels_file = output_path / f"{dataset_name}_augmented_labels.npy"
        stats_file = output_path / f"{dataset_name}_augmentation_stats.json"
        
        np.save(data_file, data)
        np.save(labels_file, labels)
        
        # Save statistics
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\nSaved augmented dataset to {output_path}")
        print(f"  Data: {data_file}")
        print(f"  Labels: {labels_file}")
        print(f"  Stats: {stats_file}")


def quick_augment(data: np.ndarray, 
                  labels: np.ndarray,
                  ratio: float = 2.0,
                  random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick augmentation with default settings
    
    Args:
        data: Original data (n_samples, n_channels, n_timepoints)
        labels: Original labels (n_samples,)
        ratio: Augmentation ratio (default 2.0 for 1:2)
        random_seed: Random seed
        
    Returns:
        Augmented data and labels (including originals)
    """
    config = AugmentationConfig()
    config.target_augmentation_ratio = ratio
    config.random_seed = random_seed
    
    pipeline = AugmentationPipeline(config)
    
    aug_data, aug_labels = pipeline.augment_dataset(data, labels)
    pipeline.print_statistics()
    
    return aug_data, aug_labels


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing Augmentation Pipeline")
    print("=" * 80)
    
    # Create sample dataset (32 samples, 32 channels, 256 timepoints)
    np.random.seed(42)
    n_samples = 32
    n_channels = 32
    n_timepoints = 256
    
    data = np.random.randn(n_samples, n_channels, n_timepoints)
    labels = np.random.randint(0, 3, n_samples)
    
    print(f"Original dataset: {data.shape}")
    print(f"Labels: {labels.shape}")
    
    # Create configuration
    config = AugmentationConfig()
    config.print_summary()
    
    # Create pipeline
    pipeline = AugmentationPipeline(config)
    
    # Augment dataset
    aug_data, aug_labels = pipeline.augment_dataset(data, labels)
    
    print(f"\nFinal dataset: {aug_data.shape}")
    print(f"Final labels: {aug_labels.shape}")
    
    # Print statistics
    pipeline.print_statistics()
    
    # Test quick augment
    print("\n\nTesting quick_augment function:")
    aug_data2, aug_labels2 = quick_augment(data, labels, ratio=2.0)
    print(f"Result: {aug_data2.shape}")
