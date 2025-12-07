"""
Beginner-level EEG augmentation techniques
Safe, reliable, and easy to implement augmentations (40-60% of augmented data)
"""

import numpy as np
from typing import Tuple, Optional, Union
from base_augmentation import NoiseBasedAugmentation, TemporalAugmentation, BaseAugmentation, AugmentationType


class GaussianNoiseAugmentation(NoiseBasedAugmentation):
    """
    Add Gaussian noise to EEG signals
    Contribution: 15-35% of augmented samples
    Reliability: HIGH - helps robustness to sensor noise
    """
    
    def __init__(self, 
                 snr_range: Tuple[float, float] = (0.001, 0.05),
                 random_seed: Optional[int] = None):
        """
        Args:
            snr_range: Range of noise levels as fraction of signal std
            random_seed: Random seed for reproducibility
        """
        super().__init__("Gaussian Noise", random_seed=random_seed)
        self.snr_range = snr_range
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Add Gaussian noise to the signal"""
        self.validate_data(data)
        
        # Random noise level within range
        noise_level = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Compute signal std and generate noise
        signal_std = np.std(data)
        noise = np.random.randn(*data.shape) * signal_std * noise_level
        
        augmented = data + noise
        
        return augmented, label


class TimeShiftAugmentation(TemporalAugmentation):
    """
    Shift signal in time (circular shift)
    Contribution: 10-25% of augmented samples
    Reliability: HIGH for event-independent tasks
    """
    
    def __init__(self,
                 shift_range: Tuple[float, float] = (0.05, 0.20),
                 random_seed: Optional[int] = None):
        """
        Args:
            shift_range: Range of shift as fraction of window length (±5-20%)
            random_seed: Random seed
        """
        super().__init__("Time Shift", random_seed=random_seed)
        self.shift_range = shift_range
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Apply time shift"""
        self.validate_data(data)
        
        if data.ndim == 1:
            n_samples = len(data)
        else:
            n_samples = data.shape[1]
        
        # Random shift amount
        shift_fraction = np.random.uniform(self.shift_range[0], self.shift_range[1])
        shift_direction = np.random.choice([-1, 1])
        shift_samples = int(n_samples * shift_fraction * shift_direction)
        
        # Apply circular shift
        augmented = np.roll(data, shift_samples, axis=-1)
        
        return augmented, label


class WindowSlicingAugmentation(TemporalAugmentation):
    """
    Crop or slice temporal window
    Contribution: 10-25% of augmented samples
    Reliability: HIGH - effective for long recordings
    """
    
    def __init__(self,
                 crop_range: Tuple[float, float] = (0.80, 0.95),
                 random_seed: Optional[int] = None):
        """
        Args:
            crop_range: Range of window to keep (80-95%)
            random_seed: Random seed
        """
        super().__init__("Window Slicing", random_seed=random_seed)
        self.crop_range = crop_range
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                return_original_length: bool = True,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Crop window and optionally resize back to original length"""
        self.validate_data(data)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
        
        n_channels, n_samples = data.shape
        
        # Random crop size
        crop_fraction = np.random.uniform(self.crop_range[0], self.crop_range[1])
        crop_length = int(n_samples * crop_fraction)
        
        # Random crop position
        max_start = n_samples - crop_length
        start_idx = np.random.randint(0, max_start + 1)
        end_idx = start_idx + crop_length
        
        # Crop
        cropped = data[:, start_idx:end_idx]
        
        # Optionally resize back to original length using interpolation
        if return_original_length:
            from base_augmentation import safe_reshape_temporal
            augmented = safe_reshape_temporal(cropped, n_samples, method='interpolate')
        else:
            augmented = cropped
        
        if was_1d:
            augmented = augmented.flatten()
        
        return augmented, label


class AmplitudeScalingAugmentation(BaseAugmentation):
    """
    Scale amplitude by random factor
    Contribution: 5-15% of augmented samples
    Reliability: HIGH if physiological range preserved
    """
    
    def __init__(self,
                 scaling_range: Tuple[float, float] = (0.90, 1.10),
                 random_seed: Optional[int] = None):
        """
        Args:
            scaling_range: Range of scaling factors (0.9-1.1 = ±10%)
            random_seed: Random seed
        """
        super().__init__("Amplitude Scaling", AugmentationType.BEGINNER, random_seed=random_seed)
        self.scaling_range = scaling_range
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                per_channel: bool = False,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Scale amplitude"""
        self.validate_data(data)
        
        if per_channel and data.ndim == 2:
            # Different scaling per channel
            n_channels = data.shape[0]
            scaling_factors = np.random.uniform(
                self.scaling_range[0], 
                self.scaling_range[1], 
                size=n_channels
            )
            augmented = data * scaling_factors[:, np.newaxis]
        else:
            # Same scaling for all channels
            scaling_factor = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
            augmented = data * scaling_factor
        
        return augmented, label


class ChannelDropoutAugmentation(BaseAugmentation):
    """
    Randomly drop/mask channels to simulate bad electrodes
    Contribution: 5-15% of augmented samples
    Reliability: HIGH for robustness to missing channels
    """
    
    def __init__(self,
                 dropout_rate_range: Tuple[float, float] = (0.05, 0.20),
                 random_seed: Optional[int] = None):
        """
        Args:
            dropout_rate_range: Range of fraction of channels to drop (5-20%)
            random_seed: Random seed
        """
        super().__init__("Channel Dropout", AugmentationType.BEGINNER, random_seed=random_seed)
        self.dropout_rate_range = dropout_rate_range
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                fill_method: str = 'zero',
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Drop random channels"""
        self.validate_data(data)
        
        if data.ndim == 1:
            # Single channel, cannot drop
            return data.copy(), label
        
        n_channels = data.shape[0]
        
        # Random dropout rate
        dropout_rate = np.random.uniform(self.dropout_rate_range[0], self.dropout_rate_range[1])
        n_drop = max(1, int(n_channels * dropout_rate))
        
        # Randomly select channels to drop
        drop_indices = np.random.choice(n_channels, size=n_drop, replace=False)
        
        # Copy data and drop channels
        augmented = data.copy()
        
        if fill_method == 'zero':
            augmented[drop_indices, :] = 0
        elif fill_method == 'mean':
            for idx in drop_indices:
                augmented[idx, :] = np.mean(data, axis=0)
        elif fill_method == 'interpolate':
            # Simple interpolation from neighboring channels
            for idx in drop_indices:
                neighbors = []
                if idx > 0:
                    neighbors.append(data[idx - 1, :])
                if idx < n_channels - 1:
                    neighbors.append(data[idx + 1, :])
                if neighbors:
                    augmented[idx, :] = np.mean(neighbors, axis=0)
                else:
                    augmented[idx, :] = 0
        
        return augmented, label


# Convenience function to create all beginner augmentations
def create_beginner_augmentations(random_seed: Optional[int] = None):
    """
    Create instances of all beginner-level augmentations
    
    Returns:
        Dictionary of augmentation instances
    """
    return {
        'gaussian_noise': GaussianNoiseAugmentation(random_seed=random_seed),
        'time_shift': TimeShiftAugmentation(random_seed=random_seed),
        'window_slicing': WindowSlicingAugmentation(random_seed=random_seed),
        'amplitude_scaling': AmplitudeScalingAugmentation(random_seed=random_seed),
        'channel_dropout': ChannelDropoutAugmentation(random_seed=random_seed),
    }


if __name__ == "__main__":
    # Test each augmentation
    print("Testing Beginner Augmentations")
    print("=" * 80)
    
    # Create sample EEG data (32 channels, 256 samples)
    np.random.seed(42)
    sample_data = np.random.randn(32, 256)
    sample_label = 1
    
    augmentations = create_beginner_augmentations(random_seed=42)
    
    for name, aug in augmentations.items():
        print(f"\n{name.upper()}:")
        aug_data, aug_label = aug.augment(sample_data, sample_label)
        print(f"  Original shape: {sample_data.shape}, Augmented shape: {aug_data.shape}")
        print(f"  Original std: {np.std(sample_data):.4f}, Augmented std: {np.std(aug_data):.4f}")
        quality = aug.check_quality(sample_data, aug_data)
        print(f"  Quality metrics: {quality}")
