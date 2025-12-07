"""
Base classes for EEG data augmentation
Provides abstract interface and common utilities for all augmentation techniques
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict
import numpy as np
from enum import Enum


class AugmentationType(Enum):
    """Enumeration of augmentation technique types"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class BaseAugmentation(ABC):
    """
    Abstract base class for all EEG augmentation techniques
    
    All augmentation techniques should inherit from this class and implement
    the augment() method.
    """
    
    def __init__(self, 
                 name: str,
                 augmentation_type: AugmentationType,
                 random_seed: Optional[int] = None):
        """
        Initialize base augmentation
        
        Args:
            name: Name of the augmentation technique
            augmentation_type: Type/complexity level of augmentation
            random_seed: Random seed for reproducibility
        """
        self.name = name
        self.augmentation_type = augmentation_type
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    @abstractmethod
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """
        Apply augmentation to EEG data
        
        Args:
            data: EEG data, shape (n_channels, n_samples) or (n_samples,) for single channel
            label: Optional label(s) associated with the data
            **kwargs: Additional parameters specific to the augmentation
        
        Returns:
            Tuple of (augmented_data, augmented_label)
        """
        pass
    
    def validate_data(self, data: np.ndarray) -> bool:
        """
        Validate input data format
        
        Args:
            data: Input EEG data
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Data must be numpy array, got {type(data)}")
        
        if data.ndim not in [1, 2]:
            raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")
        
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")
        
        return True
    
    def check_quality(self, 
                     original: np.ndarray, 
                     augmented: np.ndarray,
                     max_std_change: float = 3.0) -> Dict[str, float]:
        """
        Check quality of augmented data compared to original
        
        Args:
            original: Original EEG data
            augmented: Augmented EEG data
            max_std_change: Maximum allowed change in standard deviation
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        # Check amplitude preservation
        orig_std = np.std(original)
        aug_std = np.std(augmented)
        std_ratio = aug_std / (orig_std + 1e-10)
        quality_metrics['std_ratio'] = std_ratio
        quality_metrics['amplitude_preserved'] = abs(std_ratio - 1.0) < max_std_change
        
        # Check signal correlation
        if original.shape == augmented.shape:
            correlation = np.corrcoef(original.flatten(), augmented.flatten())[0, 1]
            quality_metrics['correlation'] = correlation
        else:
            quality_metrics['correlation'] = None
        
        # Check mean shift
        orig_mean = np.mean(original)
        aug_mean = np.mean(augmented)
        quality_metrics['mean_shift'] = abs(aug_mean - orig_mean)
        
        return quality_metrics
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.augmentation_type.value})"


class NoiseBasedAugmentation(BaseAugmentation):
    """Base class for noise-based augmentation techniques"""
    
    def __init__(self, name: str, noise_level: float = 0.01, **kwargs):
        super().__init__(name, AugmentationType.BEGINNER, **kwargs)
        self.noise_level = noise_level


class TemporalAugmentation(BaseAugmentation):
    """Base class for temporal/time-based augmentation techniques"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, AugmentationType.BEGINNER, **kwargs)


class FrequencyAugmentation(BaseAugmentation):
    """Base class for frequency-domain augmentation techniques"""
    
    def __init__(self, name: str, sampling_rate: int = 128, **kwargs):
        super().__init__(name, AugmentationType.INTERMEDIATE, **kwargs)
        self.sampling_rate = sampling_rate


class GenerativeAugmentation(BaseAugmentation):
    """Base class for generative augmentation techniques (GAN, VAE, etc.)"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, AugmentationType.ADVANCED, **kwargs)
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train_generator(self, training_data: np.ndarray, labels: np.ndarray):
        """
        Train the generative model
        
        Args:
            training_data: Training EEG data
            labels: Training labels
        """
        pass
    
    def check_trained(self):
        """Check if generator is trained before use"""
        if not self.is_trained:
            raise RuntimeError(f"{self.name} generator must be trained before use. Call train_generator() first.")


def compute_spectral_similarity(signal1: np.ndarray, 
                                signal2: np.ndarray,
                                fs: int = 128) -> float:
    """
    Compute spectral similarity between two signals using FFT
    
    Args:
        signal1: First signal
        signal2: Second signal
        fs: Sampling frequency
        
    Returns:
        Correlation between power spectra (0-1)
    """
    # Compute power spectral density
    fft1 = np.abs(np.fft.rfft(signal1.flatten()))
    fft2 = np.abs(np.fft.rfft(signal2.flatten()))
    
    # Ensure same length
    min_len = min(len(fft1), len(fft2))
    fft1 = fft1[:min_len]
    fft2 = fft2[:min_len]
    
    # Compute correlation
    correlation = np.corrcoef(fft1, fft2)[0, 1]
    
    return correlation


def safe_reshape_temporal(data: np.ndarray, 
                          target_length: int,
                          method: str = 'interpolate') -> np.ndarray:
    """
    Safely reshape temporal dimension of EEG data
    
    Args:
        data: Input data (n_channels, n_samples) or (n_samples,)
        target_length: Target number of samples
        method: 'interpolate', 'crop', or 'pad'
        
    Returns:
        Reshaped data
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_samples = data.shape
    
    if method == 'interpolate':
        # Use linear interpolation
        old_indices = np.linspace(0, n_samples - 1, n_samples)
        new_indices = np.linspace(0, n_samples - 1, target_length)
        reshaped = np.array([np.interp(new_indices, old_indices, data[ch]) 
                            for ch in range(n_channels)])
    elif method == 'crop':
        if target_length > n_samples:
            raise ValueError(f"Cannot crop to longer length: {n_samples} -> {target_length}")
        start = (n_samples - target_length) // 2
        reshaped = data[:, start:start + target_length]
    elif method == 'pad':
        if target_length < n_samples:
            raise ValueError(f"Cannot pad to shorter length: {n_samples} -> {target_length}")
        pad_total = target_length - n_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        reshaped = np.pad(data, ((0, 0), (pad_left, pad_right)), mode='edge')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reshaped


if __name__ == "__main__":
    # Test spectral similarity
    print("Testing spectral similarity computation...")
    signal1 = np.random.randn(128)
    signal2 = signal1 + np.random.randn(128) * 0.1
    similarity = compute_spectral_similarity(signal1, signal2)
    print(f"Spectral similarity: {similarity:.3f}")
    
    # Test safe reshape
    print("\nTesting safe temporal reshape...")
    data = np.random.randn(32, 256)
    reshaped = safe_reshape_temporal(data, 128, method='interpolate')
    print(f"Original shape: {data.shape}, Reshaped: {reshaped.shape}")
