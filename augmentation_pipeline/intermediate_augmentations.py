"""
Intermediate-level EEG augmentation techniques
Signal-aware, frequency-domain, and mixing-based augmentations (20-30% of augmented data)
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq
from base_augmentation import FrequencyAugmentation, BaseAugmentation, AugmentationType


class FrequencyFilteringAugmentation(FrequencyAugmentation):
    """
    Apply slight variations in frequency filtering
    Contribution: 5-15% of augmented samples
    Reliability: MEDIUM-HIGH - good when features live in frequency bands
    """
    
    def __init__(self,
                 freq_shift_range: Tuple[float, float] = (1.0, 2.0),
                 sampling_rate: int = 128,
                 random_seed: Optional[int] = None):
        """
        Args:
            freq_shift_range: Range to shift band edges (±1-2 Hz)
            sampling_rate: Sampling frequency in Hz
            random_seed: Random seed
        """
        super().__init__("Frequency Filtering", sampling_rate=sampling_rate, random_seed=random_seed)
        self.freq_shift_range = freq_shift_range
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                band: str = 'broadband',
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Apply frequency filtering with shifted edges"""
        self.validate_data(data)
        
        # Standard EEG bands (can be shifted)
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
            'broadband': (0.5, 45),
        }
        
        if band not in freq_bands:
            band = 'broadband'
        
        low_freq, high_freq = freq_bands[band]
        
        # Random shift
        shift = np.random.uniform(self.freq_shift_range[0], self.freq_shift_range[1])
        shift_direction = np.random.choice([-1, 1])
        shift *= shift_direction
        
        # Apply shift (ensuring valid range)
        low_freq = max(0.1, low_freq + shift)
        high_freq = min(self.sampling_rate / 2 - 1, high_freq + shift)
        
        # Design bandpass filter
        nyquist = self.sampling_rate / 2
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist
        
        # Ensure valid range
        low_normalized = max(0.01, min(0.99, low_normalized))
        high_normalized = max(low_normalized + 0.01, min(0.99, high_normalized))
        
        try:
            sos = scipy_signal.butter(4, [low_normalized, high_normalized], 
                                     btype='band', output='sos')
            
            # Apply filter
            if data.ndim == 1:
                augmented = scipy_signal.sosfilt(sos, data)
            else:
                augmented = np.array([scipy_signal.sosfilt(sos, data[ch]) 
                                     for ch in range(data.shape[0])])
        except Exception as e:
            # If filtering fails, return original
            print(f"Warning: Filtering failed ({e}), returning original data")
            augmented = data.copy()
        
        return augmented, label


class TimeFrequencyAugmentation(FrequencyAugmentation):
    """
    Time-frequency domain augmentation using STFT
    Contribution: 10-30% of augmented samples
    Reliability: HIGH when models accept TF input
    """
    
    def __init__(self,
                 mask_time_prob: float = 0.1,
                 mask_freq_prob: float = 0.1,
                 sampling_rate: int = 128,
                 random_seed: Optional[int] = None):
        """
        Args:
            mask_time_prob: Probability of masking time bins
            mask_freq_prob: Probability of masking frequency bins
            sampling_rate: Sampling frequency
            random_seed: Random seed
        """
        super().__init__("Time-Frequency Aug", sampling_rate=sampling_rate, random_seed=random_seed)
        self.mask_time_prob = mask_time_prob
        self.mask_freq_prob = mask_freq_prob
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                return_spectrogram: bool = False,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Apply time-frequency augmentation"""
        self.validate_data(data)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
        
        n_channels = data.shape[0]
        augmented_channels = []
        
        for ch in range(n_channels):
            # Compute STFT
            f, t, Zxx = scipy_signal.stft(data[ch], fs=self.sampling_rate, 
                                         nperseg=min(256, data.shape[1]))
            
            # Apply random masking in time-frequency domain
            Zxx_aug = Zxx.copy()
            
            # Time masking
            if np.random.rand() < self.mask_time_prob:
                n_time_bins = Zxx.shape[1]
                mask_width = max(1, int(n_time_bins * 0.1))
                mask_start = np.random.randint(0, max(1, n_time_bins - mask_width))
                Zxx_aug[:, mask_start:mask_start + mask_width] *= 0.1
            
            # Frequency masking
            if np.random.rand() < self.mask_freq_prob:
                n_freq_bins = Zxx.shape[0]
                mask_width = max(1, int(n_freq_bins * 0.1))
                mask_start = np.random.randint(0, max(1, n_freq_bins - mask_width))
                Zxx_aug[mask_start:mask_start + mask_width, :] *= 0.1
            
            if return_spectrogram:
                # Return magnitude spectrogram
                augmented_channels.append(np.abs(Zxx_aug))
            else:
                # Inverse STFT to get back time-domain signal
                _, x_aug = scipy_signal.istft(Zxx_aug, fs=self.sampling_rate)
                # Ensure same length as input
                if len(x_aug) < data.shape[1]:
                    x_aug = np.pad(x_aug, (0, data.shape[1] - len(x_aug)))
                elif len(x_aug) > data.shape[1]:
                    x_aug = x_aug[:data.shape[1]]
                augmented_channels.append(x_aug)
        
        augmented = np.array(augmented_channels)
        
        if was_1d and not return_spectrogram:
            augmented = augmented.flatten()
        
        return augmented, label


class MixupAugmentation(BaseAugmentation):
    """
    Mixup: Linear interpolation between two samples
    Contribution: 10-30% of augmented samples
    Reliability: MEDIUM-HIGH - improves generalization
    Caveat: Label interpolation may not make sense for all tasks
    """
    
    def __init__(self,
                 alpha: float = 0.2,
                 random_seed: Optional[int] = None):
        """
        Args:
            alpha: Beta distribution parameter (lower = more mixing)
            random_seed: Random seed
        """
        super().__init__("Mixup", AugmentationType.INTERMEDIATE, random_seed=random_seed)
        self.alpha = alpha
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                data2: Optional[np.ndarray] = None,
                label2: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """
        Apply mixup augmentation
        
        Args:
            data: First sample
            label: First label
            data2: Second sample (if None, must be provided in kwargs)
            label2: Second label (if None, must be provided in kwargs)
        """
        self.validate_data(data)
        
        if data2 is None:
            raise ValueError("Mixup requires a second sample. Provide data2 parameter.")
        
        self.validate_data(data2)
        
        # Ensure same shape
        if data.shape != data2.shape:
            raise ValueError(f"Shapes must match for mixup: {data.shape} vs {data2.shape}")
        
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix data
        augmented = lam * data + (1 - lam) * data2
        
        # Mix labels if both provided
        if label is not None and label2 is not None:
            if isinstance(label, (int, float)) and isinstance(label2, (int, float)):
                # Scalar labels - create soft label
                augmented_label = np.array([lam, 1 - lam])
                # Store original labels as well
                augmented_label = {'soft': augmented_label, 'hard1': label, 'hard2': label2, 'lambda': lam}
            else:
                # Already array labels
                augmented_label = lam * np.array(label) + (1 - lam) * np.array(label2)
        else:
            augmented_label = label
        
        return augmented, augmented_label


class CutMixAugmentation(BaseAugmentation):
    """
    CutMix: Cut and paste segments from one sample into another
    Contribution: 10-20% of augmented samples
    Reliability: MEDIUM - helps with local features
    """
    
    def __init__(self,
                 alpha: float = 1.0,
                 random_seed: Optional[int] = None):
        """
        Args:
            alpha: Beta distribution parameter
            random_seed: Random seed
        """
        super().__init__("CutMix", AugmentationType.INTERMEDIATE, random_seed=random_seed)
        self.alpha = alpha
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                data2: Optional[np.ndarray] = None,
                label2: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Apply CutMix augmentation"""
        self.validate_data(data)
        
        if data2 is None:
            raise ValueError("CutMix requires a second sample. Provide data2 parameter.")
        
        self.validate_data(data2)
        
        if data.shape != data2.shape:
            raise ValueError(f"Shapes must match: {data.shape} vs {data2.shape}")
        
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Determine cut size based on lambda
        if data.ndim == 1:
            cut_size = int(len(data) * (1 - lam))
            cut_start = np.random.randint(0, len(data) - cut_size + 1)
            
            augmented = data.copy()
            augmented[cut_start:cut_start + cut_size] = data2[cut_start:cut_start + cut_size]
        else:
            # For multi-channel, cut in time dimension
            n_samples = data.shape[1]
            cut_size = int(n_samples * (1 - lam))
            cut_start = np.random.randint(0, n_samples - cut_size + 1)
            
            augmented = data.copy()
            augmented[:, cut_start:cut_start + cut_size] = data2[:, cut_start:cut_start + cut_size]
        
        # Adjust label based on cut ratio
        if label is not None and label2 is not None:
            actual_ratio = 1 - (cut_size / (len(data) if data.ndim == 1 else data.shape[1]))
            if isinstance(label, (int, float)) and isinstance(label2, (int, float)):
                augmented_label = {'soft': np.array([actual_ratio, 1 - actual_ratio]), 
                                  'hard1': label, 'hard2': label2, 'lambda': actual_ratio}
            else:
                augmented_label = actual_ratio * np.array(label) + (1 - actual_ratio) * np.array(label2)
        else:
            augmented_label = label
        
        return augmented, augmented_label


# Convenience function
def create_intermediate_augmentations(sampling_rate: int = 128, random_seed: Optional[int] = None):
    """
    Create instances of all intermediate-level augmentations
    
    Args:
        sampling_rate: Sampling frequency for frequency-based augmentations
        random_seed: Random seed
        
    Returns:
        Dictionary of augmentation instances
    """
    return {
        'frequency_filtering': FrequencyFilteringAugmentation(
            sampling_rate=sampling_rate, random_seed=random_seed
        ),
        'timefreq_augmentation': TimeFrequencyAugmentation(
            sampling_rate=sampling_rate, random_seed=random_seed
        ),
        'mixup': MixupAugmentation(random_seed=random_seed),
        'cutmix': CutMixAugmentation(random_seed=random_seed),
    }


if __name__ == "__main__":
    # Test each augmentation
    print("Testing Intermediate Augmentations")
    print("=" * 80)
    
    # Create sample EEG data
    np.random.seed(42)
    sample_data1 = np.random.randn(32, 256)
    sample_data2 = np.random.randn(32, 256)
    sample_label1 = 0
    sample_label2 = 1
    
    # Test frequency filtering
    print("\nFREQUENCY FILTERING:")
    freq_aug = FrequencyFilteringAugmentation(sampling_rate=128, random_seed=42)
    aug_data, _ = freq_aug.augment(sample_data1, sample_label1)
    print(f"  Shape: {aug_data.shape}")
    print(f"  Std: {np.std(aug_data):.4f}")
    
    # Test time-frequency
    print("\nTIME-FREQUENCY AUGMENTATION:")
    tf_aug = TimeFrequencyAugmentation(sampling_rate=128, random_seed=42)
    aug_data, _ = tf_aug.augment(sample_data1, sample_label1)
    print(f"  Shape: {aug_data.shape}")
    
    # Test mixup
    print("\nMIXUP:")
    mixup_aug = MixupAugmentation(random_seed=42)
    aug_data, aug_label = mixup_aug.augment(sample_data1, sample_label1, 
                                            data2=sample_data2, label2=sample_label2)
    print(f"  Shape: {aug_data.shape}")
    print(f"  Label: {aug_label}")
    
    # Test cutmix
    print("\nCUTMIX:")
    cutmix_aug = CutMixAugmentation(random_seed=42)
    aug_data, aug_label = cutmix_aug.augment(sample_data1, sample_label1,
                                             data2=sample_data2, label2=sample_label2)
    print(f"  Shape: {aug_data.shape}")
    print(f"  Label: {aug_label}")
