"""
Preprocessing module for gru_xnet
Handles STFT transformation and data normalization
"""

import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional, Dict
import warnings


class STFTPreprocessor:
    """
    Short-Time Fourier Transform preprocessor for EEG signals
    Converts time-domain signals to time-frequency representations
    """
    
    def __init__(
        self,
        sampling_rate: int,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        window: str = 'hann',
        freq_range: Optional[Tuple[float, float]] = None
    ):
        """
        Args:
            sampling_rate: Sampling rate in Hz
            nperseg: Length of each segment for STFT
            noverlap: Number of points to overlap between segments
            nfft: Length of FFT (default: nperseg)
            window: Window function ('hann', 'hamming', 'blackman', etc.)
            freq_range: (min_freq, max_freq) to keep, or None for all frequencies
        """
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.nfft = nfft if nfft is not None else nperseg
        self.window = window
        self.freq_range = freq_range
        
        # Calculate expected output dimensions
        self._calculate_output_shape()
    
    def _calculate_output_shape(self):
        """Calculate expected STFT output dimensions"""
        # Number of frequency bins
        self.n_freq_bins = self.nfft // 2 + 1
        
        # If frequency range is specified, update n_freq_bins
        if self.freq_range is not None:
            freq_resolution = self.sampling_rate / self.nfft
            min_idx = int(self.freq_range[0] / freq_resolution)
            max_idx = int(self.freq_range[1] / freq_resolution)
            self.n_freq_bins = max_idx - min_idx + 1
            self.freq_start_idx = min_idx
            self.freq_end_idx = max_idx + 1
        else:
            self.freq_start_idx = 0
            self.freq_end_idx = self.n_freq_bins
    
    def compute_stft(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Compute STFT for a single channel
        
        Args:
            signal_data: 1D array of EEG signal
            
        Returns:
            stft_magnitude: 2D array (n_freq_bins, n_time_bins)
        """
        # Compute STFT
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.sampling_rate,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft
        )
        
        # Get magnitude spectrum
        stft_magnitude = np.abs(Zxx)
        
        # Apply frequency range filter if specified
        stft_magnitude = stft_magnitude[self.freq_start_idx:self.freq_end_idx, :]
        
        return stft_magnitude
    
    def transform(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Transform EEG data to STFT representation
        
        Args:
            eeg_data: (n_channels, n_timepoints) or (n_samples, n_channels, n_timepoints)
            
        Returns:
            stft_features: (n_channels, n_freq_bins, n_time_bins) or 
                          (n_samples, n_channels, n_freq_bins, n_time_bins)
        """
        if eeg_data.ndim == 2:
            # Single sample: (n_channels, n_timepoints)
            return self._transform_single(eeg_data)
        elif eeg_data.ndim == 3:
            # Multiple samples: (n_samples, n_channels, n_timepoints)
            return self._transform_batch(eeg_data)
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {eeg_data.shape}")
    
    def _transform_single(self, eeg_data: np.ndarray) -> np.ndarray:
        """Transform a single sample"""
        n_channels, n_timepoints = eeg_data.shape
        
        # Initialize output
        # We need to compute one to get n_time_bins
        stft_temp = self.compute_stft(eeg_data[0])
        n_freq_bins, n_time_bins = stft_temp.shape
        
        stft_features = np.zeros((n_channels, n_freq_bins, n_time_bins), dtype=np.float32)
        stft_features[0] = stft_temp
        
        # Compute STFT for each channel
        for ch in range(1, n_channels):
            stft_features[ch] = self.compute_stft(eeg_data[ch])
        
        return stft_features
    
    def _transform_batch(self, eeg_data: np.ndarray) -> np.ndarray:
        """Transform a batch of samples"""
        n_samples, n_channels, n_timepoints = eeg_data.shape
        
        # Compute STFT for first sample to get dimensions
        stft_temp = self._transform_single(eeg_data[0])
        n_freq_bins, n_time_bins = stft_temp.shape[1], stft_temp.shape[2]
        
        # Initialize output
        stft_features = np.zeros(
            (n_samples, n_channels, n_freq_bins, n_time_bins),
            dtype=np.float32
        )
        stft_features[0] = stft_temp
        
        # Transform each sample
        for i in range(1, n_samples):
            stft_features[i] = self._transform_single(eeg_data[i])
        
        return stft_features
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Calculate output shape for given input shape
        
        Args:
            input_shape: (n_channels, n_timepoints) or (n_samples, n_channels, n_timepoints)
            
        Returns:
            output_shape: Corresponding output shape
        """
        if len(input_shape) == 2:
            n_channels, n_timepoints = input_shape
            # Estimate n_time_bins
            n_time_bins = (n_timepoints - self.noverlap) // (self.nperseg - self.noverlap)
            return (n_channels, self.n_freq_bins, n_time_bins)
        elif len(input_shape) == 3:
            n_samples, n_channels, n_timepoints = input_shape
            n_time_bins = (n_timepoints - self.noverlap) // (self.nperseg - self.noverlap)
            return (n_samples, n_channels, self.n_freq_bins, n_time_bins)
        else:
            raise ValueError(f"Expected 2D or 3D shape, got {input_shape}")


class MultiDatasetSTFTPreprocessor:
    """
    Preprocessor that handles multiple datasets with different sampling rates
    """
    
    def __init__(
        self,
        dataset_configs: Dict[str, Dict],
        target_freq_bins: int = 129,
        target_time_bins: int = 126
    ):
        """
        Args:
            dataset_configs: Dict mapping dataset name to config dict
                            e.g., {'DEAP': {'sampling_rate': 128, 'nperseg': 256}}
            target_freq_bins: Target number of frequency bins (for standardization)
            target_time_bins: Target number of time bins (for standardization)
        """
        self.dataset_configs = dataset_configs
        self.target_freq_bins = target_freq_bins
        self.target_time_bins = target_time_bins
        
        # Create preprocessor for each dataset
        self.preprocessors = {}
        for dataset_name, config in dataset_configs.items():
            self.preprocessors[dataset_name] = STFTPreprocessor(**config)
    
    def transform(
        self,
        eeg_data: np.ndarray,
        dataset_name: str,
        standardize: bool = True
    ) -> np.ndarray:
        """
        Transform EEG data using dataset-specific STFT
        
        Args:
            eeg_data: EEG data to transform
            dataset_name: Name of dataset ('DEAP', 'GAMEEMO', 'SEEDIV')
            standardize: Whether to standardize output shape
            
        Returns:
            stft_features: STFT representation
        """
        if dataset_name not in self.preprocessors:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Apply STFT
        stft_features = self.preprocessors[dataset_name].transform(eeg_data)
        
        # Standardize shape if requested
        if standardize:
            stft_features = self._standardize_shape(stft_features)
        
        return stft_features
    
    def _standardize_shape(self, stft_features: np.ndarray) -> np.ndarray:
        """
        Standardize STFT output to target shape using interpolation
        
        Args:
            stft_features: (..., n_freq_bins, n_time_bins)
            
        Returns:
            standardized: (..., target_freq_bins, target_time_bins)
        """
        from scipy.ndimage import zoom
        
        *batch_dims, n_freq_bins, n_time_bins = stft_features.shape
        
        if n_freq_bins == self.target_freq_bins and n_time_bins == self.target_time_bins:
            return stft_features
        
        # Calculate zoom factors
        freq_zoom = self.target_freq_bins / n_freq_bins
        time_zoom = self.target_time_bins / n_time_bins
        
        # Reshape to 3D for processing
        original_shape = stft_features.shape
        if len(batch_dims) > 1:
            stft_features = stft_features.reshape(-1, n_freq_bins, n_time_bins)
        
        # Apply zoom to each sample
        standardized = []
        for i in range(stft_features.shape[0]):
            zoomed = zoom(
                stft_features[i],
                (freq_zoom, time_zoom),
                order=1  # Linear interpolation
            )
            standardized.append(zoomed)
        
        standardized = np.array(standardized)
        
        # Reshape back to original batch dimensions
        if len(batch_dims) > 1:
            new_shape = batch_dims + [self.target_freq_bins, self.target_time_bins]
            standardized = standardized.reshape(new_shape)
        
        return standardized


class EEGNormalizer:
    """
    Normalize EEG data (can be applied before or after STFT)
    """
    
    def __init__(self, method: str = 'zscore', axis: Optional[int] = None):
        """
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust')
            axis: Axis along which to normalize (None for global)
        """
        self.method = method
        self.axis = axis
        self.stats = {}
    
    def fit(self, data: np.ndarray):
        """Compute normalization statistics"""
        if self.method == 'zscore':
            self.stats['mean'] = np.mean(data, axis=self.axis, keepdims=True)
            self.stats['std'] = np.std(data, axis=self.axis, keepdims=True)
            # Avoid division by zero
            self.stats['std'] = np.where(
                self.stats['std'] < 1e-8,
                1.0,
                self.stats['std']
            )
        elif self.method == 'minmax':
            self.stats['min'] = np.min(data, axis=self.axis, keepdims=True)
            self.stats['max'] = np.max(data, axis=self.axis, keepdims=True)
            # Avoid division by zero
            range_val = self.stats['max'] - self.stats['min']
            self.stats['range'] = np.where(range_val < 1e-8, 1.0, range_val)
        elif self.method == 'robust':
            self.stats['median'] = np.median(data, axis=self.axis, keepdims=True)
            q75, q25 = np.percentile(data, [75, 25], axis=self.axis, keepdims=True)
            self.stats['iqr'] = q75 - q25
            self.stats['iqr'] = np.where(self.stats['iqr'] < 1e-8, 1.0, self.stats['iqr'])
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        if not self.stats:
            warnings.warn("Normalizer not fitted, fitting on provided data")
            self.fit(data)
        
        if self.method == 'zscore':
            return (data - self.stats['mean']) / self.stats['std']
        elif self.method == 'minmax':
            return (data - self.stats['min']) / self.stats['range']
        elif self.method == 'robust':
            return (data - self.stats['median']) / self.stats['iqr']
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)


def create_dataset_stft_configs(
    target_freq_range: Tuple[float, float] = (0.5, 50.0)
) -> Dict[str, Dict]:
    """
    Create STFT configurations for each dataset
    
    Args:
        target_freq_range: (min_freq, max_freq) in Hz to keep
        
    Returns:
        configs: Dictionary of STFT configurations
    """
    configs = {
        'DEAP': {
            'sampling_rate': 128,
            'nperseg': 256,
            'noverlap': 128,
            'nfft': 256,
            'window': 'hann',
            'freq_range': target_freq_range
        },
        'GAMEEMO': {
            'sampling_rate': 128,
            'nperseg': 128,  # Shorter window for shorter signals
            'noverlap': 64,
            'nfft': 128,
            'window': 'hann',
            'freq_range': target_freq_range
        },
        'SEEDIV': {
            'sampling_rate': 200,
            'nperseg': 400,
            'noverlap': 200,
            'nfft': 400,
            'window': 'hann',
            'freq_range': target_freq_range
        }
    }
    
    return configs


if __name__ == "__main__":
    # Test STFT preprocessor
    print("Testing STFT Preprocessor...")
    
    # Simulate EEG data
    n_channels = 32
    n_timepoints = 8064
    sampling_rate = 128
    
    # Single sample
    eeg_single = np.random.randn(n_channels, n_timepoints)
    
    # Create preprocessor
    preprocessor = STFTPreprocessor(
        sampling_rate=sampling_rate,
        nperseg=256,
        noverlap=128,
        freq_range=(0.5, 50.0)
    )
    
    # Transform
    stft_single = preprocessor.transform(eeg_single)
    print(f"Single sample - Input: {eeg_single.shape}, Output: {stft_single.shape}")
    
    # Batch of samples
    n_samples = 10
    eeg_batch = np.random.randn(n_samples, n_channels, n_timepoints)
    stft_batch = preprocessor.transform(eeg_batch)
    print(f"Batch - Input: {eeg_batch.shape}, Output: {stft_batch.shape}")
    
    # Test multi-dataset preprocessor
    print("\nTesting Multi-Dataset Preprocessor...")
    configs = create_dataset_stft_configs()
    multi_preprocessor = MultiDatasetSTFTPreprocessor(configs)
    
    # Test DEAP
    deap_data = np.random.randn(5, 32, 8064)
    deap_stft = multi_preprocessor.transform(deap_data, 'DEAP', standardize=True)
    print(f"DEAP - Input: {deap_data.shape}, Output: {deap_stft.shape}")
    
    # Test GAMEEMO
    gameemo_data = np.random.randn(5, 14, 640)
    gameemo_stft = multi_preprocessor.transform(gameemo_data, 'GAMEEMO', standardize=True)
    print(f"GAMEEMO - Input: {gameemo_data.shape}, Output: {gameemo_stft.shape}")
    
    # Test SEEDIV
    seediv_data = np.random.randn(5, 62, 28000)
    seediv_stft = multi_preprocessor.transform(seediv_data, 'SEEDIV', standardize=True)
    print(f"SEEDIV - Input: {seediv_data.shape}, Output: {seediv_stft.shape}")
    
    print("\nAll tests passed!")
