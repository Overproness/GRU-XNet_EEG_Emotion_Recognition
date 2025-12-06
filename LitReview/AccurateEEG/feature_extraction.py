"""
Feature Extraction Module for EEG Signals
Implements: Statistical Features, Hjorth Parameters, Spectral Entropy, RMS, and Wavelet Packet Decomposition
"""

import numpy as np
import pywt
from scipy import signal
from scipy.stats import skew, kurtosis


class FeatureExtractor:
    """Extract features from EEG signals as described in the paper"""
    
    def __init__(self, sampling_rate=128, wavelet='db4', level=4):
        """
        Initialize feature extractor
        
        Args:
            sampling_rate: Sampling rate of EEG signal (Hz)
            wavelet: Wavelet type for WPD
            level: Decomposition level for WPD
        """
        self.sampling_rate = sampling_rate
        self.wavelet = wavelet
        self.level = level
        
    def extract_statistical_features(self, signal_data):
        """
        Extract statistical features: mean, variance, std, skewness, kurtosis
        
        Args:
            signal_data: 1D numpy array of EEG signal
            
        Returns:
            Dictionary of statistical features
        """
        features = {
            'mean': np.mean(signal_data),
            'variance': np.var(signal_data),
            'std': np.std(signal_data),
            'skewness': skew(signal_data),
            'kurtosis': kurtosis(signal_data)
        }
        return features
    
    def extract_hjorth_parameters(self, signal_data):
        """
        Extract Hjorth parameters: Activity, Mobility, Complexity
        
        Args:
            signal_data: 1D numpy array of EEG signal
            
        Returns:
            Dictionary of Hjorth parameters
        """
        # Activity (variance of signal)
        activity = np.var(signal_data)
        
        # First derivative
        first_deriv = np.diff(signal_data)
        
        # Mobility (square root of variance of first derivative / variance of signal)
        if activity > 0:
            mobility = np.sqrt(np.var(first_deriv) / activity)
        else:
            mobility = 0.0
        
        # Second derivative
        second_deriv = np.diff(first_deriv)
        
        # Complexity (mobility of first derivative / mobility of signal)
        var_first = np.var(first_deriv)
        var_second = np.var(second_deriv)
        
        if var_first > 0 and mobility > 0:
            mobility_deriv = np.sqrt(var_second / var_first)
            complexity = mobility_deriv / mobility
        else:
            complexity = 0.0
        
        features = {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }
        return features
    
    def extract_spectral_entropy(self, signal_data):
        """
        Extract spectral entropy
        
        Args:
            signal_data: 1D numpy array of EEG signal
            
        Returns:
            Spectral entropy value
        """
        # Compute power spectral density
        freqs, psd = signal.periodogram(signal_data, fs=self.sampling_rate)
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # Calculate spectral entropy
        # Remove zeros to avoid log(0)
        psd_norm = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        
        return {'spectral_entropy': spectral_entropy}
    
    def extract_rms(self, signal_data):
        """
        Extract Root Mean Square (RMS)
        
        Args:
            signal_data: 1D numpy array of EEG signal
            
        Returns:
            RMS value
        """
        rms = np.sqrt(np.mean(signal_data ** 2))
        return {'rms': rms}
    
    def extract_band_powers(self, signal_data):
        """
        Extract power in different frequency bands (delta, theta, alpha, beta, gamma)
        
        Args:
            signal_data: 1D numpy array of EEG signal
            
        Returns:
            Dictionary of band powers
        """
        # Compute power spectral density
        freqs, psd = signal.periodogram(signal_data, fs=self.sampling_rate)
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 64)
        }
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in bands.items():
            # Find indices for frequency band
            idx = np.logical_and(freqs >= low_freq, freqs < high_freq)
            # Calculate band power
            band_powers[f'{band_name}_power'] = np.trapz(psd[idx], freqs[idx])
        
        return band_powers
    
    def extract_wpd_features(self, signal_data):
        """
        Extract features using Wavelet Packet Decomposition
        
        Args:
            signal_data: 1D numpy array of EEG signal
            
        Returns:
            Dictionary of WPD features
        """
        # Perform wavelet packet decomposition
        wp = pywt.WaveletPacket(data=signal_data, wavelet=self.wavelet, 
                                mode='symmetric', maxlevel=self.level)
        
        # Get all nodes at the specified level
        nodes = [node.path for node in wp.get_level(self.level, 'freq')]
        
        wpd_features = {}
        for i, node in enumerate(nodes):
            # Get coefficients for this node
            coeffs = wp[node].data
            
            # Extract statistical features from coefficients
            wpd_features[f'wpd_node_{i}_mean'] = np.mean(coeffs)
            wpd_features[f'wpd_node_{i}_std'] = np.std(coeffs)
            wpd_features[f'wpd_node_{i}_energy'] = np.sum(coeffs ** 2)
        
        return wpd_features
    
    def extract_all_features(self, signal_data):
        """
        Extract all features from EEG signal
        
        Args:
            signal_data: 1D numpy array of EEG signal
            
        Returns:
            Dictionary containing all features
        """
        all_features = {}
        
        # Statistical features
        all_features.update(self.extract_statistical_features(signal_data))
        
        # Hjorth parameters
        all_features.update(self.extract_hjorth_parameters(signal_data))
        
        # Spectral entropy
        all_features.update(self.extract_spectral_entropy(signal_data))
        
        # RMS
        all_features.update(self.extract_rms(signal_data))
        
        # Band powers
        all_features.update(self.extract_band_powers(signal_data))
        
        # Wavelet packet decomposition
        all_features.update(self.extract_wpd_features(signal_data))
        
        return all_features
    
    def extract_features_multi_channel(self, signal_array):
        """
        Extract features from multi-channel EEG data
        
        Args:
            signal_array: 2D numpy array (channels x samples)
            
        Returns:
            1D numpy array of concatenated features from all channels
        """
        all_channel_features = []
        
        for channel_idx in range(signal_array.shape[0]):
            channel_signal = signal_array[channel_idx, :]
            channel_features = self.extract_all_features(channel_signal)
            
            # Convert dictionary to ordered list of values
            feature_values = list(channel_features.values())
            all_channel_features.extend(feature_values)
        
        return np.array(all_channel_features)


def get_feature_names(num_channels=14, wavelet_level=4):
    """
    Get feature names for all extracted features
    
    Args:
        num_channels: Number of EEG channels
        wavelet_level: Decomposition level for WPD
        
    Returns:
        List of feature names
    """
    # Base features per channel
    base_features = [
        'mean', 'variance', 'std', 'skewness', 'kurtosis',
        'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
        'spectral_entropy', 'rms',
        'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
    ]
    
    # WPD features (2^level nodes, 3 features per node)
    num_wpd_nodes = 2 ** wavelet_level
    wpd_features = []
    for i in range(num_wpd_nodes):
        wpd_features.extend([
            f'wpd_node_{i}_mean',
            f'wpd_node_{i}_std',
            f'wpd_node_{i}_energy'
        ])
    
    # Combine all features
    all_features_per_channel = base_features + wpd_features
    
    # Create feature names for all channels
    feature_names = []
    for ch_idx in range(num_channels):
        for feat_name in all_features_per_channel:
            feature_names.append(f'ch{ch_idx}_{feat_name}')
    
    return feature_names
