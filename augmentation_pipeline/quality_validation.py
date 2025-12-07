"""
Validation and quality checks for augmented EEG data
Ensures augmented data maintains physiological properties and doesn't introduce artifacts
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist


class DataQualityValidator:
    """
    Validate quality of augmented EEG data
    
    Implements checks recommended in literature:
    - Amplitude preservation
    - Spectral similarity
    - Statistical distribution matching
    - Maximum Mean Discrepancy (MMD)
    """
    
    def __init__(self, 
                 max_amplitude_change: float = 3.0,
                 spectral_similarity_threshold: float = 0.7):
        """
        Args:
            max_amplitude_change: Maximum allowed std deviation change (in multiples)
            spectral_similarity_threshold: Minimum spectral correlation required
        """
        self.max_amplitude_change = max_amplitude_change
        self.spectral_similarity_threshold = spectral_similarity_threshold
        
        self.validation_results = []
    
    def validate_amplitude(self, 
                          original: np.ndarray,
                          augmented: np.ndarray) -> Dict[str, float]:
        """
        Check if amplitude is preserved
        
        Args:
            original: Original data
            augmented: Augmented data
            
        Returns:
            Dictionary with amplitude metrics
        """
        orig_std = np.std(original)
        aug_std = np.std(augmented)
        
        std_ratio = aug_std / (orig_std + 1e-10)
        
        results = {
            'original_std': float(orig_std),
            'augmented_std': float(aug_std),
            'std_ratio': float(std_ratio),
            'amplitude_preserved': abs(std_ratio - 1.0) < self.max_amplitude_change,
        }
        
        return results
    
    def validate_spectral_similarity(self,
                                    original: np.ndarray,
                                    augmented: np.ndarray,
                                    fs: int = 128) -> Dict[str, float]:
        """
        Check spectral similarity using power spectral density
        
        Args:
            original: Original data
            augmented: Augmented data
            fs: Sampling frequency
            
        Returns:
            Dictionary with spectral metrics
        """
        # Flatten if multi-channel
        orig_flat = original.flatten()
        aug_flat = augmented.flatten()
        
        # Compute FFT
        fft_orig = np.abs(np.fft.rfft(orig_flat))
        fft_aug = np.abs(np.fft.rfft(aug_flat))
        
        # Ensure same length
        min_len = min(len(fft_orig), len(fft_aug))
        fft_orig = fft_orig[:min_len]
        fft_aug = fft_aug[:min_len]
        
        # Compute correlation
        correlation = np.corrcoef(fft_orig, fft_aug)[0, 1]
        
        # Compute relative power in different bands
        freqs = np.fft.rfftfreq(len(orig_flat), 1/fs)[:min_len]
        
        def band_power(fft_vals, freqs, low, high):
            mask = (freqs >= low) & (freqs < high)
            return np.sum(fft_vals[mask]**2)
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
        }
        
        band_similarity = {}
        for band_name, (low, high) in bands.items():
            orig_power = band_power(fft_orig, freqs, low, high)
            aug_power = band_power(fft_aug, freqs, low, high)
            ratio = aug_power / (orig_power + 1e-10)
            band_similarity[f'{band_name}_power_ratio'] = float(ratio)
        
        results = {
            'spectral_correlation': float(correlation),
            'spectral_similar': correlation >= self.spectral_similarity_threshold,
            **band_similarity,
        }
        
        return results
    
    def validate_statistical_distribution(self,
                                         original: np.ndarray,
                                         augmented: np.ndarray) -> Dict[str, float]:
        """
        Check if statistical distributions match using KS test
        
        Args:
            original: Original data
            augmented: Augmented data
            
        Returns:
            Dictionary with distribution metrics
        """
        orig_flat = original.flatten()
        aug_flat = augmented.flatten()
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(orig_flat, aug_flat)
        
        # Mean and variance
        orig_mean = np.mean(orig_flat)
        aug_mean = np.mean(aug_flat)
        orig_var = np.var(orig_flat)
        aug_var = np.var(aug_flat)
        
        results = {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'distributions_similar': ks_pvalue > 0.05,  # p > 0.05 means similar
            'original_mean': float(orig_mean),
            'augmented_mean': float(aug_mean),
            'mean_shift': float(abs(aug_mean - orig_mean)),
            'original_variance': float(orig_var),
            'augmented_variance': float(aug_var),
            'variance_ratio': float(aug_var / (orig_var + 1e-10)),
        }
        
        return results
    
    def compute_mmd(self,
                   original: np.ndarray,
                   augmented: np.ndarray,
                   kernel: str = 'rbf',
                   gamma: float = 1.0) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between original and augmented data
        
        Lower MMD = more similar distributions
        
        Args:
            original: Original data (n_samples, n_features)
            augmented: Augmented data (n_samples, n_features)
            kernel: Kernel type ('rbf' or 'linear')
            gamma: RBF kernel parameter
            
        Returns:
            MMD value
        """
        # Reshape to 2D if needed
        if original.ndim > 2:
            original = original.reshape(len(original), -1)
        if augmented.ndim > 2:
            augmented = augmented.reshape(len(augmented), -1)
        
        n = len(original)
        m = len(augmented)
        
        if kernel == 'rbf':
            # RBF kernel
            def rbf_kernel(X, Y):
                dists = cdist(X, Y, 'sqeuclidean')
                return np.exp(-gamma * dists)
            
            K_XX = rbf_kernel(original, original)
            K_YY = rbf_kernel(augmented, augmented)
            K_XY = rbf_kernel(original, augmented)
        else:
            # Linear kernel
            K_XX = np.dot(original, original.T)
            K_YY = np.dot(augmented, augmented.T)
            K_XY = np.dot(original, augmented.T)
        
        # Compute MMD^2
        mmd_squared = (np.sum(K_XX) / (n * n) + 
                      np.sum(K_YY) / (m * m) - 
                      2 * np.sum(K_XY) / (n * m))
        
        mmd = np.sqrt(max(mmd_squared, 0))
        
        return float(mmd)
    
    def full_validation(self,
                       original: np.ndarray,
                       augmented: np.ndarray,
                       fs: int = 128) -> Dict[str, any]:
        """
        Run all validation checks
        
        Args:
            original: Original data
            augmented: Augmented data
            fs: Sampling frequency
            
        Returns:
            Dictionary with all validation results
        """
        results = {}
        
        # Amplitude check
        results['amplitude'] = self.validate_amplitude(original, augmented)
        
        # Spectral similarity
        results['spectral'] = self.validate_spectral_similarity(original, augmented, fs)
        
        # Statistical distribution
        results['distribution'] = self.validate_statistical_distribution(original, augmented)
        
        # MMD (sample a subset if data is large)
        if len(original.shape) > 1 and original.shape[0] > 100:
            # Sample subset for MMD computation
            sample_size = min(100, original.shape[0])
            orig_sample = original[:sample_size]
            aug_sample = augmented[:sample_size]
        else:
            orig_sample = original
            aug_sample = augmented
        
        results['mmd'] = self.compute_mmd(orig_sample, aug_sample)
        
        # Overall quality score (0-1, higher is better)
        quality_score = 0.0
        if results['amplitude']['amplitude_preserved']:
            quality_score += 0.25
        if results['spectral']['spectral_similar']:
            quality_score += 0.25
        if results['distribution']['distributions_similar']:
            quality_score += 0.25
        if results['mmd'] < 0.5:  # Arbitrary threshold
            quality_score += 0.25
        
        results['overall_quality_score'] = quality_score
        results['quality_passed'] = quality_score >= 0.5
        
        return results
    
    def validate_batch(self,
                      original_batch: np.ndarray,
                      augmented_batch: np.ndarray,
                      fs: int = 128,
                      n_samples: int = 10) -> Dict[str, any]:
        """
        Validate a batch of augmented samples
        
        Args:
            original_batch: Batch of original data (n_samples, ...)
            augmented_batch: Batch of augmented data (n_samples, ...)
            fs: Sampling frequency
            n_samples: Number of samples to validate
            
        Returns:
            Aggregated validation results
        """
        n_validate = min(n_samples, len(original_batch), len(augmented_batch))
        
        all_results = []
        for i in range(n_validate):
            results = self.full_validation(original_batch[i], augmented_batch[i], fs)
            all_results.append(results)
        
        # Aggregate results
        aggregated = {
            'n_samples_validated': n_validate,
            'pass_rate': sum(r['quality_passed'] for r in all_results) / n_validate,
            'avg_quality_score': np.mean([r['overall_quality_score'] for r in all_results]),
            'avg_mmd': np.mean([r['mmd'] for r in all_results]),
            'amplitude_pass_rate': sum(r['amplitude']['amplitude_preserved'] for r in all_results) / n_validate,
            'spectral_pass_rate': sum(r['spectral']['spectral_similar'] for r in all_results) / n_validate,
            'distribution_pass_rate': sum(r['distribution']['distributions_similar'] for r in all_results) / n_validate,
        }
        
        return aggregated
    
    def print_validation_report(self, results: Dict[str, any]):
        """Print human-readable validation report"""
        print("\n" + "=" * 80)
        print("DATA QUALITY VALIDATION REPORT")
        print("=" * 80)
        
        if 'n_samples_validated' in results:
            # Batch results
            print(f"Samples validated: {results['n_samples_validated']}")
            print(f"Overall pass rate: {results['pass_rate']*100:.1f}%")
            print(f"Average quality score: {results['avg_quality_score']:.3f}")
            print(f"Average MMD: {results['avg_mmd']:.4f}")
            print("\nDetailed pass rates:")
            print(f"  Amplitude preservation: {results['amplitude_pass_rate']*100:.1f}%")
            print(f"  Spectral similarity: {results['spectral_pass_rate']*100:.1f}%")
            print(f"  Distribution matching: {results['distribution_pass_rate']*100:.1f}%")
        else:
            # Single sample results
            print(f"Overall quality score: {results['overall_quality_score']:.3f}")
            print(f"Quality passed: {'   YES' if results['quality_passed'] else '✗ NO'}")
            print(f"MMD: {results['mmd']:.4f}")
            
            print("\nAmplitude Check:")
            print(f"  Original std: {results['amplitude']['original_std']:.4f}")
            print(f"  Augmented std: {results['amplitude']['augmented_std']:.4f}")
            print(f"  Std ratio: {results['amplitude']['std_ratio']:.4f}")
            print(f"  Passed: {'  ' if results['amplitude']['amplitude_preserved'] else '✗'}")
            
            print("\nSpectral Similarity:")
            print(f"  Correlation: {results['spectral']['spectral_correlation']:.4f}")
            print(f"  Passed: {'  ' if results['spectral']['spectral_similar'] else '✗'}")
            
            print("\nDistribution Matching:")
            print(f"  KS statistic: {results['distribution']['ks_statistic']:.4f}")
            print(f"  KS p-value: {results['distribution']['ks_pvalue']:.4f}")
            print(f"  Passed: {'  ' if results['distribution']['distributions_similar'] else '✗'}")
        
        print("=" * 80)


def visualize_augmentation_quality(original: np.ndarray,
                                   augmented: np.ndarray,
                                   fs: int = 128,
                                   save_path: Optional[str] = None):
    """
    Visualize quality comparison between original and augmented data
    
    Args:
        original: Original data (single sample)
        augmented: Augmented data (single sample)
        fs: Sampling frequency
        save_path: Path to save figure (if None, display instead)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain comparison (first channel if multi-channel)
    if original.ndim > 1:
        orig_signal = original[0]
        aug_signal = augmented[0]
    else:
        orig_signal = original
        aug_signal = augmented
    
    time = np.arange(len(orig_signal)) / fs
    
    axes[0, 0].plot(time, orig_signal, label='Original', alpha=0.7)
    axes[0, 0].plot(time, aug_signal, label='Augmented', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Time Domain Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Power spectrum
    fft_orig = np.abs(np.fft.rfft(orig_signal))
    fft_aug = np.abs(np.fft.rfft(aug_signal))
    freqs = np.fft.rfftfreq(len(orig_signal), 1/fs)
    
    axes[0, 1].plot(freqs, fft_orig, label='Original', alpha=0.7)
    axes[0, 1].plot(freqs, fft_aug, label='Augmented', alpha=0.7)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_title('Power Spectrum')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, min(50, fs/2)])
    
    # Distribution comparison
    axes[1, 0].hist(orig_signal, bins=50, alpha=0.5, label='Original', density=True)
    axes[1, 0].hist(aug_signal, bins=50, alpha=0.5, label='Augmented', density=True)
    axes[1, 0].set_xlabel('Amplitude')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Amplitude Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Difference signal
    if orig_signal.shape == aug_signal.shape:
        diff = aug_signal - orig_signal
        axes[1, 1].plot(time, diff)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Difference')
        axes[1, 1].set_title('Augmented - Original')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test validation
    print("Testing Data Quality Validator")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    original = np.random.randn(32, 256)  # 32 channels, 256 samples
    
    # Create augmented data (with small noise)
    augmented = original + np.random.randn(32, 256) * 0.1
    
    # Validate
    validator = DataQualityValidator()
    results = validator.full_validation(original, augmented, fs=128)
    validator.print_validation_report(results)
    
    # Test batch validation
    print("\n\nTesting Batch Validation")
    print("=" * 80)
    
    original_batch = np.random.randn(10, 32, 256)
    augmented_batch = original_batch + np.random.randn(10, 32, 256) * 0.1
    
    batch_results = validator.validate_batch(original_batch, augmented_batch, fs=128)
    validator.print_validation_report(batch_results)
