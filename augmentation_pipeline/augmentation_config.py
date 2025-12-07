"""
Configuration for EEG Data Augmentation Pipeline
Defines augmentation techniques, their parameters, and contribution ratios
Target: 1:2 ratio (original:augmented) - 32 original → 64 augmented samples
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline"""
    
    # Target augmentation ratio (for 32 originals → 64 augmented)
    target_augmentation_ratio: float = 2.0
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # ==================== BEGINNER AUGMENTATIONS (40-60% of augmented) ====================
    
    # Additive Gaussian Noise (15-35% contribution)
    use_gaussian_noise: bool = True
    gaussian_noise_contribution: float = 0.50  # 50% of augmented samples (25% of total 2.0)
    gaussian_noise_snr_range: Tuple[float, float] = (0.001, 0.05)  # std multiplier
    
    # Time Shift (10-25% contribution)
    use_time_shift: bool = True
    time_shift_contribution: float = 0.30  # 30% of augmented samples (15% of total 2.0)
    time_shift_range: Tuple[float, float] = (0.05, 0.20)  # ±5-20% of window length
    
    # Window Slicing/Cropping (10-25% contribution)
    use_window_slicing: bool = True
    window_slicing_contribution: float = 0.30  # 30% of augmented samples (15% of total 2.0)
    window_crop_range: Tuple[float, float] = (0.80, 0.95)  # keep 80-95% of window
    
    # Amplitude Scaling (5-15% contribution)
    use_amplitude_scaling: bool = True
    amplitude_scaling_contribution: float = 0.20  # 20% of augmented samples (10% of total 2.0)
    amplitude_scaling_range: Tuple[float, float] = (0.90, 1.10)  # ±10% scaling
    
    # Channel Dropout (5-15% contribution)
    use_channel_dropout: bool = True
    channel_dropout_contribution: float = 0.20  # 20% of augmented samples (10% of total 2.0)
    channel_dropout_rate: Tuple[float, float] = (0.05, 0.20)  # drop 5-20% channels
    
    # ==================== INTERMEDIATE AUGMENTATIONS (20-30% of augmented) ====================
    
    # Frequency Filtering (5-15% contribution)
    use_frequency_filtering: bool = True
    frequency_filtering_contribution: float = 0.16  # 16% of augmented samples (8% of total 2.0)
    freq_shift_range: Tuple[float, float] = (1.0, 2.0)  # ±1-2 Hz shift
    
    # Time-Frequency Augmentation (10-30% contribution)
    use_timefreq_augmentation: bool = True
    timefreq_augmentation_contribution: float = 0.24  # 24% of augmented samples (12% of total 2.0)
    timefreq_mask_time_prob: float = 0.1  # probability of masking time bins
    timefreq_mask_freq_prob: float = 0.1  # probability of masking freq bins
    
    # Mixup (10-30% contribution)
    use_mixup: bool = True
    mixup_contribution: float = 0.30  # 30% of augmented samples (15% of total 2.0)
    mixup_alpha: float = 0.2  # Beta distribution parameter
    
    # CutMix (10-20% contribution) - DISABLED to maintain 2.0 ratio
    use_cutmix: bool = False
    cutmix_contribution: float = 0.00  # 0% (disabled to keep total at 2.0)
    cutmix_alpha: float = 1.0  # Beta distribution parameter
    
    # ==================== ADVANCED AUGMENTATIONS (10-25% of augmented) ====================
    
    # SMOTE in feature space (10-30% contribution, minority classes only)
    use_smote: bool = True
    smote_contribution: float = 0.20  # 20% of augmented samples (10% of total 2.0)
    smote_k_neighbors: int = 5
    
    # Note: GAN/VAE/Diffusion models require pre-training, marked as optional
    use_gan: bool = False
    gan_contribution: float = 0.00  # 0% initially, can be enabled after GAN training
    
    use_vae: bool = False
    vae_contribution: float = 0.00  # 0% initially, can be enabled after VAE training
    
    # ==================== DATASET-SPECIFIC SETTINGS ====================
    
    # Sampling rates for different datasets
    dataset_sampling_rates: Dict[str, int] = field(default_factory=lambda: {
        'DEAP': 128,  # Hz
        'GAMEEMO': 128,  # Hz (needs verification)
        'SEED-IV': 200,  # Hz
    })
    
    # Frequency bands for filtering (standard EEG bands)
    frequency_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    })
    
    # ==================== VALIDATION SETTINGS ====================
    
    # Quality checks
    enable_quality_checks: bool = True
    max_amplitude_change: float = 3.0  # max std deviation change allowed
    spectral_similarity_threshold: float = 0.7  # min spectral correlation
    
    # Visualization
    save_augmentation_examples: bool = True
    num_examples_to_visualize: int = 5
    
    def validate_contributions(self) -> bool:
        """Validate that contribution ratios sum to approximately target ratio"""
        total_contribution = (
            (self.gaussian_noise_contribution if self.use_gaussian_noise else 0) +
            (self.time_shift_contribution if self.use_time_shift else 0) +
            (self.window_slicing_contribution if self.use_window_slicing else 0) +
            (self.amplitude_scaling_contribution if self.use_amplitude_scaling else 0) +
            (self.channel_dropout_contribution if self.use_channel_dropout else 0) +
            (self.frequency_filtering_contribution if self.use_frequency_filtering else 0) +
            (self.timefreq_augmentation_contribution if self.use_timefreq_augmentation else 0) +
            (self.mixup_contribution if self.use_mixup else 0) +
            (self.cutmix_contribution if self.use_cutmix else 0) +
            (self.smote_contribution if self.use_smote else 0) +
            (self.gan_contribution if self.use_gan else 0) +
            (self.vae_contribution if self.use_vae else 0)
        )
        
        return abs(total_contribution - self.target_augmentation_ratio) < 0.1
    
    def get_contribution_summary(self) -> Dict[str, float]:
        """Get summary of all active augmentation contributions"""
        contributions = {}
        
        if self.use_gaussian_noise:
            contributions['Gaussian Noise'] = self.gaussian_noise_contribution
        if self.use_time_shift:
            contributions['Time Shift'] = self.time_shift_contribution
        if self.use_window_slicing:
            contributions['Window Slicing'] = self.window_slicing_contribution
        if self.use_amplitude_scaling:
            contributions['Amplitude Scaling'] = self.amplitude_scaling_contribution
        if self.use_channel_dropout:
            contributions['Channel Dropout'] = self.channel_dropout_contribution
        if self.use_frequency_filtering:
            contributions['Frequency Filtering'] = self.frequency_filtering_contribution
        if self.use_timefreq_augmentation:
            contributions['Time-Frequency Aug'] = self.timefreq_augmentation_contribution
        if self.use_mixup:
            contributions['Mixup'] = self.mixup_contribution
        if self.use_cutmix:
            contributions['CutMix'] = self.cutmix_contribution
        if self.use_smote:
            contributions['SMOTE'] = self.smote_contribution
        if self.use_gan:
            contributions['GAN'] = self.gan_contribution
        if self.use_vae:
            contributions['VAE'] = self.vae_contribution
        
        return contributions
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("EEG AUGMENTATION CONFIGURATION")
        print("=" * 80)
        print(f"Target Augmentation Ratio: 1:{self.target_augmentation_ratio}")
        print(f"  (Example: 32 original -> {int(32 * self.target_augmentation_ratio)} augmented)")
        print()
        
        contributions = self.get_contribution_summary()
        total = sum(contributions.values())
        
        print("AUGMENTATION TECHNIQUES:")
        print("-" * 80)
        
        # Beginner techniques
        print("\nBEGINNER (Safe & Reliable):")
        beginner_techs = ['Gaussian Noise', 'Time Shift', 'Window Slicing', 
                          'Amplitude Scaling', 'Channel Dropout']
        for tech in beginner_techs:
            if tech in contributions:
                pct = (contributions[tech] / total) * 100
                print(f"  {tech:25s}: {contributions[tech]:.2f} ({pct:.1f}% of augmented)")
        
        # Intermediate techniques
        print("\nINTERMEDIATE (Signal-Aware):")
        intermediate_techs = ['Frequency Filtering', 'Time-Frequency Aug', 'Mixup', 'CutMix']
        for tech in intermediate_techs:
            if tech in contributions or tech.lower() in contributions:
                key = tech if tech in contributions else tech.lower()
                pct = (contributions[key] / total) * 100
                print(f"  {tech:25s}: {contributions[key]:.2f} ({pct:.1f}% of augmented)")
        
        # Advanced techniques
        print("\nADVANCED (Generative):")
        advanced_techs = ['SMOTE', 'GAN', 'VAE']
        for tech in advanced_techs:
            if tech in contributions:
                pct = (contributions[tech] / total) * 100
                print(f"  {tech:25s}: {contributions[tech]:.2f} ({pct:.1f}% of augmented)")
        
        print("-" * 80)
        print(f"TOTAL CONTRIBUTION: {total:.2f}")
        
        if self.validate_contributions():
            print("   Configuration valid: contributions match target ratio")
        else:
            print(f"    Warning: Total contribution ({total:.2f}) differs from target ({self.target_augmentation_ratio:.2f})")
        
        print("=" * 80)


# Default configuration instance
DEFAULT_CONFIG = AugmentationConfig()


if __name__ == "__main__":
    # Test configuration
    config = AugmentationConfig()
    config.print_summary()
