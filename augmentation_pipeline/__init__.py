"""
EEG Data Augmentation Pipeline
A comprehensive framework for augmenting EEG data with quality validation
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main components for easy access
from .augmentation_config import AugmentationConfig, DEFAULT_CONFIG
from .augmentation_pipeline import AugmentationPipeline, quick_augment
from .dataset_loaders import get_loader, DEAPLoader, SEEDIVLoader, GAMEEMOLoader
from .quality_validation import DataQualityValidator, visualize_augmentation_quality

# Import augmentation technique creators
from .beginner_augmentations import (
    create_beginner_augmentations,
    GaussianNoiseAugmentation,
    TimeShiftAugmentation,
    WindowSlicingAugmentation,
    AmplitudeScalingAugmentation,
    ChannelDropoutAugmentation,
)

from .intermediate_augmentations import (
    create_intermediate_augmentations,
    FrequencyFilteringAugmentation,
    TimeFrequencyAugmentation,
    MixupAugmentation,
    CutMixAugmentation,
)

from .advanced_augmentations import (
    create_advanced_augmentations,
    SMOTEAugmentation,
    AdversarialAugmentation,
    SegmentPermutationAugmentation,
)

__all__ = [
    # Config
    'AugmentationConfig',
    'DEFAULT_CONFIG',
    
    # Pipeline
    'AugmentationPipeline',
    'quick_augment',
    
    # Loaders
    'get_loader',
    'DEAPLoader',
    'SEEDIVLoader',
    'GAMEEMOLoader',
    
    # Validation
    'DataQualityValidator',
    'visualize_augmentation_quality',
    
    # Beginner augmentations
    'create_beginner_augmentations',
    'GaussianNoiseAugmentation',
    'TimeShiftAugmentation',
    'WindowSlicingAugmentation',
    'AmplitudeScalingAugmentation',
    'ChannelDropoutAugmentation',
    
    # Intermediate augmentations
    'create_intermediate_augmentations',
    'FrequencyFilteringAugmentation',
    'TimeFrequencyAugmentation',
    'MixupAugmentation',
    'CutMixAugmentation',
    
    # Advanced augmentations
    'create_advanced_augmentations',
    'SMOTEAugmentation',
    'AdversarialAugmentation',
    'SegmentPermutationAugmentation',
]
