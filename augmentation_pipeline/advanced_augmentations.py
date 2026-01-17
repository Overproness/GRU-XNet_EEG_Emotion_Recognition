"""
Advanced-level EEG augmentation techniques
Generative models and sophisticated synthetic data generation (10-25% of augmented data)
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from base_augmentation import GenerativeAugmentation, BaseAugmentation, AugmentationType


class SMOTEAugmentation(BaseAugmentation):
    """
    SMOTE (Synthetic Minority Over-sampling Technique) in feature space
    Contribution: 10-30% for minority classes
    Reliability: MEDIUM - works well on features, less on raw signals
    """
    
    def __init__(self,
                 k_neighbors: int = 5,
                 random_seed: Optional[int] = None):
        """
        Args:
            k_neighbors: Number of nearest neighbors to use
            random_seed: Random seed
        """
        super().__init__("SMOTE", AugmentationType.ADVANCED, random_seed=random_seed)
        self.k_neighbors = k_neighbors
        self.fitted_data = None
        self.fitted_labels = None
    
    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        Fit SMOTE on training data
        
        Args:
            data: Training data, shape (n_samples, n_features) or (n_samples, n_channels, n_timepoints)
            labels: Training labels
        """
        self.fitted_data = data
        self.fitted_labels = labels
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """
        Generate synthetic sample using SMOTE
        
        This finds k nearest neighbors of the same class and interpolates
        """
        if self.fitted_data is None:
            raise RuntimeError("SMOTE must be fitted first. Call fit() before augment().")
        
        self.validate_data(data)
        
        # Find samples of same class
        if label is not None:
            same_class_indices = np.where(self.fitted_labels == label)[0]
            if len(same_class_indices) < 2:
                # Not enough samples, return original
                return data.copy(), label
            
            same_class_data = self.fitted_data[same_class_indices]
        else:
            same_class_data = self.fitted_data
        
        # Flatten for distance computation
        data_flat = data.flatten()
        same_class_flat = same_class_data.reshape(len(same_class_data), -1)
        
        # Find k nearest neighbors
        distances = np.linalg.norm(same_class_flat - data_flat, axis=1)
        k = min(self.k_neighbors, len(same_class_data) - 1)
        nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self
        
        # Randomly select one neighbor
        neighbor_idx = np.random.choice(nearest_indices)
        neighbor = same_class_data[neighbor_idx]
        
        # Interpolate
        alpha = np.random.rand()
        augmented = data + alpha * (neighbor - data)
        
        return augmented, label


class AdversarialAugmentation(BaseAugmentation):
    """
    Add small adversarial perturbations for robustness
    Contribution: 5-15%
    Reliability: MEDIUM - helps robustness but can change true label
    """
    
    def __init__(self,
                 epsilon: float = 0.01,
                 random_seed: Optional[int] = None):
        """
        Args:
            epsilon: Maximum perturbation magnitude (as fraction of signal std)
            random_seed: Random seed
        """
        super().__init__("Adversarial", AugmentationType.ADVANCED, random_seed=random_seed)
        self.epsilon = epsilon
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """
        Add small adversarial perturbation to improve model robustness
        """
        self.validate_data(data)
        
        # Generate random perturbation
        perturbation = np.random.randn(*data.shape)
        perturbation = perturbation / (np.linalg.norm(perturbation) + 1e-10)
        
        # Scale by epsilon and signal std
        signal_std = np.std(data)
        perturbation *= self.epsilon * signal_std
        
        augmented = data + perturbation
        
        return augmented, label


class SegmentPermutationAugmentation(BaseAugmentation):
    """
    Permute segments within window
    Contribution: 5-15%
    Reliability: MEDIUM - can destroy temporal relationships
    Use only if temporal order is not critical
    """
    
    def __init__(self,
                 n_segments: int = 4,
                 random_seed: Optional[int] = None):
        """
        Args:
            n_segments: Number of segments to divide window into
            random_seed: Random seed
        """
        super().__init__("Segment Permutation", AugmentationType.INTERMEDIATE, random_seed=random_seed)
        self.n_segments = n_segments
    
    def augment(self, 
                data: np.ndarray, 
                label: Optional[Union[int, np.ndarray]] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[Union[int, np.ndarray]]]:
        """Permute temporal segments"""
        self.validate_data(data)
        
        if data.ndim == 1:
            n_samples = len(data)
            segment_length = n_samples // self.n_segments
            
            # Create segments
            segments = []
            for i in range(self.n_segments):
                start = i * segment_length
                end = start + segment_length if i < self.n_segments - 1 else n_samples
                segments.append(data[start:end])
            
            # Permute
            np.random.shuffle(segments)
            
            # Concatenate
            augmented = np.concatenate(segments)
        else:
            # Multi-channel
            n_samples = data.shape[1]
            segment_length = n_samples // self.n_segments
            
            segments = []
            for i in range(self.n_segments):
                start = i * segment_length
                end = start + segment_length if i < self.n_segments - 1 else n_samples
                segments.append(data[:, start:end])
            
            # Permute
            np.random.shuffle(segments)
            
            # Concatenate
            augmented = np.concatenate(segments, axis=1)
        
        return augmented, label


def create_advanced_augmentations(random_seed: Optional[int] = None):
    """
    Create instances of all advanced-level augmentations
    
    Returns:
        Dictionary of augmentation instances
    """
    return {
        'smote': SMOTEAugmentation(random_seed=random_seed),
        'adversarial': AdversarialAugmentation(random_seed=random_seed),
        'segment_permutation': SegmentPermutationAugmentation(random_seed=random_seed),
    }


if __name__ == "__main__":
    # Test each augmentation
    print("Testing Advanced Augmentations")
    print("=" * 80)
    
    # Create sample EEG data
    np.random.seed(42)
    n_samples = 100
    sample_data = np.random.randn(n_samples, 32, 256)  # 100 samples, 32 channels, 256 timepoints
    sample_labels = np.random.randint(0, 3, n_samples)
    
    # Test SMOTE
    print("\nSMOTE:")
    smote = SMOTEAugmentation(random_seed=42)
    smote.fit(sample_data, sample_labels)
    aug_data, aug_label = smote.augment(sample_data[0], sample_labels[0])
    print(f"  Original shape: {sample_data[0].shape}, Augmented shape: {aug_data.shape}")
    print(f"  Label: {aug_label}")
    
    # Test Adversarial
    print("\nADVERSARIAL:")
    adv = AdversarialAugmentation(random_seed=42)
    aug_data, _ = adv.augment(sample_data[0], sample_labels[0])
    print(f"  Shape: {aug_data.shape}")
    print(f"  Perturbation magnitude: {np.linalg.norm(aug_data - sample_data[0]):.4f}")
    
    # Test Segment Permutation
    print("\nSEGMENT PERMUTATION:")
    seg_perm = SegmentPermutationAugmentation(random_seed=42)
    aug_data, _ = seg_perm.augment(sample_data[0], sample_labels[0])
    print(f"  Shape: {aug_data.shape}")