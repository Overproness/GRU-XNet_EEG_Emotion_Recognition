"""
Configuration for gru_xnet training and evaluation
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class STFTConfig:
    """STFT preprocessing configuration"""
    # Frequency range to keep (Hz)
    freq_range: Tuple[float, float] = (0.5, 50.0)
    
    # Target dimensions after standardization
    target_freq_bins: int = 129
    target_time_bins: int = 126
    
    # Dataset-specific STFT parameters
    # Will be auto-configured based on sampling rates


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Model type
    model_type: str = 'dynamic'  # 'standard' or 'dynamic'
    
    # BiGRU parameters
    gru_hidden_size: int = 128
    gru_num_layers: int = 2
    
    # Attention parameters
    num_attention_heads: int = 4
    
    # Dropout
    dropout: float = 0.5
    
    # Number of classes (will be set based on task)
    n_classes: int = 2


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32  # Reduced from 128 to fit GPU memory with 62-channel model
    num_epochs: int = 30
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_params: Dict = field(default_factory=lambda: {
        'T_max': 30,  # For cosine
        'eta_min': 1e-6
    })
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # Gradient clipping
    grad_clip: Optional[float] = 1.0
    
    # Gradient accumulation (to maintain effective batch size with smaller batches)
    gradient_accumulation_steps: int = 4  # Effective batch size = 32 * 4 = 128
    
    # Mixed precision training
    use_amp: bool = True
    
    # Device
    device: str = 'cuda'  # Will auto-detect if not available


@dataclass
class DataConfig:
    """Data loading configuration"""
    # Dataset paths
    base_dir: str = r"YOUR BASE DIRECTORY PATH HERE"
    augmented_dir: str = "YOUR AUGMENTED DATA DIRECTORY PATH HERE"
    
    # Datasets to use
    datasets: List[str] = field(default_factory=lambda: ['DEAP', 'GAMEEMO', 'SEEDIV'])
    
    # Augmentation ratio (original:augmented)
    use_original: bool = True
    use_augmented: bool = True
    augmentation_ratio: float = 2.0  # 1:2 ratio
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation
    use_loso_cv: bool = False  # Leave-One-Subject-Out
    cv_subject: Optional[int] = None
    
    # Class mapping for SEEDIV (4-class to binary)
    seediv_binary_mapping: Dict[int, int] = field(default_factory=lambda: {
        0: 0,  # Neutral -> Negative
        1: 1,  # Sad -> Negative  
        2: 0,  # Fear -> Negative
        3: 1   # Happy -> Positive
    })
    
    # Alternative: Keep 4-class
    seediv_keep_multiclass: bool = False
    
    # Balancing
    balance_classes: bool = True
    balance_datasets: bool = True  # Equal samples from each dataset
    
    # Caching
    cache_stft: bool = False  # Disabled by default due to memory constraints with large datasets
    cache_dir: str = "cache/gru_xnet_stft"
    
    # Number of workers for data loading
    num_workers: int = 0  # Set to 0 to avoid multiprocessing overhead with large data
    
    # Memory optimization
    use_memory_mapping: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Experiment name and paths
    experiment_name: str = "gru_xnet_multi_dataset"
    output_dir: str = "outputs/gru_xnet"
    
    # Sub-configurations
    stft: STFTConfig = field(default_factory=STFTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Logging
    log_interval: int = 10  # Log every N batches
    save_best_only: bool = True
    save_interval: int = 5  # Save checkpoint every N epochs
    
    # Evaluation
    eval_interval: int = 1  # Evaluate every N epochs
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
        
        # Create cache directory
        cache_path = os.path.join(
            self.data.base_dir,
            self.data.cache_dir
        )
        os.makedirs(cache_path, exist_ok=True)
    
    def get_run_name(self) -> str:
        """Generate a unique run name based on configuration"""
        datasets_str = '_'.join(self.data.datasets)
        return (
            f"{self.experiment_name}_"
            f"{datasets_str}_"
            f"bs{self.training.batch_size}_"
            f"lr{self.training.learning_rate}_"
            f"gru{self.model.gru_hidden_size}_"
            f"heads{self.model.num_attention_heads}"
        )
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        config_dict['stft'] = STFTConfig(**config_dict['stft'])
        config_dict['model'] = ModelConfig(**config_dict['model'])
        config_dict['training'] = TrainingConfig(**config_dict['training'])
        config_dict['data'] = DataConfig(**config_dict['data'])
        
        return cls(**config_dict)


# Preset configurations
def get_quick_test_config() -> ExperimentConfig:
    """Configuration for quick testing"""
    config = ExperimentConfig(
        experiment_name="quick_test"
    )
    
    # Reduce for faster testing
    config.training.num_epochs = 3
    config.training.batch_size = 32
    config.data.train_ratio = 0.1
    config.data.val_ratio = 0.05
    config.data.test_ratio = 0.05
    
    return config


def get_full_training_config() -> ExperimentConfig:
    """Configuration for full training"""
    return ExperimentConfig(
        experiment_name="gru_xnet_full_training"
    )


def get_loso_config(subject_id: int) -> ExperimentConfig:
    """Configuration for Leave-One-Subject-Out cross-validation"""
    config = ExperimentConfig(
        experiment_name=f"gru_xnet_loso_subject_{subject_id}"
    )
    
    config.data.use_loso_cv = True
    config.data.cv_subject = subject_id
    
    return config


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    
    # Create default config
    config = ExperimentConfig()
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Run name: {config.get_run_name()}")
    print(f"Output dir: {config.output_dir}")
    print(f"Datasets: {config.data.datasets}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"GRU hidden size: {config.model.gru_hidden_size}")
    print(f"Attention heads: {config.model.num_attention_heads}")
    
    # Save and load
    config_path = os.path.join(config.output_dir, 'config_test.json')
    config.save(config_path)
    print(f"\nConfig saved to: {config_path}")
    
    loaded_config = ExperimentConfig.load(config_path)
    print("Config loaded successfully!")
    
    # Test preset configs
    print("\n--- Quick Test Config ---")
    quick_config = get_quick_test_config()
    print(f"Epochs: {quick_config.training.num_epochs}")
    print(f"Batch size: {quick_config.training.batch_size}")
    
    print("\n--- LOSO Config ---")
    loso_config = get_loso_config(subject_id=1)
    print(f"Experiment: {loso_config.experiment_name}")
    print(f"Use LOSO: {loso_config.data.use_loso_cv}")
    print(f"CV Subject: {loso_config.data.cv_subject}")
    
    print("\nAll configuration tests passed!")
