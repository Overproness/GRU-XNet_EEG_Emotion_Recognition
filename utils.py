"""
Utility functions for training and evaluation
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(preferred_device: str = 'cuda') -> torch.device:
    """
    Get the device to use for training
    
    Args:
        preferred_device: Preferred device ('cuda' or 'cpu')
        
    Returns:
        torch.device
    """
    if preferred_device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float):
        """
        Args:
            score: Current validation score
        """
        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == 'min':
                improved = score < (self.best_score - self.min_delta)
            else:
                improved = score > (self.best_score + self.min_delta)
            
            if improved:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True


class MetricTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            # Only append scalar values
            if isinstance(value, (int, float)):
                self.metrics[key].append(value)
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get metric history"""
        return self.metrics
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest metrics"""
        return {key: values[-1] for key, values in self.metrics.items() if len(values) > 0}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str
) -> int:
    """
    Load model checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        path: Path to checkpoint
        
    Returns:
        epoch: Epoch of loaded checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Dictionary with 'train' and 'val' metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'loss' in history['train']:
        axes[0].plot(history['train']['loss'], label='Train Loss')
    if 'loss' in history['val']:
        axes[0].plot(history['val']['loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'accuracy' in history['train']:
        axes[1].plot(history['train']['accuracy'], label='Train Accuracy')
    if 'accuracy' in history['val']:
        axes[1].plot(history['val']['accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
):
    """
    Print classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def log_model_info(model: torch.nn.Module):
    """
    Log model information
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("Model Information")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    print("Seed set")
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    for i, loss in enumerate([1.0, 0.9, 0.85, 0.84, 0.83, 0.82]):
        early_stopping(loss)
        print(f"  Epoch {i}: Loss={loss}, Counter={early_stopping.counter}, Stop={early_stopping.should_stop}")
    print("Early stopping tested")
    
    # Test metric tracker
    tracker = MetricTracker()
    for i in range(5):
        tracker.update({'loss': 1.0 - i*0.1, 'accuracy': 50 + i*5})
    print(f"Metric tracker: {tracker.get_latest()}")
    
    # Test confusion matrix plotting
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    plot_confusion_matrix(y_true, y_pred, class_names=['Negative', 'Positive'])
    print("Confusion matrix plotted")
    
    print("\nAll utility tests passed!")
