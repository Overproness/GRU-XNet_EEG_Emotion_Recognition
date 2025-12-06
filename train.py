"""
Training script for CBSAtt

Features:
- Full training pipeline with data loading, model creation, and optimization
- Automatic checkpointing at each epoch
- Resume training from checkpoint with --resume argument
- Auto-resume prompt if latest checkpoint is found
- Mixed precision training with gradient accumulation
- Early stopping and learning rate scheduling
- Comprehensive metrics tracking and visualization

Usage:
    # Fresh training
    python train.py --config full
    
    # Resume from specific checkpoint
    python train.py --config full --resume outputs/cbsatt/checkpoints/checkpoint_epoch_5.pth
    
    # Test only (loads best model)
    python train.py --config full --test-only
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig, get_full_training_config, get_quick_test_config
from model import create_cbsatt_model
from data_loader import create_data_loaders
from utils import (
    set_seed,
    get_device,
    EarlyStopping,
    MetricTracker,
    save_checkpoint,
    load_checkpoint,
    plot_training_history
)


class Trainer:
    """Training manager for CBSAtt"""
    
    def __init__(self, config: ExperimentConfig, resume_from_checkpoint: str = None):
        """
        Args:
            config: Experiment configuration
            resume_from_checkpoint: Path to checkpoint to resume from (None for fresh start)
        """
        self.config = config
        
        # Set seed for reproducibility
        set_seed(config.seed, deterministic=config.deterministic)
        
        # Device
        self.device = get_device(config.training.device)
        print(f"Using device: {self.device}")
        
        # Create data loaders
        print("\nCreating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        # Get data dimensions from first batch
        sample_batch = next(iter(self.train_loader))
        stft_sample = sample_batch[0]
        n_channels, n_freq_bins, n_time_bins = stft_sample.shape[1:]
        
        print(f"\nSTFT dimensions:")
        print(f"  Channels: {n_channels} (padded to max across datasets)")
        print(f"  Frequency bins: {n_freq_bins}")
        print(f"  Time bins: {n_time_bins}")
        
        # Note: Channels are zero-padded to match the max across datasets (62 for SEEDIV)
        self.n_channels = n_channels
        
        # Create model
        print("\nCreating model...")
        self.model = create_cbsatt_model(
            n_channels=n_channels,
            n_freq_bins=n_freq_bins,
            n_time_bins=n_time_bins,
            n_classes=config.model.n_classes,
            model_type=config.model.model_type,
            gru_hidden_size=config.model.gru_hidden_size,
            num_attention_heads=config.model.num_attention_heads,
            dropout=config.model.dropout
        )
        self.model = self.model.to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if config.training.use_scheduler:
            if config.training.scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    **config.training.scheduler_params
                )
            elif config.training.scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    **config.training.scheduler_params
                )
            elif config.training.scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    **config.training.scheduler_params
                )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Early stopping
        self.early_stopping = None
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.patience,
                min_delta=config.training.min_delta,
                mode='min'
            )
        
        # Metric tracking
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Checkpoint directory
        self.checkpoint_dir = os.path.join(config.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        elif self._get_latest_checkpoint():
            # Auto-resume from latest checkpoint if exists
            latest_checkpoint = self._get_latest_checkpoint()
            print(f"\nFound existing checkpoint: {latest_checkpoint}")
            response = input("Resume training from checkpoint? (y/n): ").lower()
            if response == 'y':
                self._load_checkpoint(latest_checkpoint)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Gradient accumulation setup
        accumulation_steps = self.config.training.gradient_accumulation_steps
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (stft_features, labels) in enumerate(pbar):
            stft_features = stft_features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(stft_features)
                    loss = self.criterion(outputs, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Only step optimizer every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Gradient clipping
                    if self.config.training.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.grad_clip
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(stft_features)
                loss = self.criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                loss.backward()
                
                # Only step optimizer every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    if self.config.training.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.grad_clip
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Metrics (multiply back by accumulation_steps to get actual loss)
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        for stft_features, labels in tqdm(self.val_loader, desc="Validation"):
            stft_features = stft_features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(stft_features)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels)
        }
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.get_run_name()}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_metrics.update(train_metrics)
            
            # Validate
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.val_metrics.update(val_metrics)
                
                # Print epoch summary
                print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Learning Rate: {current_lr:.6f}")
                
                # Save best model and checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_acc = val_metrics['accuracy']
                    print(f"  ✓ New best model (Val Loss: {val_metrics['loss']:.4f})")
                
                # Save checkpoint at each epoch
                self._save_checkpoint(epoch, val_metrics['loss'], is_best=is_best)
                
                # Early stopping
                if self.early_stopping is not None:
                    self.early_stopping(val_metrics['loss'])
                    if self.early_stopping.should_stop:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Save training history
        self._save_training_history()
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint at current epoch"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'current_epoch': epoch,
            'train_metrics': self.train_metrics.get_history(),
            'val_metrics': self.val_metrics.get_history(),
            'config': self.config.get_run_name()
        }
        
        # Save epoch checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model if applicable
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        # Save as latest checkpoint (for easy resuming)
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training state"""
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler and scaler if they exist
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        # Restore metric history
        if 'train_metrics' in checkpoint:
            self.train_metrics.metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            self.val_metrics.metrics = checkpoint['val_metrics']
        
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best val loss: {self.best_val_loss:.4f}, Best val acc: {self.best_val_acc:.2f}%")
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint if exists"""
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    def test(self) -> Dict[str, float]:
        """Evaluate on test set"""
        print("\nEvaluating on test set...")
        
        # Load best model
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            load_checkpoint(self.model, self.optimizer, best_model_path)
            print("Loaded best model for testing")
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for stft_features, labels in tqdm(self.test_loader, desc="Testing"):
                stft_features = stft_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(stft_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels)
        }
    
    def _save_training_history(self):
        """Save training history to file"""
        history = {
            'train': self.train_metrics.get_history(),
            'val': self.val_metrics.get_history(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config.get_run_name()
        }
        
        history_path = os.path.join(self.config.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, dict):
                    history_serializable[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    history_serializable[key] = value
            
            json.dump(history_serializable, f, indent=4)
        
        print(f"Training history saved to {history_path}")
        
        # Plot training curves
        plot_path = os.path.join(self.config.output_dir, 'figures', 'training_curves.png')
        plot_training_history(history, save_path=plot_path)
        print(f"Training curves saved to {plot_path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CBSAtt model')
    parser.add_argument('--config', type=str, default='full', 
                       choices=['full', 'quick'],
                       help='Configuration preset')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'quick':
        config = get_quick_test_config()
    else:
        config = get_full_training_config()
    
    # Save configuration
    config.save(os.path.join(config.output_dir, 'config.json'))
    
    # Create trainer
    trainer = Trainer(config, resume_from_checkpoint=args.resume)
    
    if args.test_only:
        # Only test
        test_results = trainer.test()
    else:
        # Train and test
        trainer.train()
        test_results = trainer.test()
    
    # Save test results
    test_results_path = os.path.join(config.output_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        # Convert numpy arrays to lists
        test_results_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in test_results.items()
        }
        json.dump(test_results_serializable, f, indent=4)
    
    print(f"Test results saved to {test_results_path}")


if __name__ == "__main__":
    main()
