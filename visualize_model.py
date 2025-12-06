"""
Comprehensive Visualization Script for CBSAtt Model

This script loads a trained model and generates various visualizations:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Per-class Metrics (Precision, Recall, F1-Score)
- Prediction Distribution
- Dataset-wise Performance Analysis
- Attention Weights Visualization (if available)

Usage:
    python visualize_model.py --checkpoint outputs/cbsatt/checkpoints/checkpoint_epoch_30.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig
from model import create_cbsatt_model
from data_loader import create_data_loaders
from utils import set_seed, get_device


class ModelVisualizer:
    """Comprehensive model visualization and analysis"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: ExperimentConfig = None,
        output_dir: str = None
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config: Experiment configuration (if None, loads from checkpoint)
            output_dir: Output directory for visualizations
        """
        self.checkpoint_path = checkpoint_path
        self.device = get_device('cuda')
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load or use provided config
        if config is None:
            # Try to load config from outputs directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_dir = os.path.dirname(checkpoint_dir)  # Go up from checkpoints/
            config_path = os.path.join(output_dir, 'config.json')
            
            if os.path.exists(config_path):
                print(f"Loading config from: {config_path}")
                self.config = ExperimentConfig.load(config_path)
            elif 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                print("Warning: No config found, using default")
                self.config = ExperimentConfig()
        else:
            self.config = config
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(
                self.config.output_dir,
                'visualizations'
            )
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Saving visualizations to: {self.output_dir}")
        
        # Set seed for reproducibility
        set_seed(self.config.seed, deterministic=False)
        
        # Extract model architecture info from checkpoint
        print("\nExtracting model architecture from checkpoint...")
        model_state_keys = checkpoint['model_state_dict'].keys()
        channel_cnn_keys = [k for k in model_state_keys if 'channel_cnns' in k and 'conv1.weight' in k]
        n_channels = len(channel_cnn_keys)
        
        print(f"Model was trained with {n_channels} channels")
        
        # Get STFT dimensions from config
        n_freq_bins = self.config.stft.target_freq_bins
        n_time_bins = self.config.stft.target_time_bins
        
        print(f"STFT dimensions: {n_freq_bins} freq bins, {n_time_bins} time bins")
        
        # Create data loaders
        print("\nCreating data loaders...")
        _, _, self.test_loader = create_data_loaders(self.config)
        
        print(f"Data dimensions: {n_channels} channels (from checkpoint), {n_freq_bins} freq bins, {n_time_bins} time bins")
        
        # Store model info
        self.n_channels = n_channels
        self.n_freq_bins = n_freq_bins
        self.n_time_bins = n_time_bins
        
        # Create model
        print("\nCreating model...")
        self.model = create_cbsatt_model(
            n_channels=n_channels,
            n_freq_bins=n_freq_bins,
            n_time_bins=n_time_bins,
            n_classes=self.config.model.n_classes,
            model_type=self.config.model.model_type,
            gru_hidden_size=self.config.model.gru_hidden_size,
            num_attention_heads=self.config.model.num_attention_heads,
            dropout=self.config.model.dropout
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Store checkpoint info
        self.epoch = checkpoint.get('epoch', 'unknown')
        self.best_val_acc = checkpoint.get('best_val_acc', 'unknown')
        
        print(f"Checkpoint info:")
        print(f"  Epoch: {self.epoch}")
        print(f"  Best validation accuracy: {self.best_val_acc}")
        
    def get_predictions(self):
        """
        Get model predictions on test set
        
        Returns:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            sample_info: Additional info (dataset names, etc.)
        """
        print("\nGenerating predictions on test set...")
        
        y_true = []
        y_pred = []
        y_proba = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Predicting"):
                stft_features, labels = batch
                
                # Extract only the channels that the model was trained with
                # (in case data has more channels due to padding)
                if stft_features.shape[1] > self.n_channels:
                    stft_features = stft_features[:, :self.n_channels, :, :]
                
                stft_features = stft_features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(stft_features)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_proba.extend(probs.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)
        
        print(f"Generated predictions for {len(y_true)} samples")
        
        return y_true, y_pred, y_proba
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Plot confusion matrix"""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Define class names
        if class_names is None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                class_names = ['Negative', 'Positive']
            else:
                class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Plot absolute confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            cbar_kws={'label': 'Count'}
        )
        axes[0].set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Plot normalized confusion matrix
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            cbar_kws={'label': 'Percentage'}
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")
        
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_proba):
        """Plot ROC curve"""
        print("\nGenerating ROC curve...")
        
        # Check actual number of classes in data
        n_classes_model = y_proba.shape[1]
        unique_labels = np.unique(y_true)
        n_classes_data = len(unique_labels)
        
        print(f"Model outputs: {n_classes_model} classes")
        print(f"Data has: {n_classes_data} unique labels: {unique_labels}")
        
        # If data has more classes than model, filter to binary
        if n_classes_data > n_classes_model:
            print(f"Warning: Data has {n_classes_data} classes but model has {n_classes_model}. Filtering to first {n_classes_model} classes.")
            # Keep only samples with labels in [0, n_classes_model-1]
            valid_mask = y_true < n_classes_model
            y_true = y_true[valid_mask]
            y_proba = y_proba[valid_mask]
            print(f"Filtered to {len(y_true)} samples")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        if n_classes_model == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})'
            )
        else:
            # Multi-class classification
            for i in range(n_classes_model):
                # One-vs-Rest
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(
                    fpr, tpr,
                    lw=2,
                    label=f'Class {i} (AUC = {roc_auc:.4f})'
                )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to: {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_proba):
        """Plot precision-recall curve"""
        print("\nGenerating precision-recall curve...")
        
        n_classes = y_proba.shape[1]
        
        # Filter data if needed (same as ROC curve)
        unique_labels = np.unique(y_true)
        if len(unique_labels) > n_classes:
            valid_mask = y_true < n_classes
            y_true = y_true[valid_mask]
            y_proba = y_proba[valid_mask]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        if n_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])
            
            plt.plot(
                recall, precision,
                lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})'
            )
        else:
            # Multi-class classification
            for i in range(n_classes):
                # One-vs-Rest
                y_true_binary = (y_true == i).astype(int)
                precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
                avg_precision = average_precision_score(y_true_binary, y_proba[:, i])
                
                plt.plot(
                    recall, precision,
                    lw=2,
                    label=f'Class {i} (AP = {avg_precision:.4f})'
                )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'precision_recall_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved precision-recall curve to: {save_path}")
        
        plt.close()
    
    def plot_classification_metrics(self, y_true, y_pred, class_names=None):
        """Plot per-class classification metrics"""
        print("\nGenerating classification metrics...")
        
        # Define class names
        if class_names is None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                class_names = ['Negative', 'Positive']
            else:
                class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Calculate metrics for each class
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'classification_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved classification metrics to: {save_path}")
        
        plt.close()
        
        # Also save detailed classification report
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Epoch: {self.epoch}\n\n")
            f.write(report)
        print(f"Saved classification report to: {report_path}")
    
    def plot_prediction_distribution(self, y_true, y_pred, class_names=None):
        """Plot prediction distribution"""
        print("\nGenerating prediction distribution...")
        
        # Define class names
        if class_names is None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                class_names = ['Negative', 'Positive']
            else:
                class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Count predictions
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # True label distribution
        axes[0].bar(class_names, counts_true, alpha=0.8, color='skyblue', edgecolor='black')
        axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, count in enumerate(counts_true):
            axes[0].text(i, count, str(count), ha='center', va='bottom', fontsize=11)
        
        # Predicted label distribution
        axes[1].bar(class_names, counts_pred, alpha=0.8, color='lightcoral', edgecolor='black')
        axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, count in enumerate(counts_pred):
            axes[1].text(i, count, str(count), ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'prediction_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction distribution to: {save_path}")
        
        plt.close()
    
    def plot_confidence_distribution(self, y_true, y_pred, y_proba):
        """Plot confidence distribution for correct and incorrect predictions"""
        print("\nGenerating confidence distribution...")
        
        # Get confidence (max probability)
        confidence = np.max(y_proba, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (y_true == y_pred)
        correct_confidence = confidence[correct_mask]
        incorrect_confidence = confidence[~correct_mask]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(
            correct_confidence,
            bins=30,
            alpha=0.7,
            label=f'Correct ({len(correct_confidence)})',
            color='green',
            edgecolor='black'
        )
        axes[0].hist(
            incorrect_confidence,
            bins=30,
            alpha=0.7,
            label=f'Incorrect ({len(incorrect_confidence)})',
            color='red',
            edgecolor='black'
        )
        axes[0].set_xlabel('Confidence', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Box plot
        data_to_plot = [correct_confidence, incorrect_confidence]
        bp = axes[1].boxplot(
            data_to_plot,
            labels=['Correct', 'Incorrect'],
            patch_artist=True,
            showmeans=True
        )
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[1].set_ylabel('Confidence', fontsize=12)
        axes[1].set_title('Confidence Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add mean values as text
        axes[1].text(
            1, np.mean(correct_confidence),
            f'μ={np.mean(correct_confidence):.3f}',
            ha='center', va='bottom', fontsize=10
        )
        axes[1].text(
            2, np.mean(incorrect_confidence),
            f'μ={np.mean(incorrect_confidence):.3f}',
            ha='center', va='bottom', fontsize=10
        )
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, 'confidence_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confidence distribution to: {save_path}")
        
        plt.close()
    
    def save_metrics_summary(self, y_true, y_pred, y_proba):
        """Save comprehensive metrics summary as JSON"""
        print("\nSaving metrics summary...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None).tolist()
        recall_per_class = recall_score(y_true, y_pred, average=None).tolist()
        f1_per_class = f1_score(y_true, y_pred, average=None).tolist()
        
        # Confidence statistics
        confidence = np.max(y_proba, axis=1)
        correct_mask = (y_true == y_pred)
        
        summary = {
            'checkpoint': self.checkpoint_path,
            'epoch': self.epoch,
            'best_val_acc': self.best_val_acc,
            'test_samples': len(y_true),
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted)
            },
            'per_class_metrics': {
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1_score': f1_per_class
            },
            'confidence_stats': {
                'mean_confidence': float(np.mean(confidence)),
                'mean_correct_confidence': float(np.mean(confidence[correct_mask])),
                'mean_incorrect_confidence': float(np.mean(confidence[~correct_mask])),
                'std_confidence': float(np.std(confidence))
            },
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Save to JSON
        save_path = os.path.join(self.output_dir, 'metrics_summary.json')
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved metrics summary to: {save_path}")
        
        return summary
    
    def generate_all_visualizations(self, class_names=None):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*80)
        
        # Get predictions
        y_true, y_pred, y_proba = self.get_predictions()
        
        # Filter data if needed (SEEDIV has 4 classes but model is binary)
        n_classes_model = y_proba.shape[1]
        unique_labels = np.unique(y_true)
        
        if len(unique_labels) > n_classes_model:
            print(f"\nWarning: Data has {len(unique_labels)} unique labels but model has {n_classes_model} classes")
            print(f"Unique labels in data: {unique_labels}")
            print(f"Filtering to samples with labels < {n_classes_model}")
            
            valid_mask = y_true < n_classes_model
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            y_proba = y_proba[valid_mask]
            
            print(f"Filtered from {len(valid_mask)} to {len(y_true)} samples")
        
        # Generate all plots
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        self.plot_roc_curve(y_true, y_proba)
        self.plot_precision_recall_curve(y_true, y_proba)
        self.plot_classification_metrics(y_true, y_pred, class_names)
        self.plot_prediction_distribution(y_true, y_pred, class_names)
        self.plot_confidence_distribution(y_true, y_pred, y_proba)
        
        # Save metrics summary
        summary = self.save_metrics_summary(y_true, y_pred, y_proba)
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"\nAll visualizations saved to: {self.output_dir}")
        print(f"\nOverall Accuracy: {summary['overall_metrics']['accuracy']:.4f}")
        print(f"Macro F1-Score: {summary['overall_metrics']['f1_macro']:.4f}")
        print(f"Weighted F1-Score: {summary['overall_metrics']['f1_weighted']:.4f}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Visualize CBSAtt model performance')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/cbsatt/checkpoints/checkpoint_epoch_30.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: checkpoint_dir/visualizations)'
    )
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=None,
        help='Class names for labels (e.g., --class-names Negative Positive)'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelVisualizer(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(class_names=args.class_names)


if __name__ == '__main__':
    main()
