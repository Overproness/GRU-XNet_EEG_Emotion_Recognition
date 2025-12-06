"""
Main script for EEG-based Emotion Recognition
Implements: "Accurate EEG-Based Emotion Recognition using LSTM and BiLSTM Networks"

This script orchestrates the entire pipeline:
1. Data loading and feature extraction
2. Model training (LSTM and BiLSTM for binary and multiclass)
3. Evaluation and visualization
"""

import os
import torch
import numpy as np
import argparse
import json

from config import *
from data_loader import get_dataloaders
from models import create_model
from train import train_model
from evaluate import generate_evaluation_report
from utils import create_output_directories, plot_training_history, load_model


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_experiment(classification_type='binary', model_type='bilstm', 
                   use_cached_features=True, num_epochs=100):
    """
    Run a complete experiment
    
    Args:
        classification_type: 'binary' or 'multiclass'
        model_type: 'lstm' or 'bilstm'
        use_cached_features: Whether to use cached features
        num_epochs: Number of training epochs
    """
    # Set seed
    set_seed(RANDOM_SEED)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directories
    create_output_directories(OUTPUT_DIR)
    
    # Prepare class names and model name
    if classification_type == 'binary':
        class_names = ['Negative', 'Positive']
        num_classes = 2
        model_name = f'binary_{model_type}'
    else:
        class_names = ['LANV', 'LAPV', 'HANV', 'HAPV']
        num_classes = 4
        model_name = f'multiclass_{model_type}'
    
    print(f"\n{'='*60}")
    print(f"Experiment: {classification_type.upper()} classification with {model_type.upper()}")
    print(f"{'='*60}")
    
    # Load data
    print("\n1. Loading data and extracting features...")
    train_loader, val_loader, feature_dim = get_dataloaders(
        classification_type=classification_type,
        batch_size=BATCH_SIZE,
        use_cached=use_cached_features,
        random_seed=RANDOM_SEED
    )
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"\n2. Creating {model_type.upper()} model...")
    model = create_model(model_name, input_size=feature_dim, device=device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\n3. Training model...")
    model_save_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}_best.pth')
    
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=LEARNING_RATE,
        device=device,
        save_path=model_save_path,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Plot training history
    history_plot_path = os.path.join(FIGURES_DIR, f'training_history_{model_name}.png')
    plot_training_history(history, save_path=history_plot_path)
    
    # Load best model for evaluation
    print("\n4. Loading best model for evaluation...")
    model, _, _, _ = load_model(model, None, model_save_path, device)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    results = generate_evaluation_report(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        model_name=model_name,
        save_dir=FIGURES_DIR
    )
    
    # Save results to JSON
    results_dict = {
        'model_name': model_name,
        'classification_type': classification_type,
        'num_classes': num_classes,
        'class_names': class_names,
        'metrics': {k: float(v) for k, v in results['metrics'].items()},
        'feature_dim': feature_dim,
        'num_train_samples': len(train_loader.dataset),
        'num_val_samples': len(val_loader.dataset)
    }
    
    results_json_path = os.path.join(RESULTS_DIR, f'{model_name}_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nResults saved to {results_json_path}")
    
    return results


def run_all_experiments(use_cached_features=True, num_epochs=100):
    """
    Run all experiments (binary and multiclass, LSTM and BiLSTM)
    
    Args:
        use_cached_features: Whether to use cached features
        num_epochs: Number of training epochs
    """
    experiments = [
        {'classification_type': 'binary', 'model_type': 'lstm'},
        {'classification_type': 'binary', 'model_type': 'bilstm'},
        {'classification_type': 'multiclass', 'model_type': 'lstm'},
        {'classification_type': 'multiclass', 'model_type': 'bilstm'}
    ]
    
    all_results = {}
    
    for exp in experiments:
        results = run_experiment(
            classification_type=exp['classification_type'],
            model_type=exp['model_type'],
            use_cached_features=use_cached_features,
            num_epochs=num_epochs
        )
        
        model_name = f"{exp['classification_type']}_{exp['model_type']}"
        all_results[model_name] = results['metrics']
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)
    
    for model_name, metrics in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='EEG-based Emotion Recognition using LSTM and BiLSTM'
    )
    
    parser.add_argument(
        '--classification',
        type=str,
        choices=['binary', 'multiclass', 'all'],
        default='all',
        help='Type of classification (binary, multiclass, or all)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'bilstm', 'both'],
        default='both',
        help='Model type (lstm, bilstm, or both)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached features'
    )
    
    args = parser.parse_args()
    
    use_cached = not args.no_cache
    
    # Run experiments based on arguments
    if args.classification == 'all' and args.model == 'both':
        # Run all experiments
        run_all_experiments(use_cached_features=use_cached, num_epochs=args.epochs)
    else:
        # Run specific experiments
        classifications = ['binary', 'multiclass'] if args.classification == 'all' else [args.classification]
        models = ['lstm', 'bilstm'] if args.model == 'both' else [args.model]
        
        for classification in classifications:
            for model in models:
                run_experiment(
                    classification_type=classification,
                    model_type=model,
                    use_cached_features=use_cached,
                    num_epochs=args.epochs
                )


if __name__ == '__main__':
    main()
