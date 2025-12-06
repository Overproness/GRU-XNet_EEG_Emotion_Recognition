"""
Evaluation script for EEG-based Emotion Recognition models
"""

import torch
import numpy as np
from sklearn.metrics import classification_report

from utils import calculate_metrics, plot_confusion_matrix, plot_roc_curve, print_metrics


def evaluate_model(model, data_loader, device, num_classes=2):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to evaluate on
        num_classes: Number of classes
        
    Returns:
        Dictionary containing predictions, labels, probabilities, and metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    metrics = calculate_metrics(labels, predictions, probabilities, num_classes)
    
    return {
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities,
        'metrics': metrics
    }


def generate_evaluation_report(model, val_loader, device, num_classes, 
                               class_names, model_name, save_dir):
    """
    Generate comprehensive evaluation report
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device
        num_classes: Number of classes
        class_names: List of class names
        model_name: Name of the model (for saving files)
        save_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation results
    """
    print(f"\nEvaluating {model_name}...")
    
    # Evaluate model
    results = evaluate_model(model, val_loader, device, num_classes)
    
    # Print metrics
    print_metrics(results['metrics'], model_name)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(results['labels'], results['predictions'], 
                               target_names=class_names))
    
    # Plot confusion matrix
    cm_path = f"{save_dir}/confusion_matrix_{model_name}.png"
    plot_confusion_matrix(results['labels'], results['predictions'], 
                         class_names, save_path=cm_path,
                         title=f'Confusion Matrix - {model_name}')
    
    # Plot ROC curve
    roc_path = f"{save_dir}/roc_curve_{model_name}.png"
    plot_roc_curve(results['labels'], results['probabilities'], 
                  num_classes=num_classes, save_path=roc_path,
                  title=f'ROC Curve - {model_name}')
    
    return results
