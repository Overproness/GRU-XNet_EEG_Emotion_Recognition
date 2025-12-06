"""
Script to visualize and compare results from all models
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import RESULTS_DIR, FIGURES_DIR


def load_all_results():
    """Load results from all experiments"""
    results = {}
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return results
    
    # Look for JSON result files
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('_results.json'):
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                model_name = data['model_name']
                results[model_name] = data
    
    return results


def plot_comparison_bar_chart(results, save_path=None):
    """Create bar chart comparing all models"""
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Prepare data
    data = {metric: [] for metric in metrics}
    for model in models:
        for metric in metrics:
            value = results[model]['metrics'].get(metric, 0)
            data[metric].append(value)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), color=color)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison Across All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').upper() for m in models], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    
    plt.close()


def plot_metric_heatmap(results, save_path=None):
    """Create heatmap of metrics across models"""
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Create matrix
    data_matrix = []
    for model in models:
        row = []
        for metric in metrics:
            value = results[model]['metrics'].get(metric, 0)
            row.append(value)
        data_matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=[m.replace('_', ' ').title() for m in metrics],
                yticklabels=[m.replace('_', ' ').upper() for m in models],
                vmin=0.7, vmax=1.0, cbar_kws={'label': 'Score'})
    
    ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.close()


def plot_binary_vs_multiclass(results, save_path=None):
    """Compare binary vs multiclass performance"""
    if not results:
        print("No results to plot")
        return
    
    # Separate binary and multiclass results
    binary_results = {k: v for k, v in results.items() if 'binary' in k}
    multiclass_results = {k: v for k, v in results.items() if 'multiclass' in k}
    
    if not binary_results or not multiclass_results:
        print("Need both binary and multiclass results for comparison")
        return
    
    # Create comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Binary models
    ax = axes[0]
    models = list(binary_results.keys())
    data = {metric: [binary_results[m]['metrics'][metric] for m in models] for metric in metrics}
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Binary Classification Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('_')[1].upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # Multiclass models
    ax = axes[1]
    models = list(multiclass_results.keys())
    data = {metric: [multiclass_results[m]['metrics'][metric] for m in models] for metric in metrics}
    
    x = np.arange(len(models))
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Multiclass Classification Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('_')[1].upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Binary vs Multiclass comparison saved to {save_path}")
    
    plt.close()


def print_results_table(results):
    """Print results in a formatted table"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC AUC':>10}")
    print("-"*100)
    
    # Results
    for model_name, data in sorted(results.items()):
        metrics = data['metrics']
        print(f"{model_name.replace('_', ' ').upper():<20} "
              f"{metrics['accuracy']:>10.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1_score']:>10.4f} "
              f"{metrics.get('roc_auc', 0):>10.4f}")
    
    print("="*100)
    
    # Find best models
    print("\nBEST PERFORMING MODELS:")
    print("-"*100)
    
    best_binary = max([k for k in results.keys() if 'binary' in k], 
                     key=lambda x: results[x]['metrics']['accuracy'])
    print(f"Binary Classification:    {best_binary.replace('_', ' ').upper()} "
          f"(Accuracy: {results[best_binary]['metrics']['accuracy']:.4f})")
    
    best_multiclass = max([k for k in results.keys() if 'multiclass' in k], 
                         key=lambda x: results[x]['metrics']['accuracy'])
    print(f"Multiclass Classification: {best_multiclass.replace('_', ' ').upper()} "
          f"(Accuracy: {results[best_multiclass]['metrics']['accuracy']:.4f})")
    
    print("="*100)


def main():
    """Main function"""
    print("\n" + "="*60)
    print("Results Visualization and Comparison")
    print("="*60)
    
    # Load results
    print("\nLoading results...")
    results = load_all_results()
    
    if not results:
        print("\nNo results found. Please run experiments first:")
        print("  python main.py")
        return
    
    print(f"Found results for {len(results)} models")
    
    # Print results table
    print_results_table(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Comparison bar chart
    comparison_path = os.path.join(FIGURES_DIR, 'all_models_comparison.png')
    plot_comparison_bar_chart(results, save_path=comparison_path)
    
    # Heatmap
    heatmap_path = os.path.join(FIGURES_DIR, 'metrics_heatmap.png')
    plot_metric_heatmap(results, save_path=heatmap_path)
    
    # Binary vs Multiclass
    comparison2_path = os.path.join(FIGURES_DIR, 'binary_vs_multiclass.png')
    plot_binary_vs_multiclass(results, save_path=comparison2_path)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Charts saved to: {FIGURES_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
