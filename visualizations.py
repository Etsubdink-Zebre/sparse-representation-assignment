"""
Visualizations Module
===============================================================================

This module handles:
- Creating comprehensive visualizations for model comparison
- Generating accuracy comparison charts
- Creating sparsity analysis visualizations
- Producing confusion matrices
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the assets directory exists
os.makedirs('dashboard/assets', exist_ok=True)

def create_comprehensive_comparison(baseline_metrics, sparse_metrics, sparsity_analysis):
    """
    Create comprehensive visualization comparing baseline and sparse models
    
    Args:
        baseline_metrics (dict): Baseline model metrics
        sparse_metrics (dict): Sparse model metrics
        sparsity_analysis (dict): Sparsity analysis results
    """
    print("\n" + "="*60)
    print("5. GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sparse Representation Regularization Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy Comparison
    models = ['No Regularization\n(Overfitted)', 'Sparse Representation\n(L1 Regularized)']
    train_scores = [baseline_metrics['train_accuracy'], sparse_metrics['train_accuracy']]
    test_scores = [baseline_metrics['test_accuracy'], sparse_metrics['test_accuracy']]
    
    x = np.arange(len(models))
    axes[0, 0].bar(x - 0.2, train_scores, 0.4, label='Train Accuracy', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + 0.2, test_scores, 0.4, label='Test Accuracy', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, fontsize=10)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        axes[0, 0].text(i-0.2, train+0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i+0.2, test+0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Multiple Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    baseline_train = [baseline_metrics['train_accuracy'], baseline_metrics['train_precision'], 
                     baseline_metrics['train_recall'], baseline_metrics['train_f1']]
    baseline_test = [baseline_metrics['test_accuracy'], baseline_metrics['test_precision'], 
                    baseline_metrics['test_recall'], baseline_metrics['test_f1']]
    sparse_train = [sparse_metrics['train_accuracy'], sparse_metrics['train_precision'], 
                   sparse_metrics['train_recall'], sparse_metrics['train_f1']]
    sparse_test = [sparse_metrics['test_accuracy'], sparse_metrics['test_precision'], 
                  sparse_metrics['test_recall'], sparse_metrics['test_f1']]
    
    x = np.arange(len(metrics))
    width = 0.2
    
    axes[0, 1].bar(x - 1.5*width, baseline_train, width, label='Baseline Train', alpha=0.7, color='blue')
    axes[0, 1].bar(x - 0.5*width, baseline_test, width, label='Baseline Test', alpha=0.7, color='red')
    axes[0, 1].bar(x + 0.5*width, sparse_train, width, label='Sparse Train', alpha=0.7, color='green')
    axes[0, 1].bar(x + 1.5*width, sparse_test, width, label='Sparse Test', alpha=0.7, color='orange')
    
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Comprehensive Metrics Comparison')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overfitting Gap Comparison
    overfitting_gaps = [baseline_metrics['overfitting_gap'], sparse_metrics['overfitting_gap']]
    colors = ['red' if gap > 0.05 else 'orange' for gap in overfitting_gaps]
    axes[1, 0].bar(models, overfitting_gaps, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Overfitting Gap (Train - Test)')
    axes[1, 0].set_title('Overfitting Analysis')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, gap in enumerate(overfitting_gaps):
        axes[1, 0].text(i, gap+0.005, f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add threshold line
    axes[1, 0].axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='5% Threshold')
    axes[1, 0].legend()
    
    # Plot 4: Sparsity Analysis
    labels = ['Zero Coefficients', 'Non-Zero Coefficients']
    sizes = [sparsity_analysis['zero_coefficients'], sparsity_analysis['non_zero_coefficients']]
    colors = ['lightgray', 'darkblue']
    explode = (0.1, 0)  # explode the 1st slice
    
    axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    axes[1, 1].set_title(f'Feature Sparsity\n({sparsity_analysis["zero_coefficients"]}/{sparsity_analysis["total_features"]} features eliminated)')
    
    plt.tight_layout()
    plt.savefig('dashboard/assets/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrices(y_test, baseline_pred, sparse_pred):
    """
    Create confusion matrices comparison
    
    Args:
        y_test (array): True test labels
        baseline_pred (array): Baseline model predictions
        sparse_pred (array): Sparse model predictions
    """
    from sklearn.metrics import confusion_matrix
    
    # Additional visualization: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold')
    
    # Baseline confusion matrix
    cm1 = confusion_matrix(y_test, baseline_pred)
    axes[0].imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Baseline Model (No Regularization)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Add text annotations
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            axes[0].text(j, i, str(cm1[i, j]), ha="center", va="center")
    
    # Sparse confusion matrix
    cm2 = confusion_matrix(y_test, sparse_pred)
    axes[1].imshow(cm2, interpolation='nearest', cmap=plt.cm.Greens)
    axes[1].set_title('Sparse Model (L1 Regularization)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # Add text annotations
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            axes[1].text(j, i, str(cm2[i, j]), ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig('dashboard/assets/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sparsity_visualization(model, title="Feature Coefficients"):
    """
    Create visualization of model coefficients to show sparsity
    
    Args:
        model: Trained model with coefficients
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    coefficients = model.coef_[0]
    feature_indices = range(len(coefficients))
    
    # Plot coefficients
    plt.bar(feature_indices, coefficients, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Highlight zero coefficients
    zero_mask = coefficients == 0
    non_zero_mask = ~zero_mask
    
    # Create separate colors for zero and non-zero coefficients
    colors = ['red' if zero else 'blue' for zero in zero_mask]
    plt.bar(feature_indices, coefficients, color=colors, alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Non-zero coefficients'),
                       Patch(facecolor='red', alpha=0.7, label='Zero coefficients')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('dashboard/assets/sparsity_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_visualization_summary():
    """Print summary of generated visualizations"""
    print("\n+ Visualizations generated successfully!")
    print("+ Accuracy comparison chart created")
    print("+ Comprehensive metrics comparison created")
    print("+ Overfitting gap analysis created")
    print("+ Sparsity visualization created")
    print("+ Confusion matrices comparison created")

if __name__ == "__main__":
    # Test visualization module
    baseline_metrics = {
        'train_accuracy': 0.95, 'test_accuracy': 0.85, 'overfitting_gap': 0.10,
        'train_precision': 0.94, 'test_precision': 0.84, 'train_recall': 0.96, 'test_recall': 0.86,
        'train_f1': 0.95, 'test_f1': 0.85
    }
    sparse_metrics = {
        'train_accuracy': 0.90, 'test_accuracy': 0.88, 'overfitting_gap': 0.02,
        'train_precision': 0.89, 'test_precision': 0.87, 'train_recall': 0.91, 'test_recall': 0.89,
        'train_f1': 0.90, 'test_f1': 0.88
    }
    sparsity_analysis = {
        'zero_coefficients': 184, 'non_zero_coefficients': 46, 'total_features': 230
    }
    
    create_comprehensive_comparison(baseline_metrics, sparse_metrics, sparsity_analysis)
    print_visualization_summary()
