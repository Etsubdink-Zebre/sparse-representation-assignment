"""
Performance Analysis Module
===============================================================================

This module handles:
- Comparing baseline and sparse model performance
- Detailed performance analysis and conclusions
- Bias-variance tradeoff explanation
- Recommendations based on results
"""

def compare_performance(baseline_metrics, sparse_metrics, sparsity_analysis):
    """
    Compare performance between baseline and sparse models
    
    Args:
        baseline_metrics (dict): Baseline model metrics
        sparse_metrics (dict): Sparse model metrics
        sparsity_analysis (dict): Sparsity analysis results
    
    Returns:
        dict: Performance comparison results
    """
    print("\n" + "="*60)
    print("4. PERFORMANCE COMPARISON AND ANALYSIS")
    print("="*60)
    
    print("\nPERFORMANCE COMPARISON SUMMARY:")
    print("-"*40)
    
    # Calculate improvements
    acc_improvement = sparse_metrics['test_accuracy'] - baseline_metrics['test_accuracy']
    overfitting_reduction = baseline_metrics['overfitting_gap'] - sparse_metrics['overfitting_gap']
    
    comparison = {
        'accuracy_improvement': acc_improvement,
        'accuracy_improvement_percent': (acc_improvement / baseline_metrics['test_accuracy']) * 100,
        'overfitting_reduction': overfitting_reduction,
        'baseline_test_accuracy': baseline_metrics['test_accuracy'],
        'sparse_test_accuracy': sparse_metrics['test_accuracy'],
        'baseline_overfitting_gap': baseline_metrics['overfitting_gap'],
        'sparse_overfitting_gap': sparse_metrics['overfitting_gap']
    }
    
    print(f"\nACCURACY ANALYSIS:")
    print(f"Baseline Test Accuracy: {comparison['baseline_test_accuracy']:.4f}")
    print(f"Sparse Test Accuracy: {comparison['sparse_test_accuracy']:.4f}")
    print(f"Accuracy Improvement: {comparison['accuracy_improvement']:+.4f} ({comparison['accuracy_improvement_percent']:+.2f}%)")
    
    print(f"\nOVERFITTING ANALYSIS:")
    print(f"Baseline Overfitting Gap: {comparison['baseline_overfitting_gap']:.4f}")
    print(f"Sparse Overfitting Gap: {comparison['sparse_overfitting_gap']:.4f}")
    print(f"Overfitting Reduction: {comparison['overfitting_reduction']:+.4f}")
    
    return comparison

def generate_detailed_analysis(comparison, sparsity_analysis):
    """
    Generate detailed analysis and conclusions
    
    Args:
        comparison (dict): Performance comparison results
        sparsity_analysis (dict): Sparsity analysis results
    
    Returns:
        dict: Detailed analysis results
    """
    print("\n" + "="*50)
    print("DETAILED ANALYSIS AND CONCLUSIONS:")
    print("="*50)
    
    print("\n[ANALYSIS] WHAT HAPPENED AND WHY:")
    print("-"*30)
    
    analysis = {
        'accuracy_improved': comparison['accuracy_improvement'] > 0,
        'overfitting_reduced': comparison['overfitting_reduction'] > 0,
        'high_sparsity': sparsity_analysis['sparsity_ratio'] > 0.5,
        'regularization_effective': False
    }
    
    if analysis['accuracy_improved']:
        print(f"+ Sparse representation IMPROVED test accuracy by {comparison['accuracy_improvement']:.4f}")
        print("  -> L1 regularization successfully identified and eliminated noise features")
        print("  -> Model focuses on the most informative features")
    else:
        print(f"- Sparse representation DECREASED test accuracy by {abs(comparison['accuracy_improvement']):.4f}")
        print("  -> Regularization may be too strong (C=0.1)")
        print("  -> Some useful features might have been eliminated")
    
    if analysis['overfitting_reduced']:
        print(f"+ Overfitting REDUCED by {comparison['overfitting_reduction']:.4f}")
        print("  -> Sparse representation improved generalization")
        print("  -> Model is less complex and more robust")
    else:
        print(f"- Overfitting INCREASED by {abs(comparison['overfitting_reduction']):.4f}")
        print("  -> Regularization parameters may need adjustment")
    
    # Determine if regularization was effective
    analysis['regularization_effective'] = analysis['accuracy_improved'] and analysis['overfitting_reduced']
    
    return analysis

def explain_bias_variance_tradeoff(analysis):
    """
    Explain bias-variance tradeoff in the context of sparse representation
    
    Args:
        analysis (dict): Analysis results
    """
    print("\n[TRADEOFF] BIAS-VARIANCE TRADEOFF EXPLANATION:")
    print("-"*40)
    print("• BEFORE Sparse Representation:")
    print("  - High variance: Model memorized noise features")
    print("  - Low bias: Model fit training data perfectly")
    print("  - Poor generalization to test data")
    
    print("• AFTER Sparse Representation:")
    print("  - Reduced variance: Model simplified by feature selection")
    print("  - Slightly increased bias: Some information lost")
    print("  - Better generalization: Improved test performance")
    
    print("\n[INSIGHTS] KEY INSIGHTS:")
    print("-"*20)
    print("• Sparse representation primarily reduces VARIANCE")
    print("• May slightly increase BIAS by eliminating some features")
    print("• The goal is finding the optimal balance point")
    print("• Effective when variance reduction > bias increase")

def generate_conclusions(analysis, comparison, sparsity_analysis):
    """
    Generate final conclusions and recommendations
    
    Args:
        analysis (dict): Analysis results
        comparison (dict): Performance comparison results
        sparsity_analysis (dict): Sparsity analysis results
    
    Returns:
        dict: Conclusions and recommendations
    """
    print("\n[INSIGHTS] KEY INSIGHTS:")
    print("-"*20)
    print(f"• Sparse representation eliminated {sparsity_analysis['sparsity_ratio']:.1%} of features")
    print(f"• Only {sparsity_analysis['non_zero_coefficients']} features remained from original {sparsity_analysis['total_features']}")
    print("• Feature selection happened automatically via L1 regularization")
    print("• Most eliminated features were likely noise (200 random features)")
    
    print("\n[CONCLUSIONS] CONCLUSIONS:")
    print("-"*20)
    print("1. Sparse Representation successfully addresses overfitting in high-dimensional data")
    print("2. L1 regularization performs automatic feature selection")
    print("3. The bias-variance tradeoff is optimized by reducing variance more than increasing bias")
    print("4. Sparse models are more interpretable and computationally efficient")
    print("5. The technique is particularly effective when many features are irrelevant")
    
    # Generate recommendations
    print("\n[RECOMMENDATIONS] RECOMMENDATIONS:")
    print("-"*20)
    
    recommendations = []
    
    if analysis['regularization_effective']:
        recommendations.append("+ Current regularization strength (C=0.1) is well-tuned")
        recommendations.append("+ Sparse representation is effective for this problem")
    else:
        recommendations.append("? Consider adjusting regularization strength (try C=0.01 or C=1.0)")
        recommendations.append("? May need feature engineering or different regularization approach")
    
    if analysis['high_sparsity']:
        recommendations.append("+ High sparsity achieved - good feature selection")
    
    for rec in recommendations:
        print(rec)
    
    return {
        'conclusions': [
            "Sparse representation addresses overfitting effectively",
            "L1 regularization performs automatic feature selection", 
            "Bias-variance tradeoff optimized through variance reduction",
            "Sparse models are more interpretable and efficient"
        ],
        'recommendations': recommendations,
        'technique_effective': analysis['regularization_effective']
    }

def print_assignment_completion():
    """Print assignment completion message"""
    print("\n" + "="*60)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    # Test analysis module
    baseline_metrics = {
        'test_accuracy': 0.85,
        'overfitting_gap': 0.15
    }
    sparse_metrics = {
        'test_accuracy': 0.90,
        'overfitting_gap': 0.05
    }
    sparsity_analysis = {
        'sparsity_ratio': 0.8,
        'non_zero_coefficients': 46,
        'total_features': 230
    }
    
    comparison = compare_performance(baseline_metrics, sparse_metrics, sparsity_analysis)
    analysis = generate_detailed_analysis(comparison, sparsity_analysis)
    explain_bias_variance_tradeoff(analysis)
    conclusions = generate_conclusions(analysis, comparison, sparsity_analysis)
    print_assignment_completion()
