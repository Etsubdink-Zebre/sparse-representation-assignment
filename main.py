"""
Main Orchestrator for Sparse Representation Assignment
===============================================================================

This file orchestrates the complete sparse representation assignment by:
1. Displaying peer teaching material
2. Preparing data with overfitting scenario
3. Training baseline and sparse models
4. Analyzing and comparing performance
5. Generating comprehensive visualizations

Run this file to execute the complete assignment.
"""

# Import all modules
def display_peer_teaching_material():
    """Display peer teaching material from markdown file"""
    try:
        with open('peer_teaching_material.md', 'r', encoding='utf-8') as f:
            content = f.read()
        # Replace problematic Unicode characters with ASCII alternatives
        content = content.replace('→', '->')
        content = content.replace('←', '<-')
        content = content.replace('•', '-')
        content = content.replace('✓', '+')
        content = content.replace('✗', 'x')
        print(content)
    except FileNotFoundError:
        print("Peer teaching material not found. Please ensure peer_teaching_material.md exists.")
from data_preparation import load_and_prepare_data, get_dataset_info
from baseline_model import train_baseline_model, evaluate_model
from sparse_model import train_sparse_model, analyze_sparsity
from analysis import compare_performance, generate_detailed_analysis, explain_bias_variance_tradeoff, generate_conclusions, print_assignment_completion
from visualizations import create_comprehensive_comparison, create_confusion_matrices, print_visualization_summary

def main():
    """Main function to run the complete sparse representation assignment"""
    
    # 1. Display peer teaching material
    display_peer_teaching_material()
    
    # 2. Data preparation with overfitting scenario
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    dataset_info = get_dataset_info(X_train, X_test)
    
    print(f"\nDataset Risk Assessment:")
    print(f"Feature-to-sample ratio: {dataset_info['feature_to_sample_ratio']:.2f}")
    if dataset_info['high_risk_overfitting']:
        print("! High risk of overfitting detected - perfect for demonstration!")
    
    # 3. Train baseline model (overfitted)
    baseline_model = train_baseline_model(X_train, y_train)
    baseline_metrics, baseline_train_pred, baseline_test_pred = evaluate_model(
        baseline_model, X_train, X_test, y_train, y_test, "Baseline"
    )
    
    # 4. Train sparse model with L1 regularization
    sparse_model = train_sparse_model(X_train, y_train)
    sparse_metrics, sparse_train_pred, sparse_test_pred = evaluate_model(
        sparse_model, X_train, X_test, y_train, y_test, "Sparse"
    )
    
    # 5. Analyze sparsity
    sparsity_analysis = analyze_sparsity(sparse_model, X_train.shape[1])
    
    # 6. Performance comparison and analysis
    comparison = compare_performance(baseline_metrics, sparse_metrics, sparsity_analysis)
    analysis = generate_detailed_analysis(comparison, sparsity_analysis)
    explain_bias_variance_tradeoff(analysis)
    conclusions = generate_conclusions(analysis, comparison, sparsity_analysis)
    
    # 7. Generate visualizations
    create_comprehensive_comparison(baseline_metrics, sparse_metrics, sparsity_analysis)
    create_confusion_matrices(y_test, baseline_test_pred, sparse_test_pred)
    print_visualization_summary()
    
    # 8. Assignment completion
    print_assignment_completion()
    
    return {
        'baseline_metrics': baseline_metrics,
        'sparse_metrics': sparse_metrics,
        'sparsity_analysis': sparsity_analysis,
        'comparison': comparison,
        'analysis': analysis,
        'conclusions': conclusions
    }

if __name__ == "__main__":
    results = main()
