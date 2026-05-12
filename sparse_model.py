"""
Sparse Model Module
===============================================================================

This module handles:
- Training sparse representation model with L1 regularization
- Evaluating sparse model performance
- Analyzing sparsity effects
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_sparse_model(X_train, y_train, C=0.1, random_state=42, max_iter=5000):
    """
    Train logistic regression model with L1 regularization for sparse representation
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        C (float): Regularization strength (smaller = stronger regularization)
        random_state (int): Random seed for reproducibility
        max_iter (int): Maximum iterations for convergence
    
    Returns:
        LogisticRegression: Trained sparse model
    """
    print("\nAPPLYING SPARSE REPRESENTATION (L1 REGULARIZATION)")
    print("-"*50)
    
    model = LogisticRegression(
        penalty='l1',           # L1 regularization for sparsity
        solver='liblinear',     # Required solver for L1 penalty
        C=C,                    # Small C = strong regularization
        max_iter=max_iter,
        random_state=random_state
    )
    
    print(f"Training sparse model with C={C}...")
    model.fit(X_train, y_train)
    print("+ Sparse model training completed")
    
    return model

def analyze_sparsity(model, total_features):
    """
    Analyze the sparsity of the trained model
    
    Args:
        model: Trained sparse model
        total_features (int): Total number of features
    
    Returns:
        dict: Sparsity analysis results
    """
    print("\n" + "-"*40)
    print("SPARSITY ANALYSIS")
    print("-"*40)
    
    # Count zero and non-zero coefficients
    zero_count = np.sum(model.coef_ == 0)
    non_zero_count = total_features - zero_count
    sparsity_ratio = zero_count / total_features
    
    analysis = {
        'total_features': total_features,
        'zero_coefficients': int(zero_count),
        'non_zero_coefficients': int(non_zero_count),
        'sparsity_ratio': sparsity_ratio,
        'feature_reduction_percentage': sparsity_ratio * 100
    }
    
    print(f"Total features: {analysis['total_features']}")
    print(f"Zero coefficients: {analysis['zero_coefficients']}")
    print(f"Non-zero coefficients: {analysis['non_zero_coefficients']}")
    print(f"Sparsity ratio: {analysis['sparsity_ratio']:.2%}")
    print(f"Feature reduction: {analysis['feature_reduction_percentage']:.1f}% of features eliminated")
    
    if analysis['sparsity_ratio'] > 0.5:
        print("+ High sparsity achieved - effective feature selection")
    else:
        print("? Low sparsity - may need stronger regularization")
    
    return analysis

def get_feature_importance(model, feature_names=None):
    """
    Get feature importance from sparse model coefficients
    
    Args:
        model: Trained sparse model
        feature_names (list, optional): Names of features
    
    Returns:
        dict: Feature importance information
    """
    coefficients = model.coef_[0]
    
    # Get non-zero features
    non_zero_indices = np.where(coefficients != 0)[0]
    non_zero_coeffs = coefficients[non_zero_indices]
    
    # Sort by absolute coefficient value
    sorted_indices = np.argsort(np.abs(non_zero_coeffs))[::-1]
    important_features = non_zero_indices[sorted_indices]
    important_coeffs = non_zero_coeffs[sorted_indices]
    
    importance_info = {
        'important_indices': important_features,
        'important_coefficients': important_coeffs,
        'num_important_features': len(important_features)
    }
    
    if feature_names is not None:
        importance_info['important_feature_names'] = [feature_names[i] for i in important_features]
    
    return importance_info

if __name__ == "__main__":
    # Test sparse model
    from data_preparation import load_and_prepare_data
    from baseline_model import evaluate_model
    
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    # Train sparse model
    model = train_sparse_model(X_train, y_train)
    
    # Evaluate sparse model
    metrics, train_pred, test_pred = evaluate_model(model, X_train, X_test, y_train, y_test, "Sparse")
    
    # Analyze sparsity
    sparsity = analyze_sparsity(model, X_train.shape[1])
    print(f"Sparse model metrics: {metrics}")
    print(f"Sparsity analysis: {sparsity}")
