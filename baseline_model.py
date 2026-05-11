"""
Baseline Model Module
===============================================================================

This module handles:
- Training baseline model without regularization
- Evaluating baseline model performance
- Demonstrating overfitting behavior
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_baseline_model(X_train, y_train, random_state=42, max_iter=5000):
    """
    Train baseline logistic regression model without regularization
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        random_state (int): Random seed for reproducibility
        max_iter (int): Maximum iterations for convergence
    
    Returns:
        LogisticRegression: Trained baseline model
    """
    print("\nBASELINE MODEL: NO REGULARIZATION (EXPECTED TO OVERFIT)")
    print("-"*50)
    
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=max_iter,
        random_state=random_state
    )
    
    print("Training baseline model...")
    model.fit(X_train, y_train)
    print("+ Baseline model training completed")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Baseline"):
    """
    Evaluate model performance using multiple metrics
    
    Args:
        model: Trained model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        model_name (str): Name of the model for display
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'train_precision': precision_score(y_train, train_pred),
        'test_precision': precision_score(y_test, test_pred),
        'train_recall': recall_score(y_train, train_pred),
        'test_recall': recall_score(y_test, test_pred),
        'train_f1': f1_score(y_train, train_pred),
        'test_f1': f1_score(y_test, test_pred),
        'overfitting_gap': accuracy_score(y_train, train_pred) - accuracy_score(y_test, test_pred)
    }
    
    # Display results
    print(f"\n{model_name.upper()} MODEL PERFORMANCE:")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f} | Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Train Precision: {metrics['train_precision']:.4f} | Test Precision: {metrics['test_precision']:.4f}")
    print(f"Train Recall: {metrics['train_recall']:.4f} | Test Recall: {metrics['test_recall']:.4f}")
    print(f"Train F1-Score: {metrics['train_f1']:.4f} | Test F1-Score: {metrics['test_f1']:.4f}")
    
    # Overfitting analysis
    print(f"\nOverfitting Gap (Train-Test Accuracy): {metrics['overfitting_gap']:.4f}")
    if metrics['overfitting_gap'] > 0.05:
        print("✓ Model shows signs of OVERFITTING (gap > 5%)")
    else:
        print("? Model may not be sufficiently overfitted")
    
    return metrics, train_pred, test_pred

def get_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix for evaluation"""
    return confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    # Test baseline model
    from data_preparation import load_and_prepare_data
    
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    model = train_baseline_model(X_train, y_train)
    metrics, train_pred, test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"Baseline metrics: {metrics}")
