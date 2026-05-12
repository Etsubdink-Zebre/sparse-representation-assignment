"""
Data Preparation Module
===============================================================================

This module handles:
- Loading the breast cancer dataset
- Adding noise features to create overfitting scenario
- Data splitting and scaling
- Preparing data for both baseline and sparse models
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(noise_features=200, test_size=0.3, random_state=42):
    """
    Load breast cancer dataset and add noise features for overfitting scenario
    
    Args:
        noise_features (int): Number of noise features to add
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("="*60)
    print("DATA PREPARATION: OVERFITTING SETUP")
    print("="*60)
    
    # Load original dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Original features: {X.shape[1]}")
    
    # Add noise features to create overfitting scenario
    np.random.seed(random_state)
    noise = np.random.randn(X.shape[0], noise_features)
    
    # Combine original features with noise
    X = np.hstack((X, noise))
    
    print(f"\nAfter adding noise:")
    print(f"New dataset shape: {X.shape}")
    print(f"Original features: {30}, Noise features: {noise_features}, Total: {X.shape[1]}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nData split:")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nData scaled using StandardScaler")
    
    return X_train, X_test, y_train, y_test, scaler

def get_dataset_info(X_train, X_test):
    """Get information about the prepared dataset"""
    total_features = X_train.shape[1]
    train_samples = X_train.shape[0]
    test_samples = X_test.shape[0]
    
    return {
        "total_features": total_features,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "feature_to_sample_ratio": total_features / train_samples,
        "high_risk_overfitting": total_features > train_samples / 2
    }

if __name__ == "__main__":
    # Test data preparation
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    info = get_dataset_info(X_train, X_test)
    print(f"\nDataset Info: {info}")
