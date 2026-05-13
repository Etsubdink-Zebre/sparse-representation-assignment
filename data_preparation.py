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
import json
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def export_dataset_preview(data, output_path='dashboard/assets/imported_dataset.json'):
    """Export imported dataset for dashboard table display."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    feature_names = list(data.feature_names)
    rows = []
    for idx, sample in enumerate(data.data):
        row = {feature_names[i]: float(sample[i]) for i in range(len(feature_names))}
        row["target"] = int(data.target[idx])
        row["target_label"] = "malignant" if int(data.target[idx]) == 0 else "benign"
        rows.append(row)

    payload = {
        "name": "breast_cancer",
        "description": "Original imported sklearn breast cancer dataset (before synthetic noise injection).",
        "row_count": len(rows),
        "columns": feature_names + ["target", "target_label"],
        "rows": rows
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    # Fallback for file:// mode where fetch() may be blocked.
    js_output_path = output_path.replace('.json', '.js')
    with open(js_output_path, 'w', encoding='utf-8') as f:
        f.write("window.IMPORTED_DATASET = ")
        json.dump(payload, f)
        f.write(";")

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

    # Export dataset preview used by the dashboard "Dataset Table" tab.
    export_dataset_preview(data)
    
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
