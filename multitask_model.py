"""
Multi-Task Learning Implementation
Technique #13 - Multi-Task Learning

Multi-Task Learning (MTL) is a machine learning paradigm where multiple related tasks 
are learned simultaneously, exploiting commonalities and differences across tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs('dashboard/assets', exist_ok=True)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def create_multitask_dataset():
    """
    Create a multi-task learning scenario from breast cancer dataset
    Task 1: Tumor classification (original binary classification)
    Task 2: Severity prediction (derived from tumor characteristics)
    Task 3: Risk assessment (derived from multiple features)
    """
    print("CREATING MULTI-TASK LEARNING DATASET")
    print("="*50)
    
    # Load original dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Original features: {X.shape[1]}")
    
    # Add noise features for complexity
    np.random.seed(42)
    noise_features = np.random.randn(X.shape[0], 200)
    X_extended = np.hstack([X, noise_features])
    
    print(f"After adding noise: {X_extended.shape}")
    print(f"Total features: {X_extended.shape[1]} (Original: {X.shape[1]}, Noise: 200)")
    
    # Create multiple tasks
    # Task 1: Original classification (benign/malignant)
    task1_y = y
    
    # Task 2: Severity prediction (based on feature means)
    feature_means = np.mean(X, axis=1)
    task2_y = (feature_means > np.median(feature_means)).astype(int)
    
    # Task 3: Risk assessment (based on multiple features)
    risk_scores = X[:, 0] + X[:, 1] + X[:, 2]  # Using first 3 features
    task3_y = (risk_scores > np.median(risk_scores)).astype(int)
    
    print(f"\nMulti-Task Setup:")
    print(f"Task 1 - Classification: {np.sum(task1_y)} positive, {len(task1_y)-np.sum(task1_y)} negative")
    print(f"Task 2 - Severity: {np.sum(task2_y)} high, {len(task2_y)-np.sum(task2_y)} low")
    print(f"Task 3 - Risk: {np.sum(task3_y)} high, {len(task3_y)-np.sum(task3_y)} low")
    
    return X_extended, task1_y, task2_y, task3_y

def train_single_task_models(X, task1_y, task2_y, task3_y):
    """
    Train separate models for each task (baseline approach)
    """
    print("\nTRAINING SINGLE-TASK MODELS (BASELINE)")
    print("-"*40)
    
    # Split data
    X_train, X_test, y1_train, y1_test = train_test_split(X, task1_y, test_size=0.3, random_state=42)
    _, _, y2_train, y2_test = train_test_split(X, task2_y, test_size=0.3, random_state=42)
    _, _, y3_train, y3_test = train_test_split(X, task3_y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train separate models
    models = {}
    
    for i, (y_train, y_test, task_name) in enumerate([
        (y1_train, y1_test, "Classification"),
        (y2_train, y2_test, "Severity"),
        (y3_train, y3_test, "Risk")
    ]):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        models[task_name] = {
            'model': model,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"+ {task_name} model trained - Accuracy: {accuracy:.4f}")
    
    return models, scaler

def train_multitask_model(X, task1_y, task2_y, task3_y):
    """
    Train multi-task learning model (shared representation)
    """
    print("\nTRAINING MULTI-TASK MODEL")
    print("-"*40)
    
    # Split data
    X_train, X_test, y1_train, y1_test = train_test_split(X, task1_y, test_size=0.3, random_state=42)
    _, _, y2_train, y2_test = train_test_split(X, task2_y, test_size=0.3, random_state=42)
    _, _, y3_train, y3_test = train_test_split(X, task3_y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create multi-task targets (simplified approach)
    # In practice, you'd use more sophisticated MTL architectures
    # Here we use a shared representation approach
    
    multitask_model = {
        'shared_model': LogisticRegression(max_iter=1000, random_state=42),
        'task_specific_models': {}
    }
    
    # Train shared model on all tasks combined
    all_y_train = np.concatenate([y1_train, y2_train, y3_train])
    all_X_train = np.tile(X_train_scaled, (3, 1))
    
    multitask_model['shared_model'].fit(all_X_train, all_y_train)
    
    # Fine-tune for each task
    for y_train, y_test, task_name in [
        (y1_train, y1_test, "Classification"),
        (y2_train, y2_test, "Severity"), 
        (y3_train, y3_test, "Risk")
    ]:
        # Start with shared model weights
        task_model = LogisticRegression(max_iter=1000, random_state=42)
        task_model.coef_ = multitask_model['shared_model'].coef_.copy()
        task_model.intercept_ = multitask_model['shared_model'].intercept_.copy()
        
        # Fine-tune on task-specific data
        task_model.fit(X_train_scaled, y_train)
        
        y_pred = task_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        multitask_model['task_specific_models'][task_name] = {
            'model': task_model,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"+ Multi-task {task_name} - Accuracy: {accuracy:.4f}")
    
    return multitask_model, scaler

def compare_approaches(single_models, multitask_model):
    """
    Compare single-task vs multi-task learning approaches
    """
    print("\nMULTI-TASK LEARNING ANALYSIS")
    print("="*50)
    
    print("\nSINGLE-TASK vs MULTI-TASK COMPARISON:")
    print("-"*50)
    
    tasks = ["Classification", "Severity", "Risk"]
    
    single_accuracies = []
    multitask_accuracies = []
    
    for task in tasks:
        single_acc = single_models[task]['accuracy']
        multitask_acc = multitask_model['task_specific_models'][task]['accuracy']
        
        single_accuracies.append(single_acc)
        multitask_accuracies.append(multitask_acc)
        
        improvement = multitask_acc - single_acc
        print(f"{task}:")
        print(f"  Single-Task: {single_acc:.4f}")
        print(f"  Multi-Task:  {multitask_acc:.4f}")
        print(f"  Improvement:  {improvement:+.4f}")
        print()
    
    # Overall comparison
    avg_single = np.mean(single_accuracies)
    avg_multitask = np.mean(multitask_accuracies)
    
    print(f"OVERALL PERFORMANCE:")
    print(f"  Average Single-Task: {avg_single:.4f}")
    print(f"  Average Multi-Task:  {avg_multitask:.4f}")
    print(f"  Average Improvement:   {avg_multitask - avg_single:+.4f}")
    
    return {
        'single_accuracies': single_accuracies,
        'multitask_accuracies': multitask_accuracies,
        'avg_single': avg_single,
        'avg_multitask': avg_multitask,
        'improvement': avg_multitask - avg_single
    }

def visualize_multitask_results(comparison_results):
    """
    Create visualizations for multi-task learning comparison
    """
    print("\nGENERATING MULTI-TASK VISUALIZATIONS")
    print("="*50)
    
    tasks = ["Classification", "Severity", "Risk"]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar comparison
    x = np.arange(len(tasks))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_results['single_accuracies'], width, 
              label='Single-Task', alpha=0.8, color='blue')
    ax1.bar(x + width/2, comparison_results['multitask_accuracies'], width,
              label='Multi-Task', alpha=0.8, color='orange')
    
    ax1.set_xlabel('Tasks')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Single-Task vs Multi-Task Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average performance comparison
    approaches = ['Single-Task', 'Multi-Task']
    averages = [comparison_results['avg_single'], comparison_results['avg_multitask']]
    colors = ['blue', 'orange']
    
    bars = ax2.bar(approaches, averages, color=colors, alpha=0.8)
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Overall Performance Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{avg:.3f}', ha='center', va='bottom')
    
    # 3. Improvement visualization
    improvements = [comparison_results['single_accuracies'][i] - comparison_results['multitask_accuracies'][i] 
                 for i in range(len(tasks))]
    
    colors_improvement = ['green' if imp >= 0 else 'red' for imp in improvements]
    bars3 = ax3.bar(tasks, improvements, color=colors_improvement, alpha=0.8)
    ax3.set_ylabel('Accuracy Difference')
    ax3.set_title('Multi-Task Improvement per Task')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary pie chart
    better_count = sum(1 for imp in improvements if imp > 0)
    worse_count = len(improvements) - better_count
    
    sizes = [better_count, worse_count]
    labels = [f'Improved ({better_count})', f'Worse ({worse_count})']
    colors = ['green', 'red']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Multi-Task Impact Summary')
    
    plt.tight_layout()
    plt.savefig('dashboard/assets/multitask_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to demonstrate multi-task learning
    """
    print("MULTI-TASK LEARNING DEMONSTRATION")
    print("="*50)
    print("Technique #13 - Multi-Task Learning")
    print()
    
    # Create multi-task dataset
    X, task1_y, task2_y, task3_y = create_multitask_dataset()
    
    # Train single-task models
    single_models, scaler1 = train_single_task_models(X, task1_y, task2_y, task3_y)
    
    # Train multi-task model
    multitask_model, scaler2 = train_multitask_model(X, task1_y, task2_y, task3_y)
    
    # Compare approaches
    comparison_results = compare_approaches(single_models, multitask_model)
    
    # Create visualizations
    visualize_multitask_results(comparison_results)
    
    print("\nMULTI-TASK LEARNING CONCLUSIONS:")
    print("="*50)
    print("Multi-Task Learning Benefits:")
    print("+ Shared representations improve generalization")
    print("+ Knowledge transfer between related tasks")
    print("+ Reduced overfitting through regularization across tasks")
    print("+ More efficient use of training data")
    print()
    print("When Multi-Task Learning is Most Effective:")
    print("+ Tasks are related (e.g., medical diagnosis tasks)")
    print("+ Limited training data per task")
    print("+ Shared underlying patterns exist")
    print("+ Computational efficiency is important")
    
    return {
        'single_models': single_models,
        'multitask_model': multitask_model,
        'comparison': comparison_results
    }

if __name__ == "__main__":
    results = main()
