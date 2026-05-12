# Sparse Representation Assignment

Machine Learning Assignment 2: Regularization

## Setup Instructions

### 1. Install Packages (System Python)
```bash
# Install required packages using system Python
python -m pip install -r requirements.txt
```

### 2. Run the Assignment
```bash
python main.py
```

### 📝 Note: Virtual Environment Issues
- Virtual environment creation failed due to system permissions
- Assignment runs successfully with system Python
- All required packages are installed in system Python environment

## Project Structure

This assignment is organized into modular files for better maintainability:

```
Sparse_Representation_Assignment/
├── main.py                       # MAIN ORCHESTRATOR - RUN THIS FILE
├── peer_teaching_material.md    # Educational content (markdown)
├── data_preparation.py          # Data loading and overfitting setup
├── baseline_model.py            # Baseline model without regularization
├── sparse_model.py              # Sparse representation model with L1
├── analysis.py                  # Performance comparison and conclusions
├── visualizations.py            # Comprehensive charts and plots
├── requirements.txt             # Package dependencies
└── README.md                    # This file
```

### Assignment Requirements Coverage

| Requirement | File(s) | Status |
|-------------|---------|---------|
| **1. Peer Teaching Material** | `peer_teaching_material.md` | Complete |
| **2. Overfitting Implementation** | `data_preparation.py`, `baseline_model.py` | Complete |
| **3. Sparse Representation Application** | `sparse_model.py` | Complete |
| **4. Comparison & Analysis** | `analysis.py` | Complete |

### 📊 What the Code Does

1. **Creates Overfitting Scenario**: 
   - Uses breast cancer dataset (30 original features)
   - Adds 200 noise features → 230 total features
   - 398 training samples vs 230 features (high overfitting risk)

2. **Trains Two Models**:
   - **Baseline**: No regularization (expected to overfit)
   - **Sparse**: L1 regularization (automatic feature selection)

3. **Comprehensive Evaluation**:
   - Multiple metrics: Accuracy, Precision, Recall, F1-Score
   - Overfitting gap analysis
   - Sparsity analysis (feature elimination)

4. **Rich Visualizations**:
   - Accuracy comparison charts
   - Multi-metrics comparison
   - Overfitting gap analysis
   - Sparsity pie chart
   - Confusion matrices

### 🔍 Key Features

- **Modular Design**: Each component in separate files for clarity
- **Educational Content**: Comprehensive peer teaching material in markdown
- **Automatic Analysis**: Code explains results and provides insights
- **Professional Visualizations**: Multiple charts with detailed analysis
- **Bias-Variance Explanation**: Clear tradeoff analysis

### 📈 Functional Results

**Actual Performance Achieved:**
- **Baseline Model**: Train: 100.0%, Test: 96.5% (3.5% overfitting gap)
- **Sparse Model**: Train: 97.5%, Test: 97.7% (-0.2% gap - no overfitting)
- **Feature Elimination**: 96.1% sparsity (221/230 features removed)
- **Performance Gain**: +1.21% test accuracy, -3.68% overfitting reduction

### 🔍 What the Code Actually Does

**1. Data Preparation (`data_preparation.py`)**
- Loads breast cancer dataset (30 original features)
- Adds 200 random noise features → 230 total features
- Creates high-dimensional scenario (398 samples vs 230 features)
- Splits data 70/30 train/test with StandardScaler

**2. Baseline Model (`baseline_model.py`)**
- Trains Logistic Regression without regularization
- Evaluates with Accuracy, Precision, Recall, F1-Score
- Demonstrates overfitting with perfect train accuracy

**3. Sparse Representation (`sparse_model.py`)**
- Applies L1 regularization (C=0.1) for automatic feature selection
- Achieves 96.1% feature elimination
- Maintains/improves test performance while reducing overfitting

**4. Analysis (`analysis.py`)**
- Compares baseline vs sparse model performance
- Explains bias-variance tradeoff improvements
- Provides detailed conclusions and recommendations

**5. Visualizations (`visualizations.py`)**
- Generates 4 comprehensive charts:
  - Accuracy comparison (train/test)
  - Multi-metrics comparison (precision, recall, F1)
  - Overfitting gap analysis
  - Sparsity pie chart (221/230 features eliminated)
  - Confusion matrices comparison

### 🎯 Key Achievements

✅ **Peer Teaching Material**: Complete markdown documentation  
✅ **Overfitting Implementation**: High-risk scenario with 230 features  
✅ **Sparse Representation**: 96.1% automatic feature elimination  
✅ **Performance Analysis**: Comprehensive before/after comparison  

### � Technical Implementation

- **Algorithm**: Logistic Regression with L1 regularization
- **Regularization Strength**: C=0.1 (optimal for this dataset)
- **Dataset**: Breast Cancer (sklearn.datasets.load_breast_cancer)
- **Noise Addition**: 200 Gaussian features to create overfitting
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: matplotlib charts with detailed analysis

### 🚀 Running Instructions

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run complete assignment
python main.py
```

The code will automatically:
1. Display peer teaching material
2. Prepare data with overfitting scenario
3. Train baseline and sparse models
4. Generate comprehensive analysis
5. Create all visualizations
6. Print detailed results and conclusions

---

**Group Members**: [Etsubdink Zebre], [Betel Nigusu]  
**Assigned Technique**: #12 - Sparse Representation  
**Course**: Machine Learning - Assignment 2
