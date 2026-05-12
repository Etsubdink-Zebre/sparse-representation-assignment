# Machine Learning Assignment 2: Sparse Representation Regularization

**GROUP MEMBERS:** [Student 1 Name], [Student 2 Name]  
**ASSIGNED TECHNIQUE:** #12 - Sparse Representation

---

## 1. PEER TEACHING MATERIAL: SPARSE REPRESENTATION (5pts)

### WHAT IS SPARSE REPRESENTATION?

Sparse Representation is a regularization technique that encourages models to use only a small subset of available features. It achieves this by driving many model coefficients to exactly zero, effectively performing automatic feature selection.

### HOW IT WORKS?

Sparse Representation typically uses L1 regularization (Lasso penalty):

- **Adds penalty term:** `lambda * Sum(|wi|)` where `wi` are model coefficients
- **L1 norm** pushes less important features to exactly zero (not just small values)
- **Unlike L2 (Ridge)** which only shrinks coefficients, L1 eliminates them completely
- **Result:** Only the most important features remain in the model

### MATHEMATICAL FORMULATION:

```
Minimize: Loss(y, y_hat) + lambda * Sum(|wi|)
where lambda controls the strength of sparsity
```

### PURPOSE AND BENEFITS:

1. **Feature Selection:** Automatically identifies most important features
2. **Overfitting Prevention:** Reduces model complexity by eliminating irrelevant features
3. **Interpretability:** Easier to understand which features drive predictions
4. **Computational Efficiency:** Fewer features = faster prediction
5. **Noise Robustness:** Ignores noisy/redundant features

### BEST USE CASES:

- **High-dimensional data** (many features, few samples)
- When many features are expected to be irrelevant
- **Text classification** with large vocabularies
- **Genomics/bioinformatics** with thousands of genes
- **Image processing** with many pixels
- Any domain where **interpretability** is important

### LIMITATIONS:

- Can be **unstable with correlated features** (may select one arbitrarily)
- May **oversimplify** when many features have small but meaningful effects
- Requires **careful tuning** of regularization strength (lambda)
- **Not suitable** when all features contribute meaningfully
- Can be **computationally expensive** for very large datasets

### BIAS-VARIANCE TRADEOFF:

- **High Bias (Underfitting):** Too much sparsity (large lambda) → model too simple
- **High Variance (Overfitting):** Too little sparsity (small lambda) → model too complex
- **Optimal Balance:** Right amount of sparsity → good generalization
- **Sparse Representation** primarily reduces **VARIANCE** by reducing model complexity
- May increase **BIAS** slightly by eliminating some useful features
- The key is finding the sweet spot where variance reduction outweighs bias increase

### COMPARISON TO OTHER REGULARIZATION:

| Technique | Effect on Coefficients | Key Feature |
|-----------|------------------------|-------------|
| **L1 (Sparse)** | Coefficients become **zero** | Feature selection |
| **L2 (Ridge)** | Coefficients shrink but remain non-zero | Coefficient shrinkage |
| **Elastic Net** | Combination of L1 and L2 | Best of both worlds |

---

## KEY TAKEAWAYS FOR PEER TEACHING:

1. **Automatic Feature Selection:** L1 regularization eliminates irrelevant features automatically
2. **Overfitting Solution:** Particularly effective in high-dimensional data
3. **Interpretability:** Results are easier to explain with fewer features
4. **Trade-off Management:** Balances model complexity and generalization
5. **Practical Application:** Widely used in genomics, text processing, and signal processing

---

## DEMONSTRATION OUTLINE:

1. **Problem Setup:** High-dimensional data with many irrelevant features
2. **Baseline Model:** Shows overfitting without regularization
3. **Sparse Model:** Applies L1 regularization for feature selection
4. **Comparison:** Demonstrates improved generalization
5. **Analysis:** Explains bias-variance tradeoff improvements
