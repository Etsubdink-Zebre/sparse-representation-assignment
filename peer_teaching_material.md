# 🧠 Machine Learning Assignment 2: Sparse Representation Regularization

> **GROUP MEMBERS:** Etsubdink Zabre, Betel Negusu
> **ASSIGNED TECHNIQUE:** `#12 — Sparse Representation`

---

## 📚 Table of Contents

- [What is Sparse Representation?](#what-is-sparse-representation)
- [How It Works — Low-Level Mechanics](#how-it-works--low-level-mechanics)
- [Mathematical Formulation](#mathematical-formulation)
- [Purpose and Benefits](#purpose-and-benefits--with-mechanisms)
- [Bias-Variance Tradeoff](#bias-variance-tradeoff--detailed)
- [Limitations](#limitations--technical-analysis)
- [Comparison to Other Techniques](#comparison-to-other-regularization-techniques)
- [Best Use Cases](#best-use-cases)
- [Demonstration Outline](#demonstration-outline)
- [Key Takeaways](#key-takeaways)

---

## What is Sparse Representation?

In machine learning, a model trained on high-dimensional data faces a structural problem: **not all features carry signal**. Many are noise, redundancy, or irrelevant correlations. A model that assigns non-zero weights to all of them doesn't just waste computation — it actively learns noise, which hurts generalization.

**Sparse Representation** is a regularization strategy that constrains a model so that most of its learned coefficients are *exactly zero*, not approximately zero. The model is forced to "explain" the output using only a **small, selective subset** of the input features. This is not mere weight shrinkage — it is structural pruning of the parameter space *during training itself*.

The mathematical mechanism that produces this behavior is the **L1 norm penalty**, which has a geometric property that encourages solutions at the corners of a polytope — corners where coordinates are zero.

---

## How It Works — Low-Level Mechanics

### The Objective Function

In standard empirical risk minimization (ERM), a model minimizes only its loss:

```
Minimize: L(y, ŷ)
```

where `L` is typically Mean Squared Error (regression) or Cross-Entropy (classification), `y` is the true label, and `ŷ = f(X; w)` is the prediction parameterized by weight vector **w**.

Sparse Representation modifies this to a **regularized objective**:

```
Minimize: L(y, ŷ)  +  λ · Σᵢ |wᵢ|
```

The second term is the **L1 penalty** (also called the Lasso penalty) — the sum of absolute values of all model weights, scaled by hyperparameter `λ > 0`.

---

### Why L1 Produces Exact Zeros — The Geometry

> This is the core insight that distinguishes L1 from L2 regularization.

Consider a 2D weight space `(w₁, w₂)`. The loss function `L(w)` forms a smooth bowl centered at the unconstrained optimal point **w\***. The L1 constraint:

```
|w₁| + |w₂| ≤ t
```

defines a **diamond-shaped (rhombus) feasible region**. When the contours of the loss function are overlaid on this diamond, they almost always first touch it **at one of its corners** — and corners of the L1 diamond sit exactly on the coordinate axes, where one or more weights are **exactly zero**.

By contrast, the L2 constraint `w₁² + w₂² ≤ t` defines a **circle** with no corners. Loss contours intersect it at smooth, off-axis points — producing small but **non-zero** coefficients everywhere.

In `p` dimensions, the L1 ball is a **cross-polytope** with `2p` corner vertices, all on coordinate axes — explaining why L1 routinely zeros out features at scale.

---

### Gradient Perspective

The gradient of the L1 penalty with respect to `wᵢ` is:

```
∂/∂wᵢ (λ|wᵢ|) = λ · sign(wᵢ)
```

This gradient is **constant in magnitude** regardless of how small `wᵢ` gets (±λ everywhere except at 0). Even a tiny weight faces the same magnitude of pull toward zero.

With L2, the gradient of `λwᵢ²` is `2λwᵢ` — it **decays to zero** as the weight shrinks, so the weight is never fully eliminated.

Because the L1 penalty is **non-differentiable at zero**, it is optimized using subgradient methods or coordinate descent via the **soft-thresholding operator**:

```
wᵢ ← sign(wᵢ) · max(|wᵢ| − λ·α, 0)
```

where `α` is the learning rate. This operator explicitly sets any weight whose magnitude falls below `λ·α` to **exactly zero** at each update — producing hard sparsity.

---

## Mathematical Formulation

For a linear model with weight vector **w** ∈ ℝᵈ, the full **Lasso objective** is:

$$\hat{w} = \arg\min_w \frac{1}{2n} \sum_{i=1}^{n}(y_i - w^\top x_i)^2 + \lambda \sum_{j=1}^{d} |w_j|$$

**Terms defined:**

| Symbol | Meaning |
|--------|---------|
| `n` | Number of training samples |
| `d` | Number of features (dimensionality) |
| `xᵢ ∈ ℝᵈ` | Feature vector of sample `i` |
| `yᵢ` | True label of sample `i` |
| `wⱼ` | Weight (coefficient) for feature `j` |
| `λ ≥ 0` | Regularization strength (hyperparameter) |

> When `λ = 0` → reduces to **ordinary least squares**.
> As `λ → ∞` → all weights driven to **zero**.

For **logistic regression** (classification), the MSE loss is replaced by log-loss:

$$\hat{w} = \arg\min_w -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log\sigma(w^\top x_i) + (1-y_i)\log(1-\sigma(w^\top x_i))\right] + \lambda\sum_{j=1}^{d}|w_j|$$

where `σ(·)` is the sigmoid function.

---

## Purpose and Benefits — With Mechanisms

### 1. 🎯 Automatic Feature Selection
Non-zero entries of **ŵ** directly identify retained features. No separate selection step is needed — this is computationally equivalent to removing irrelevant columns from the design matrix during training.

### 2. 🛡️ Overfitting Prevention via Capacity Reduction
The VC dimension of a linear model grows with the number of non-zero parameters. By eliminating parameters, L1 reduces the model's **effective capacity** — bounding the generalization error gap between training and test loss.

### 3. 🔍 Interpretability
A model with 8 non-zero coefficients (out of 500 features) is humanly interpretable. Directly relevant in regulated domains (healthcare, finance) where explainability is legally required.

### 4. ⚡ Computational Efficiency at Inference
After training, inference requires only `dot(x_sparse, w_nonzero)` — operations proportional to the number of **retained** features, not total features. Critical when `d` is in the thousands or millions (e.g., NLP bag-of-words models).

### 5. 🔇 Noise and Redundancy Suppression
If a feature's marginal reduction in loss is less than `λ`, it is zeroed out — effectively filtering noise whose variance-to-signal ratio is too high to justify the L1 cost of keeping it active.

---

## Bias-Variance Tradeoff — Detailed

The expected test error of any model decomposes as:

```
E[Test Error] = Bias² + Variance + Irreducible Noise
```

**Effect of λ on the tradeoff:**

| λ Value | Effect | Bias | Variance | Outcome |
|---------|--------|------|----------|---------|
| `λ → 0` | No regularization | Low | **High** | Overfitting |
| `λ = λ*` (optimal) | Balanced sparsity | Moderate | Moderate | **Best generalization** |
| `λ → ∞` | All weights → 0 | **High** | ~0 | Underfitting |

**Key insight:**
- **Variance decreases** → model cannot freely overfit; the hypothesis space is constrained
- **Bias increases** → zeroing out features may eliminate some true signal
- Sparse Representation primarily **reduces variance**, more aggressively than L2 because it *eliminates* parameters entirely

> ✅ The optimal `λ*` is found via **k-fold cross-validation**: train on `k−1` folds, evaluate on the held-out fold, repeat `k` times, select `λ` minimizing average validation error.

---

## Limitations — Technical Analysis

### ⚠️ Instability with Correlated Features
If two features `x₁` and `x₂` are highly correlated (`corr(x₁, x₂) ≈ 1`), L1 will typically zero out one **arbitrarily** depending on numerical initialization. Running L1 on bootstrapped samples of the same dataset may select entirely different features.

### ⚠️ Group Effects Not Captured
When meaningful signal is distributed across a group of correlated features, L1 selects only one representative and discards the rest. **Elastic Net** addresses this by combining L1 and L2 penalties.

### ⚠️ Suboptimal When All Features Are Relevant
If the true generating process involves all `d` features (dense solution), L1's sparsity assumption is incorrect — it introduces unnecessary bias by zeroing out legitimate predictors.

### ⚠️ Hyperparameter Sensitivity
A factor-of-2 change in `λ` can cause large discrete jumps in the support of **ŵ**, making behavior hard to predict without systematic cross-validation.

### ⚠️ Non-trivial Optimization
Because the L1 objective is non-differentiable at zero, standard gradient descent cannot be applied directly. Solvers such as **coordinate descent**, **proximal gradient methods**, or **LARS** (Least Angle Regression) are required.

---

## Comparison to Other Regularization Techniques

| Technique | Penalty Term | Gradient Behavior | Coefficients | Best For |
|-----------|-------------|-------------------|--------------|----------|
| **L1 (Lasso / Sparse)** | `λ Σ\|wᵢ\|` | Constant magnitude (±λ) | **Exactly zero** for irrelevant features | Feature selection, high-dimensional data |
| **L2 (Ridge)** | `λ Σwᵢ²` | Proportional to weight (2λwᵢ) | Small but **non-zero** | Multicollinearity, dense solutions |
| **Elastic Net** | `αλΣ\|wᵢ\| + (1−α)λΣwᵢ²` | Hybrid | Some zero, others shrunk | Correlated features + sparsity desired |
| **Dropout (Neural Nets)** | Stochastic masking | N/A | Random zeros at train time | Deep networks |

> Elastic Net introduces mixing parameter `α ∈ [0,1]`:
> - `α = 1` → pure **Lasso**
> - `α = 0` → pure **Ridge**

---

## Best Use Cases

```
High-dimensional data (d >> n)
├── Genomics / Bioinformatics   → 20,000+ gene features, few samples
├── Text Classification (BoW)  → 50,000+ token vocabulary
├── Signal Processing           → Compressed sensing, MRI reconstruction
└── Regulated Domains           → Healthcare, finance (explainability required)
```

**When many features are expected to be irrelevant** — L1 identifies the sparse set without manual selection.

**When interpretability is critical** — a model with 10 non-zero features is auditable; a dense 10,000-feature model is not.

---

## Demonstration Outline

### Step 1 — Problem Setup
Generate a synthetic dataset:
- `n = 100` samples, `d = 500` features
- Only **10 features** carry true signal; the remaining 490 are pure Gaussian noise
- True weight vector **w\*** is sparse by construction

### Step 2 — Baseline: Unregularized Linear Regression
- Fit ordinary least squares → all 500 coefficients are non-zero
- Training MSE: **low** | Test MSE: **high** → clear overfitting
- Plot the dense, noisy coefficient vector

### Step 3 — Sparse Model: Lasso Regression
- Fit with `λ` selected via **5-fold cross-validation**
- Solver returns ≈ 10–15 non-zero coefficients
- Test MSE substantially lower; sparse coefficient vector aligns with true **w\***

### Step 4 — Comparison Table

| Model | Train MSE | Test MSE | Non-zero Coefficients |
|-------|-----------|----------|-----------------------|
| Unregularized | ~0.01 | ~2.40 | 500 |
| L2 Ridge | ~0.08 | ~0.60 | 500 |
| **L1 Lasso** | ~0.12 | **~0.18** | **12** |

### Step 5 — Regularization Path Analysis
Plot test MSE as a function of `λ`:

```
High MSE |  *                           (underfitting — too sparse)
         |    *
         |      *   ← optimal λ*
         |           *
         |               *  *  *  *     (overfitting — too dense)
Low MSE  +--------------------------------→ λ (decreasing)
```

Identify underfitting regime (large λ), overfitting regime (small λ), and the optimal valley.

---

## Key Takeaways

> 1. **L1 produces exact zeros** due to the geometry of the L1 ball and the constant-magnitude subgradient of the absolute value — not merely small weights.
> 2. **Sparse Representation = automatic feature selection** embedded in loss minimization, not a post-hoc filter.
> 3. **It primarily reduces variance** by limiting model capacity, at the cost of a modest increase in bias.
> 4. **Instability with correlated features** is a real limitation — Elastic Net is the standard remedy.
> 5. **Optimal λ must be tuned** via cross-validation; there is no universally correct value.

---

*Machine Learning Assignment — Group: Etsubdink Zabre & Betel Negusu*