# Machine Learning Assignment 2: Regularization

## 1. Peer teaching material: L2 regularization (Ridge regression)

### Purpose
- Regularization reduces overfitting by penalizing large weights and discouraging overly complex functions.
- Ridge regression improves generalization by shrinking the model coefficients toward zero.
- This is especially important for polynomial regression and any model with many degrees of freedom.

### How it works
- Ordinary least squares minimizes the squared training error:
  - `J(w) = ||y - Xw||^2`
- Ridge adds an L2 penalty to the cost:
  - `J_{ridge}(w) = ||y - Xw||^2 + alpha * ||w||^2`
- The regularization term `alpha * ||w||^2` reduces coefficient values but does not force them to zero.
- When `alpha` increases:
  - model complexity decreases
  - variance decreases
  - bias increases

### Bias and variance tradeoff
- **Bias** is error due to assumptions that are too strong (model too simple).
- **Variance** is error due to sensitivity to noise in the training data (model too complex).
- Underfitting corresponds to high bias and low variance.
- Overfitting corresponds to low bias and high variance.
- L2 regularization shifts the model toward a better balance by increasing bias slightly while reducing variance.

### Best use cases
- Polynomial regression with high-degree feature expansion.
- Regression tasks with many correlated or scaled features.
- Models where stability is more important than exact training fit.
- Datasets with moderate noise and limited training samples.

### Limitations
- Ridge does not select features; it shrinks all coefficients rather than making them zero.
- The regularization strength `alpha` must be tuned using validation or cross-validation.
- Too much regularization may cause underfitting by oversmoothing the model.
- It is less appropriate when a sparse solution is needed; in that case, L1 regularization is better.

## 2. Implementation approach and experiment design

### Problem statement
- Regression on a synthetic cubic function contaminated with Gaussian noise.
- The objective is to show how model complexity and regularization affect performance.

### Experimental setup
- Underfitting model: degree-1 polynomial regression (linear fit).
- Overfitting model: degree-15 polynomial regression without regularization.
- Regularized model: degree-15 polynomial regression with Ridge regression.
- Feature scaling is applied after polynomial expansion using `StandardScaler`.
- Alpha is selected by cross-validation from a logarithmic grid.

### Evaluation metrics
- Mean Squared Error (MSE)
- R-squared score (R2)
- Coefficient norm `||w||` to measure model complexity

## 3. Advanced regularization analysis

### Cross-validation and alpha selection
- We evaluate a set of `alpha` values on the training set with 5-fold cross-validation.
- The selected `alpha` is the one that minimizes validation MSE.
- This is more advanced than choosing a single fixed alpha arbitrarily.

### Validation curve
- The script plots training MSE and validation MSE as functions of `alpha`.
- This visualization makes the bias-variance tradeoff explicit:
  - low alpha leads to low bias but high variance,
  - high alpha leads to high bias but low variance.

## 4. Results and conclusions

### What to compare
- Underfit model: low complexity, poor training fit, moderate test performance.
- Overfit model: high complexity, excellent training fit, poor generalization.
- Ridge regularized model: high capacity with penalty, better test generalization.

### Why regularization improves performance
- The unregularized degree-15 model fits noise and shows unstable coefficients.
- Ridge shrinks coefficients and reduces oscillation in the fitted curve.
- As a result, the regularized model better approximates the true underlying function on unseen data.

### Practical conclusion
- L2 regularization is a strong tool for controlling overfitting in high-capacity models.
- Good teaching points:
  - show the training/test curves side by side,
  - highlight the chosen `alpha`,
  - explain coefficient norm reduction,
  - relate the effect to bias-variance tradeoff.

## 5. How to run

```bash
pip install -r requirements.txt
python assignment2_regularization.py
```

The script prints a performance comparison table and shows two plots:
- polynomial fit for underfit, overfit, and regularized models
- validation curve for Ridge alpha selection

## 6. Teacher notes

- Use the plots to explain why the degree-15 model overfits and why Ridge helps.
- Emphasize that the best model is not the one with the lowest training error.
- Show that regularization is a general technique used across many ML models, including neural networks and multi-task learning.
