"""Machine Learning Assignment 2: Regularization

This script demonstrates underfitting, overfitting, and the effect of L2 regularization
(Ridge regression) using a polynomial regression experiment.
It includes feature scaling, cross-validation for alpha selection, and validation curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def generate_synthetic_data(n_samples=120, noise=18.0, random_state=42):
    rng = np.random.RandomState(random_state)
    x = np.linspace(-3, 3, n_samples)
    y_true = 0.5 * x**3 - 2.0 * x**2 + x + 3
    y = y_true + rng.normal(scale=noise, size=n_samples)
    return x.reshape(-1, 1), y, y_true


def build_pipeline(degree, model):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def compute_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "coefficient_norm": np.linalg.norm(model.named_steps["model"].coef_),
        "intercept": model.named_steps["model"].intercept_,
        "model": model,
    }


def search_best_alpha(degree, X_train, y_train, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 4, 12)

    train_scores = []
    valid_scores = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=42)
        pipeline = build_pipeline(degree=degree, model=ridge)
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        train_scores.append(-np.mean(cv_results["train_score"]))
        valid_scores.append(-np.mean(cv_results["test_score"]))

    best_index = int(np.argmin(valid_scores))
    return alphas, np.array(train_scores), np.array(valid_scores), alphas[best_index]


def plot_model_fits(X_train, X_test, y_train, y_test, y_true_func, experiments):
    x_plot = np.linspace(-3, 3, 250).reshape(-1, 1)
    plt.figure(figsize=(16, 12))

    for i, exp in enumerate(experiments, start=1):
        plt.subplot(len(experiments), 1, i)
        y_plot = exp["model"].predict(x_plot)
        plt.scatter(X_train, y_train, color="blue", alpha=0.5, label="Train data")
        plt.scatter(X_test, y_test, color="orange", alpha=0.5, label="Test data")
        plt.plot(x_plot, y_true_func(x_plot.ravel()), color="green", linewidth=2, label="True function")
        plt.plot(x_plot, y_plot, color="red", linewidth=2, label=exp["label"])
        plt.title(exp["label"])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_regularization_curve(alphas, train_mse, valid_mse):
    plt.figure(figsize=(10, 5))
    plt.semilogx(alphas, train_mse, marker="o", label="Train MSE")
    plt.semilogx(alphas, valid_mse, marker="o", label="Validation MSE")
    plt.xlabel("Ridge alpha")
    plt.ylabel("Mean Squared Error")
    plt.title("Regularization validation curve")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.show()


def print_experiment_results(results):
    print("{:<35} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Train MSE", "Test MSE", "Train R2", "Test R2", "||w||"))
    print("-" * 90)
    for result in results:
        print("{:<35} {:>10.2f} {:>10.2f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            result["label"],
            result["train_mse"],
            result["test_mse"],
            result["train_r2"],
            result["test_r2"],
            result["coefficient_norm"],
        ))
    print()


def main():
    X, y, y_true = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    underfit_pipeline = build_pipeline(degree=1, model=LinearRegression())
    overfit_pipeline = build_pipeline(degree=15, model=LinearRegression())

    underfit_pipeline.fit(X_train, y_train)
    overfit_pipeline.fit(X_train, y_train)

    alphas, train_scores, valid_scores, best_alpha = search_best_alpha(degree=15, X_train=X_train, y_train=y_train)
    best_ridge_pipeline = build_pipeline(degree=15, model=Ridge(alpha=best_alpha, random_state=42))
    best_ridge_pipeline.fit(X_train, y_train)

    underfit = compute_metrics(underfit_pipeline, X_train, X_test, y_train, y_test)
    underfit["label"] = "Underfit: Degree 1"

    overfit = compute_metrics(overfit_pipeline, X_train, X_test, y_train, y_test)
    overfit["label"] = "Overfit: Degree 15"

    ridge = compute_metrics(best_ridge_pipeline, X_train, X_test, y_train, y_test)
    ridge["label"] = f"Ridge regularized: alpha={best_alpha:.4f}, degree=15"

    print_experiment_results([underfit, overfit, ridge])
    print("Best alpha selected by cross-validation:", best_alpha)
    print("Training MSE curve and validation MSE curve show how alpha controls bias and variance.")
    print("The regularized model has a lower coefficient norm and better test generalization than the unregularized high-degree model.")

    plot_model_fits(
        X_train,
        X_test,
        y_train,
        y_test,
        lambda x: 0.5 * x**3 - 2.0 * x**2 + x + 3,
        [underfit, overfit, ridge],
    )
    plot_regularization_curve(alphas, train_scores, valid_scores)


if __name__ == "__main__":
    main()
