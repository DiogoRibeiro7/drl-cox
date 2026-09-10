"""
Demonstration of the scikit-learn compatible DRLCoxEstimator API.

This script shows how to use DRL-Cox with scikit-learn pipelines,
cross-validation, and hyperparameter tuning.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drl_cox import (
    DRLCoxEstimator,
    concordance_index,
    make_drl_cox_scorer,
    simulate_cox_data,
)


def example_1_basic_usage():
    """Example 1: Basic usage of DRLCoxEstimator."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Generate synthetic data
    print("\nGenerating synthetic survival data...")
    data = simulate_cox_data(n=200, d=10, seed=42)
    X, y, zeta = data.X, data.y, data.zeta
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Events: {np.sum(zeta)} ({100 * np.mean(zeta):.1f}%)")

    # Create and fit estimator
    print("\nFitting DRL-Cox model...")
    model = DRLCoxEstimator(epsilon=0.1, gamma=3)
    model.fit(X, y, zeta)

    print("✓ Model fitted successfully")
    print(f"  Status: {model.status_}")
    print(f"  Objective: {model.objective_value_:.6f}")

    # Make predictions
    risk_scores = model.predict(X)
    print(f"\nRisk scores shape: {risk_scores.shape}")
    print(f"Risk scores range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")

    # Evaluate
    c_index = model.score(X, y, zeta)
    print(f"\nC-index: {c_index:.4f}")

    print()


def example_2_train_test_split():
    """Example 2: Train/test split evaluation."""
    print("=" * 80)
    print("EXAMPLE 2: Train/Test Split")
    print("=" * 80)

    # Generate data
    data = simulate_cox_data(n=300, d=15, seed=123)
    X, y, zeta = data.X, data.y, data.zeta

    # Split data
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test, zeta_train, zeta_test = train_test_split(
        X, y, zeta, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    # Fit model
    print("\nFitting model on training data...")
    model = DRLCoxEstimator(epsilon=0.1)
    model.fit(X_train, y_train, zeta_train)

    # Evaluate on both sets
    train_score = model.score(X_train, y_train, zeta_train)
    test_score = model.score(X_test, y_test, zeta_test)

    print("\nResults:")
    print(f"  Train C-index: {train_score:.4f}")
    print(f"  Test C-index:  {test_score:.4f}")
    print(f"  Difference:    {train_score - test_score:.4f}")

    print()


def example_3_pipeline():
    """Example 3: Using DRL-Cox in a scikit-learn pipeline."""
    print("=" * 80)
    print("EXAMPLE 3: Pipeline Integration")
    print("=" * 80)

    # Generate data
    data = simulate_cox_data(n=200, d=10, seed=42)
    X, y, zeta = data.X, data.y, data.zeta

    # Create pipeline
    print("\nCreating pipeline with StandardScaler + DRLCoxEstimator...")
    pipeline = Pipeline([("scaler", StandardScaler()), ("drl_cox", DRLCoxEstimator(epsilon=0.1))])

    print("Pipeline steps:")
    for name, step in pipeline.steps:
        print(f"  - {name}: {step.__class__.__name__}")

    # Fit pipeline
    print("\nFitting pipeline...")
    pipeline.fit(X, y, drl_cox__zeta=zeta)
    print("✓ Pipeline fitted")

    # Predict and evaluate
    risk_scores = pipeline.predict(X)
    c_index = concordance_index(risk_scores, y, zeta)

    print("\nResults:")
    print(f"  C-index: {c_index:.4f}")

    # Get pipeline parameters
    print("\nPipeline parameters:")
    params = pipeline.get_params()
    for key in ["drl_cox__epsilon", "drl_cox__gamma", "drl_cox__solver"]:
        if key in params:
            print(f"  {key}: {params[key]}")

    print()


def example_4_parameter_tuning():
    """Example 4: Manual hyperparameter tuning."""
    print("=" * 80)
    print("EXAMPLE 4: Hyperparameter Tuning")
    print("=" * 80)

    # Generate data
    data = simulate_cox_data(n=250, d=12, seed=42)
    X, y, zeta = data.X, data.y, data.zeta

    # Split data
    X_train, X_val, y_train, y_val, zeta_train, zeta_val = train_test_split(
        X, y, zeta, test_size=0.3, random_state=42
    )

    print("\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")

    # Define parameter grid
    param_grid = {"epsilon": [0.0, 0.05, 0.1, 0.2, 0.3], "gamma": [2, 3, 4]}

    print("\nParameter grid:")
    print(f"  epsilon: {param_grid['epsilon']}")
    print(f"  gamma: {param_grid['gamma']}")
    print(f"  Total combinations: {len(param_grid['epsilon']) * len(param_grid['gamma'])}")

    # Manual grid search
    print("\nPerforming grid search...")
    results = []

    for epsilon in param_grid["epsilon"]:
        for gamma in param_grid["gamma"]:
            # Fit model
            model = DRLCoxEstimator(epsilon=epsilon, gamma=gamma)
            model.fit(X_train, y_train, zeta_train)

            # Evaluate on validation set
            val_score = model.score(X_val, y_val, zeta_val)

            results.append({"epsilon": epsilon, "gamma": gamma, "val_score": val_score})

            print(f"  ε={epsilon:.2f}, γ={gamma}: C-index={val_score:.4f}")

    # Find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df["val_score"].idxmax()
    best_params = results_df.loc[best_idx]

    print("\nBest parameters:")
    print(f"  epsilon: {best_params['epsilon']}")
    print(f"  gamma: {best_params['gamma']}")
    print(f"  Validation C-index: {best_params['val_score']:.4f}")

    # Refit on full training data with best parameters
    print("\nRefitting with best parameters on full training set...")
    best_model = DRLCoxEstimator(epsilon=best_params["epsilon"], gamma=int(best_params["gamma"]))
    best_model.fit(X_train, y_train, zeta_train)
    final_score = best_model.score(X_val, y_val, zeta_val)

    print(f"Final validation C-index: {final_score:.4f}")

    print()


def example_5_model_comparison():
    """Example 5: Compare different epsilon values."""
    print("=" * 80)
    print("EXAMPLE 5: Model Comparison")
    print("=" * 80)

    # Generate data
    data = simulate_cox_data(n=200, d=10, seed=42)
    X, y, zeta = data.X, data.y, data.zeta

    # Test different epsilon values
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.5]

    print(f"\nComparing {len(epsilons)} epsilon values...")
    print()

    results = []
    for eps in epsilons:
        model = DRLCoxEstimator(epsilon=eps)
        model.fit(X, y, zeta)

        score = model.score(X, y, zeta)
        beta_norm = np.linalg.norm(model.beta_)

        results.append({"epsilon": eps, "c_index": score, "beta_norm": beta_norm})

        print(f"ε = {eps:.2f}:")
        print(f"  C-index: {score:.4f}")
        print(f"  ||β||:   {beta_norm:.4f}")
        print()

    # Visualize
    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # C-index vs epsilon
    axes[0].plot(
        results_df["epsilon"], results_df["c_index"], marker="o", linewidth=2, markersize=8
    )
    axes[0].set_xlabel("Epsilon (ε)")
    axes[0].set_ylabel("C-index")
    axes[0].set_title("Performance vs Robustness Level")
    axes[0].grid(alpha=0.3)

    # Beta norm vs epsilon
    axes[1].plot(
        results_df["epsilon"],
        results_df["beta_norm"],
        marker="s",
        linewidth=2,
        markersize=8,
        color="orange",
    )
    axes[1].set_xlabel("Epsilon (ε)")
    axes[1].set_ylabel("||β|| (L2 Norm)")
    axes[1].set_title("Coefficient Regularization Effect")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sklearn_api_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Plot saved to 'sklearn_api_comparison.png'")

    print()


def example_6_custom_scorer():
    """Example 6: Using custom scorers."""
    print("=" * 80)
    print("EXAMPLE 6: Custom Scorers")
    print("=" * 80)

    # Generate data
    data = simulate_cox_data(n=150, d=8, seed=42)
    X, y, zeta = data.X, data.y, data.zeta

    # Fit model
    print("\nFitting model...")
    model = DRLCoxEstimator(epsilon=0.1)
    model.fit(X, y, zeta)

    # Create different scorers
    cindex_scorer = make_drl_cox_scorer(metric="cindex")
    iauc_scorer = make_drl_cox_scorer(metric="iauc")

    # Evaluate with different metrics
    print("\nEvaluating with different metrics:")

    cindex_score = cindex_scorer(model, X, y, zeta)
    print(f"  C-index: {cindex_score:.4f}")

    iauc_score = iauc_scorer(model, X, y, zeta)
    print(f"  iAUC:    {iauc_score:.4f}")

    # Compare with built-in score method
    builtin_score = model.score(X, y, zeta)
    print(f"  Built-in score: {builtin_score:.4f}")
    print("  (Built-in uses C-index by default)")

    print()


def example_7_error_handling():
    """Example 7: Error handling and validation."""
    print("=" * 80)
    print("EXAMPLE 7: Error Handling and Validation")
    print("=" * 80)

    # Generate data
    data = simulate_cox_data(n=100, d=5, seed=42)
    X, y, zeta = data.X, data.y, data.zeta

    print("\nDemonstrating error handling...")

    # 1. Invalid parameters
    print("\n1. Invalid parameters:")
    try:
        model = DRLCoxEstimator(epsilon=-0.1)
        model.fit(X, y, zeta)
    except ValueError as e:
        print(f"   ✓ Caught error: {e}")

    # 2. Shape mismatch
    print("\n2. Shape mismatch:")
    try:
        model = DRLCoxEstimator()
        model.fit(X, y[:-10], zeta)  # Wrong shape
    except ValueError as e:
        print(f"   ✓ Caught error: {e}")

    # 3. Negative survival times
    print("\n3. Negative survival times:")
    try:
        y_bad = y.copy()
        y_bad[0] = -1.0
        model = DRLCoxEstimator()
        model.fit(X, y_bad, zeta)
    except ValueError as e:
        print(f"   ✓ Caught error: {e}")

    # 4. Wrong number of features in predict
    print("\n4. Wrong number of features in predict:")
    try:
        model = DRLCoxEstimator()
        model.fit(X, y, zeta)
        model.predict(X[:, :-1])  # Missing feature
    except ValueError as e:
        print(f"   ✓ Caught error: {e}")

    # 5. Predict before fit
    print("\n5. Predict before fit:")
    try:
        model = DRLCoxEstimator()
        model.predict(X)  # Not fitted
    except Exception as e:
        print(f"   ✓ Caught error: {type(e).__name__}")

    print("\n✓ All error cases handled properly")

    print()


def example_8_get_set_params():
    """Example 8: Getting and setting parameters."""
    print("=" * 80)
    print("EXAMPLE 8: Parameter Management")
    print("=" * 80)

    # Create model
    print("\nCreating model with default parameters...")
    model = DRLCoxEstimator()

    # Get parameters
    params = model.get_params()
    print("\nDefault parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Set parameters
    print("\nUpdating parameters...")
    model.set_params(epsilon=0.2, gamma=4)

    updated_params = model.get_params()
    print("\nUpdated parameters:")
    for key in ["epsilon", "gamma"]:
        print(f"  {key}: {updated_params[key]}")

    # Use in pipeline
    print("\nUsing with pipeline...")
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", DRLCoxEstimator())])

    # Get pipeline parameters
    pipeline_params = pipeline.get_params()
    print("\nPipeline parameters (subset):")
    for key in ["model__epsilon", "model__gamma", "model__solver"]:
        if key in pipeline_params:
            print(f"  {key}: {pipeline_params[key]}")

    # Set pipeline parameters
    pipeline.set_params(model__epsilon=0.15, model__gamma=5)
    print("\nAfter updating through pipeline:")
    print(f"  model__epsilon: {pipeline.named_steps['model'].epsilon}")
    print(f"  model__gamma: {pipeline.named_steps['model'].gamma}")

    print()


def example_9_reproducibility():
    """Example 9: Reproducibility demonstration."""
    print("=" * 80)
    print("EXAMPLE 9: Reproducibility")
    print("=" * 80)

    # Generate data
    data = simulate_cox_data(n=100, d=5, seed=42)
    X, y, zeta = data.X, data.y, data.zeta

    print("\nFitting same model twice...")

    # Fit model twice with same parameters
    model1 = DRLCoxEstimator(epsilon=0.1, solver_opts={"max_iter": 200})
    model1.fit(X, y, zeta)
    beta1 = model1.beta_.copy()

    model2 = DRLCoxEstimator(epsilon=0.1, solver_opts={"max_iter": 200})
    model2.fit(X, y, zeta)
    beta2 = model2.beta_.copy()

    # Check reproducibility
    diff = np.linalg.norm(beta1 - beta2)
    print(f"\nCoefficient difference: {diff:.10f}")

    if diff < 1e-6:
        print("✓ Results are reproducible (identical coefficients)")
    else:
        print("⚠ Results differ slightly (may be due to solver randomness)")

    # Compare predictions
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    pred_diff = np.linalg.norm(pred1 - pred2)

    print(f"Prediction difference: {pred_diff:.10f}")

    print()


def run_all_examples():
    """Run all examples."""
    examples = [
        example_1_basic_usage,
        example_2_train_test_split,
        example_3_pipeline,
        example_4_parameter_tuning,
        example_5_model_comparison,
        example_6_custom_scorer,
        example_7_error_handling,
        example_8_get_set_params,
        example_9_reproducibility,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Example {i} failed with error: {e}\n")
            import traceback

            traceback.print_exc()

    print("=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_usage,
            example_2_train_test_split,
            example_3_pipeline,
            example_4_parameter_tuning,
            example_5_model_comparison,
            example_6_custom_scorer,
            example_7_error_handling,
            example_8_get_set_params,
            example_9_reproducibility,
        ]

        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        print("Running all examples...")
        print()
        run_all_examples()
