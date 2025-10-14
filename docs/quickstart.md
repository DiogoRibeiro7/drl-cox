# Quick Start Guide

## Basic Example

```python
from drl_cox import simulate_cox_data, fit_drl_cox, concordance_index

# Generate synthetic survival data
data = simulate_cox_data(n=200, d=10, seed=42)

# Fit DRL-Cox model
result = fit_drl_cox(
    data,
    epsilon=0.1,      # Wasserstein radius
    p=2.0,            # Norm parameter
    gamma=3,          # Risk set constraints
    solver="ECOS"
)

# Compute risk scores
risk_scores = data.X @ result.beta

# Evaluate performance
c_index = concordance_index(risk_scores, data.y, data.zeta)
print(f"C-index: {c_index:.3f}")
```

## With Real Data

```python
from drl_cox import load_whas500_like_csv, fit_drl_cox, cross_validate_epsilon

# Load your data
data = load_whas500_like_csv("my_data.csv")

# Cross-validate epsilon
cv_results = cross_validate_epsilon(
    data,
    epsilons=[0.0, 0.05, 0.1, 0.2],
    kfolds=5,
    metric="cindex"
)

# Select best epsilon
best_eps = cv_results.groupby("epsilon")["score"].mean().idxmax()

# Fit final model
result = fit_drl_cox(data, epsilon=best_eps)
```

## Next Steps

- Read the [API Reference](api.md) for detailed documentation
- Explore more [Examples](examples.md)
- Check the [Tutorial](tutorial.md) for in-depth explanations
