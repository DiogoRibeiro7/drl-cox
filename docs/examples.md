# Examples

## Basic Usage

```python
from drl_cox import simulate_cox_data, fit_drl_cox, concordance_index

# Generate synthetic data
data = simulate_cox_data(n=200, d=10, seed=42)

# Fit DRL-Cox model
result = fit_drl_cox(data, epsilon=0.1, gamma=3)

# Compute predictions
risk_scores = data.X @ result.beta

# Evaluate
c_index = concordance_index(risk_scores, data.y, data.zeta)
print(f"C-index: {c_index:.3f}")
```

## Cross-Validation

```python
from drl_cox import cross_validate_epsilon

# Find optimal epsilon
cv_results = cross_validate_epsilon(
    data, epsilons=[0.0, 0.05, 0.1, 0.2, 0.5], kfolds=5, metric="cindex"
)

# Best epsilon
best_eps = cv_results.groupby("epsilon")["score"].mean().idxmax()
print(f"Best epsilon: {best_eps}")
```

## With Contamination

```python
from drl_cox import inject_covariate_shift, inject_outliers

# Add contamination
X_shifted = inject_covariate_shift(data.X, feature_indices=[0, 1], mean=2.0, std=1.5)
X_noisy = inject_outliers(X_shifted, ratio=0.15, severity_std=3.0)

contaminated = SurvivalDataset(X=X_noisy, y=data.y, zeta=data.zeta)

# Robust fitting
result = fit_drl_cox(contaminated, epsilon=0.2)
```

## Comparison with Baselines

```python
from drl_cox import CoxPartialLikelihood, CoxRidge, CoxLasso, concordance_index

# Fit baselines
cox_pl = CoxPartialLikelihood().fit(data.X, data.y, data.zeta)
cox_ridge = CoxRidge(alpha=1.0).fit(data.X, data.y, data.zeta)
cox_lasso = CoxLasso(alpha=0.05).fit(data.X, data.y, data.zeta)

# Compare
for name, beta in [("PL", cox_pl), ("Ridge", cox_ridge), ("Lasso", cox_lasso)]:
    risk = data.X @ beta
    c = concordance_index(risk, data.y, data.zeta)
    print(f"{name}: {c:.3f}")
```
