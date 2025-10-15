# Scikit-learn Compatible API

The `DRLCoxEstimator` class provides a scikit-learn compatible interface for DRL-Cox, enabling seamless integration with the scikit-learn ecosystem.

## Table of Contents

- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Pipeline Integration](#pipeline-integration)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Custom Scorers](#custom-scorers)
- [Examples](#examples)
- [Limitations](#limitations)

## Quick Start

```python
from drl_cox import DRLCoxEstimator, simulate_cox_data

# Generate data
data = simulate_cox_data(n=200, d=10, seed=42)
X, y, zeta = data.X, data.y, data.zeta

# Create and fit model
model = DRLCoxEstimator(epsilon=0.1)
model.fit(X, y, zeta)

# Make predictions (risk scores)
risk_scores = model.predict(X)

# Evaluate
score = model.score(X, y, zeta)
print(f"C-index: {score:.3f}")
```

## API Reference

### DRLCoxEstimator

Distributionally Robust Cox Proportional Hazards Model (scikit-learn compatible).

#### Parameters

- **epsilon** : `float`, default=0.1
  - Wasserstein radius (robustness parameter)
  - `epsilon = 0.0`: No robustness (standard Cox)
  - `epsilon > 0.0`: Increasing robustness
  - Must be >= 0.0

- **p** : `float`, default=2.0
  - Norm parameter for Wasserstein distance
  - Must be >= 1.0
  - Common: 1.0 (total variation), 2.0 (Euclidean)

- **gamma** : `int`, default=3
  - Number of risk set constraints per observation
  - Higher values = stronger guarantees, more computation
  - Must be >= 1

- **solver** : `str`, default="ECOS"
  - CVXPY solver: "ECOS", "SCS", "MOSEK", "CLARABEL"

- **solver_opts** : `dict` or `None`, default=None
  - Solver-specific options
  - Example: `{"max_iters": 500, "abstol": 1e-8}`

#### Attributes (After Fitting)

- **beta_** : `ndarray` of shape (n_features,)
  - Fitted coefficient vector

- **alpha_** : `float`
  - Fitted time-scale parameter

- **s_** : `ndarray` of shape (n_samples,)
  - Fitted slack variables

- **objective_value_** : `float`
  - Final objective function value

- **status_** : `str`
  - Solver status ("optimal", etc.)

- **n_features_in_** : `int`
  - Number of features seen during fit

#### Methods

##### fit(X, y, zeta=None)

Fit the DRL-Cox model.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
- `y` : array-like of shape (n_samples,) - Survival times (positive)
- `zeta` : array-like of shape (n_samples,) - Event indicators (0/1)

**Returns:**
- `self` : Fitted estimator

##### predict(X)

Predict risk scores for samples in X.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)

**Returns:**
- `risk_scores` : ndarray of shape (n_samples,)

##### score(X, y, zeta=None)

Return the concordance index (C-index) on test data.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
- `y` : array-like of shape (n_samples,)
- `zeta` : array-like of shape (n_samples,)

**Returns:**
- `score` : float (0.5 to 1.0, higher is better)

##### get_params(deep=True)

Get parameters for this estimator.

##### set_params(**params)

Set parameters for this estimator.

## Pipeline Integration

DRLCoxEstimator works seamlessly with scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from drl_cox import DRLCoxEstimator

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('drl_cox', DRLCoxEstimator(epsilon=0.1))
])

# Fit pipeline (pass zeta as fit parameter)
pipeline.fit(X, y, drl_cox__zeta=zeta)

# Predict
risk_scores = pipeline.predict(X)

# Score
c_index = pipeline.score(X, y, drl_cox__zeta=zeta)
```

### Getting and Setting Pipeline Parameters

```python
# Get parameters
params = pipeline.get_params()
print(params['drl_cox__epsilon'])  # Access nested parameter

# Set parameters
pipeline.set_params(drl_cox__epsilon=0.2, drl_cox__gamma=4)
```

## Hyperparameter Tuning

### Manual Grid Search (Recommended)

Due to the three-argument fit signature (X, y, zeta), manual grid search is recommended:

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val, zeta_train, zeta_val = train_test_split(
    X, y, zeta, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'epsilon': [0.0, 0.05, 0.1, 0.2],
    'gamma': [2, 3, 4]
}

# Manual grid search
results = []
for epsilon in param_grid['epsilon']:
    for gamma in param_grid['gamma']:
        model = DRLCoxEstimator(epsilon=epsilon, gamma=gamma)
        model.fit(X_train, y_train, zeta_train)
        score = model.score(X_val, y_val, zeta_val)
        results.append({
            'epsilon': epsilon,
            'gamma': gamma,
            'score': score
        })

# Find best
import pandas as pd
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['score'].idxmax()]
print(f"Best params: epsilon={best['epsilon']}, gamma={best['gamma']}")
```

### Using GridSearchCV (Limited Support)

Standard GridSearchCV has limitations with survival data:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'epsilon': [0.0, 0.1, 0.2]}

# Note: This requires custom CV splitter for proper zeta handling
grid_search = GridSearchCV(
    DRLCoxEstimator(),
    param_grid,
    cv=3,
    scoring=None  # Uses estimator's score method
)

# Limitations: zeta must be passed differently
# Recommended: Use manual approach above
```

## Custom Scorers

Create custom scorers for different metrics:

```python
from drl_cox import make_drl_cox_scorer

# C-index scorer
cindex_scorer = make_drl_cox_scorer(metric="cindex")

# iAUC scorer
iauc_scorer = make_drl_cox_scorer(metric="iauc")

# Use scorer
model = DRLCoxEstimator(epsilon=0.1)
model.fit(X_train, y_train, zeta_train)

cindex = cindex_scorer(model, X_test, y_test, zeta_test)
iauc = iauc_scorer(model, X_test, y_test, zeta_test)
```

## Examples

### Example 1: Basic Usage

```python
from drl_cox import DRLCoxEstimator, simulate_cox_data

# Generate data
data = simulate_cox_data(n=200, d=10, seed=42)
X, y, zeta = data.X, data.y, data.zeta

# Fit model
model = DRLCoxEstimator(epsilon=0.1)
model.fit(X, y, zeta)

# Predict and evaluate
risk_scores = model.predict(X)
c_index = model.score(X, y, zeta)
print(f"C-index: {c_index:.3f}")
```

### Example 2: Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test, zeta_train, zeta_test = train_test_split(
    X, y, zeta, test_size=0.2, random_state=42
)

# Fit on train
model = DRLCoxEstimator(epsilon=0.1)
model.fit(X_train, y_train, zeta_train)

# Evaluate on test
test_score = model.score(X_test, y_test, zeta_test)
print(f"Test C-index: {test_score:.3f}")
```

### Example 3: Pipeline with Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('drl_cox', DRLCoxEstimator(epsilon=0.1))
])

pipeline.fit(X_train, y_train, drl_cox__zeta=zeta_train)
score = pipeline.score(X_test, y_test, drl_cox__zeta=zeta_test)
```

### Example 4: Comparing Different Epsilon Values

```python
epsilons = [0.0, 0.05, 0.1, 0.2, 0.5]

for eps in epsilons:
    model = DRLCoxEstimator(epsilon=eps)
    model.fit(X_train, y_train, zeta_train)
    score = model.score(X_test, y_test, zeta_test)
    print(f"ε={eps:.2f}: C-index={score:.3f}")
```

### Example 5: Accessing Fitted Coefficients

```python
model = DRLCoxEstimator(epsilon=0.1)
model.fit(X, y, zeta)

# Get coefficients
coefficients = model.beta_
print(f"Coefficients: {coefficients}")

# Get feature importance (absolute values)
importance = np.abs(coefficients)
top_features = np.argsort(importance)[::-1][:5]
print(f"Top 5 features: {top_features}")
```

## Limitations

### 1. Three-Argument Fit Signature

Unlike standard scikit-learn estimators, DRL-Cox requires three arrays:
- `X`: Features
- `y`: Survival times
- `zeta`: Event indicators

This creates challenges with some scikit-learn tools like `GridSearchCV`.

**Workaround**: Use manual grid search or pass zeta as fit parameter in pipelines.

### 2. Custom Scoring Required

Standard scikit-learn scoring functions don't support survival data.

**Workaround**: Use `make_drl_cox_scorer` or the estimator's built-in `score` method.

### 3. Cross-Validation

Standard cross-validation tools require adaptation for the three-argument fit.

**Workaround**: Implement custom CV loops or use the `cross_validate_epsilon` function.

### 4. Sample Weights Not Supported

The `sample_weight` parameter is not currently supported.

### Example Workaround for GridSearchCV

```python
from sklearn.model_selection import KFold
from sklearn.base import clone

def custom_grid_search_cv(estimator, param_grid, X, y, zeta, cv=5):
    """Custom grid search for survival data."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    
    for params in ParameterGrid(param_grid):
        scores = []
        for train_idx, val_idx in kfold.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            zeta_train, zeta_val = zeta[train_idx], zeta[val_idx]
            
            # Fit and score
            model = clone(estimator).set_params(**params)
            model.fit(X_train, y_train, zeta_train)
            score = model.score(X_val, y_val, zeta_val)
            scores.append(score)
        
        results.append({
            **params,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        })
    
    return pd.DataFrame(results)

# Usage
param_grid = {'epsilon': [0.0, 0.1, 0.2]}
results = custom_grid_search_cv(
    DRLCoxEstimator(),
    param_grid,
    X, y, zeta,
    cv=5
)
```

## Best Practices

### 1. Always Provide zeta

```python
# Good
model.fit(X, y, zeta)

# Avoid (assumes all events observed)
model.fit(X, y)
```

### 2. Standardize Features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DRLCoxEstimator(epsilon=0.1)
model.fit(X_scaled, y, zeta)
```

### 3. Tune Epsilon via Cross-Validation

```python
from drl_cox import cross_validate_epsilon

cv_results = cross_validate_epsilon(
    SurvivalDataset(X, y, zeta),
    epsilons=[0.0, 0.05, 0.1, 0.2],
    kfolds=5
)

best_eps = cv_results.groupby('epsilon')['score'].mean().idxmax()
```

### 4. Check Solver Status

```python
model.fit(X, y, zeta)

if model.status_ not in ["optimal", "optimal_inaccurate"]:
    print(f"Warning: Solver status is '{model.status_}'")
    # Consider adjusting parameters
```

### 5. Validate Input Data

```python
# Check for issues before fitting
assert np.all(y > 0), "Survival times must be positive"
assert np.all(np.isin(zeta, [0, 1])), "zeta must be 0 or 1"
assert not np.any(np.isnan(X)), "X contains NaN values"
```

## Troubleshooting

### Issue: ValueError - "X and y have inconsistent shapes"

**Solution**: Ensure X, y, and zeta have matching first dimensions.

```python
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"zeta shape: {zeta.shape}")
```

### Issue: Solver fails to converge

**Solutions:**
1. Increase max iterations: `solver_opts={"max_iters": 500}`
2. Try different solver: `solver="SCS"`
3. Adjust epsilon or gamma
4. Standardize features

### Issue: GridSearchCV doesn't work

**Solution**: Use manual grid search as shown in examples above.

### Issue: Pipeline throws error with zeta

**Solution**: Pass zeta as fit parameter:

```python
pipeline.fit(X, y, estimator_name__zeta=zeta)
```

## Running Examples

Run the comprehensive examples script:

```bash
# Run all examples
python examples/sklearn_api_demo.py

# Run specific example
python examples/sklearn_api_demo.py 1  # Basic usage
python examples/sklearn_api_demo.py 3  # Pipeline
python examples/sklearn_api_demo.py 4  # Hyperparameter tuning
```

## References

- [scikit-learn Estimator API](https://scikit-learn.org/stable/developers/develop.html)
- [DRL-Cox Paper](../paper/)
- [API Reference](api.md)
