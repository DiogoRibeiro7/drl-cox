# API Reference

## Core Classes

### `SurvivalDataset`

```python
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class SurvivalDataset:
    X: np.ndarray  # Covariates (N, d)
    y: np.ndarray  # Survival times (N,)
    zeta: np.ndarray  # Event indicators (N,)
```

### `DRLCoxResult`

```python
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass
class DRLCoxResult:
    beta: np.ndarray  # Coefficient vector
    alpha: float  # Time-scale parameter
    s: np.ndarray  # Slack variables
    objective_value: float  # Final objective
    status: str  # Solver status
    info: Dict[str, Any]  # Additional info
```

## Main Functions

### `fit_drl_cox`

```python
def fit_drl_cox(
    data: SurvivalDataset,
    epsilon: float,
    p: float = 2.0,
    gamma: int = 3,
    solver: str = "CLARABEL",
    solver_opts: dict[str, Any] | None = None,
    formulation: Literal["auto", "full", "events"] = "auto",
) -> DRLCoxResult:
    """Fit the Wasserstein distributionally robust Cox model."""
```

**Parameters:**

* `data`: Training survival data
* `epsilon`: Wasserstein radius (robustness level)
* `p`: Norm parameter (default `2.0`)
* `gamma`: Number of risk set constraints per observation
* `solver`: CVXPY solver name
* `solver_opts`: Solver-specific options
* `formulation`: `"full"` builds slack constraints for every row (robust with Clarabel);
  `"events"` only for event rows (same optimum, about half the size, faster with ECOS
  and SCS); `"auto"` chooses by solver

**Returns:**

* `DRLCoxResult` with fitted parameters

---

### `cross_validate_epsilon`

```python
def cross_validate_epsilon(
    data: SurvivalDataset,
    epsilons: Iterable[float],
    p: float = 2.0,
    gamma: int = 3,
    kfolds: int = 5,
    metric: Literal["cindex", "iauc"] = "cindex",
    solver: str = "CLARABEL",
    solver_opts: Optional[Dict[str, Any]] = None,
    iauc_average: Literal["uniform", "event"] = "event",
) -> pd.DataFrame:
    """Cross-validate to select optimal epsilon."""
```

### `concordance_index`

```python
def concordance_index(risk_scores: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> float:
    """Compute Harrell's C-index for survival predictions."""
```

### `time_dependent_auc_iAUC`

```python
def time_dependent_auc_iAUC(
    risk_scores: np.ndarray,
    y: np.ndarray,
    zeta: np.ndarray,
    times: np.ndarray | None = None,
    average: Literal["uniform", "event"] = "event",
) -> float:
    """Compute time-dependent integrated AUC via IPCW."""
```

## Baseline Models

* **CoxPartialLikelihood** — Standard Cox proportional hazards via Newton–Raphson.
* **CoxRidge** — Cox model with L2 regularization.
* **CoxLasso** — Cox model with L1 regularization via coordinate descent.

## Utilities

* **simulate_cox_data** — Generate synthetic survival data for testing.
* **inject_covariate_shift** — Add distributional shift to selected features.
* **inject_outliers** — Add Gaussian noise to random observations.
* **load_whas500_like_csv** — Load survival data from CSV file.
