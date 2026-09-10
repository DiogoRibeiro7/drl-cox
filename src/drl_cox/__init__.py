"""
DRL-Cox: Distributionally Robust Cox Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Python package for distributionally robust survival analysis using
Wasserstein ambiguity sets. Implements the DRL-Cox model of Jin, Wise and
Paschalidis (CHIL 2025) for robust Cox proportional hazards regression,
together with classical Cox baselines and censoring-aware evaluation metrics.

Basic usage:
    >>> from drl_cox import simulate_cox_data, fit_drl_cox
    >>> data = simulate_cox_data(n=200, d=10, seed=42)
    >>> result = fit_drl_cox(data, epsilon=0.1, gamma=3)
    >>> print(f"Status: {result.status}")

Documentation: https://diogoribeiro7.github.io/drl-cox/
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any

from .contamination import inject_covariate_shift, inject_outliers
from .cox_baseline import CoxLasso, CoxPartialLikelihood, CoxRidge
from .cross_validation import AutoSelectionResult, auto_select_epsilon
from .datasets import load_whas500_like_csv, simulate_cox_data
from .drl_cox import (
    DEFAULT_SOLVER,
    DRLCoxResult,
    SurvivalDataset,
    cross_validate_epsilon,
    fit_drl_cox,
    kfold_indices,
    risk_linear_predictor,
)
from .estimator import DRLCoxEstimator, make_drl_cox_scorer
from .metrics import concordance_index, time_dependent_auc_iAUC
from .parallel_cv import benchmark_parallel_cv
from .parallel_cv import cross_validate_epsilon as cross_validate_epsilon_parallel

try:
    __version__ = _package_version("drl-cox")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__author__ = "Diogo Ribeiro"
__author_email__ = "dfr@esmad.ipp.pt"
__license__ = "MIT"

__all__ = [
    # Core classes and functions
    "DEFAULT_SOLVER",
    "SurvivalDataset",
    "DRLCoxResult",
    "fit_drl_cox",
    "cross_validate_epsilon",
    "cross_validate_epsilon_parallel",
    "benchmark_parallel_cv",
    "auto_select_epsilon",
    "AutoSelectionResult",
    "risk_linear_predictor",
    "kfold_indices",
    # Scikit-learn API
    "DRLCoxEstimator",
    "make_drl_cox_scorer",
    # Baseline models
    "CoxPartialLikelihood",
    "CoxRidge",
    "CoxLasso",
    # Metrics
    "concordance_index",
    "time_dependent_auc_iAUC",
    # Data utilities
    "load_whas500_like_csv",
    "simulate_cox_data",
    "get_demo_data",
    # Contamination
    "inject_covariate_shift",
    "inject_outliers",
    # Metadata
    "__version__",
    "__author__",
]


def get_demo_data(n: int = 200, d: int = 10, seed: int = 42) -> SurvivalDataset:
    """Return a small synthetic survival dataset for quick experiments.

    Parameters
    ----------
    n : int, default=200
        Number of samples.
    d : int, default=10
        Number of features.
    seed : int, default=42
        Random seed.

    Examples
    --------
    >>> from drl_cox import get_demo_data, DRLCoxEstimator
    >>> data = get_demo_data()
    >>> model = DRLCoxEstimator(epsilon=0.1).fit(data.X, data.y, data.zeta)
    """
    return simulate_cox_data(n=n, d=d, seed=seed)


def __getattr__(name: str) -> Any:
    """Give a helpful hint for the most common misspelling of the estimator name."""
    if name == "DRLCox":
        raise AttributeError(
            f"'{name}' not found. Did you mean 'DRLCoxEstimator' or 'fit_drl_cox'?"
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
