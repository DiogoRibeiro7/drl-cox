"""
DRL-Cox: Distributionally Robust Cox Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Python package for distributionally robust survival analysis using
Wasserstein ambiguity sets. Implements the DRL-Cox model for robust
Cox proportional hazards regression with contamination-resistant estimation.

Basic usage:
    >>> from drl_cox import simulate_cox_data, fit_drl_cox
    >>> data = simulate_cox_data(n=200, d=10, seed=42)
    >>> result = fit_drl_cox(data, epsilon=0.1, gamma=3)
    >>> print(f"Status: {result.status}")

Full documentation: https://github.com/diogoribeiro7/drl-cox
Paper: See paper/Distributionally Robust Learning in Survival Analysis.pdf
"""

# Version information
__version__ = "0.1.0"
__author__ = "Diogo Ribeiro"
__author_email__ = "dfr@esmad.ipp.pt"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Diogo Ribeiro"

# Package metadata
__title__ = "drl-cox"
__description__ = (
    "Distributionally Robust Cox Regression with Wasserstein ambiguity, "
    "plus classical Cox baselines and evaluation utilities."
)
__url__ = "https://github.com/diogoribeiro7/drl-cox"
__keywords__ = [
    "survival-analysis",
    "cox-regression",
    "distributionally-robust-optimization",
    "wasserstein-distance",
    "robust-statistics",
    "machine-learning",
    "biostatistics",
]

# Core DRL-Cox functionality
# Contamination utilities
from .contamination import (
    inject_covariate_shift,
    inject_outliers,
)

# Baseline Cox models
from .cox_baseline import (
    CoxLasso,
    CoxPartialLikelihood,
    CoxRidge,
)

# Data utilities
from .datasets import (
    load_whas500_like_csv,
    simulate_cox_data,
)
from .drl_cox import (
    DRLCoxResult,
    SurvivalDataset,
    cross_validate_epsilon,
    fit_drl_cox,
    kfold_indices,
    risk_linear_predictor,
)
from .cross_validation import (
    AutoSelectionResult,
    auto_select_epsilon,
)

# Scikit-learn compatible estimator
from .estimator import (
    DRLCoxEstimator,
    make_drl_cox_scorer,
)

# Survival metrics
from .metrics import (
    concordance_index,
    time_dependent_auc_iAUC,
)

# Optimized parallel cross-validation (if available)
try:
    from .parallel_cv import (
        benchmark_parallel_cv,
    )
    from .parallel_cv import (
        cross_validate_epsilon as cross_validate_epsilon_parallel,
    )

    _HAS_PARALLEL_CV = True
except ImportError:
    _HAS_PARALLEL_CV = False

# Define public API
__all__ = [
    # Core classes and functions
    "SurvivalDataset",
    "DRLCoxResult",
    "fit_drl_cox",
    "cross_validate_epsilon",
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
    # Contamination
    "inject_covariate_shift",
    "inject_outliers",
    # Version info
    "__version__",
    "__author__",
]

# Add parallel CV if available
if _HAS_PARALLEL_CV:
    __all__.extend(
        [
            "cross_validate_epsilon_parallel",
            "benchmark_parallel_cv",
        ]
    )


# Optional: Package-level initialization
def _check_dependencies():
    """Check that required dependencies are available."""
    import importlib
    import warnings

    required = {
        "numpy": "^2.0.0",
        "pandas": "^2.2.0",
        "cvxpy": "^1.5.2",
        "scipy": "^1.13.0",
    }

    optional = {
        "joblib": "parallel cross-validation",
        "tqdm": "progress bars",
        "matplotlib": "visualization",
        "seaborn": "enhanced plots",
        "skopt": "Bayesian epsilon selection",
    }

    # Check required dependencies
    for module, version in required.items():
        try:
            importlib.import_module(module)
        except ImportError as e:
            raise ImportError(
                f"Required dependency '{module}' not found. Install with: pip install {module}"
            ) from e

    # Check optional dependencies
    missing_optional = []
    for module, feature in optional.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_optional.append((module, feature))

    if missing_optional:
        features = ", ".join(f"{m} ({f})" for m, f in missing_optional)
        warnings.warn(
            f"Optional dependencies not installed: {features}. Some features may be unavailable.",
            UserWarning,
            stacklevel=2,
        )


# Optional: Deprecation handling
def __getattr__(name):
    """Handle deprecated attributes and provide helpful migration messages."""
    deprecated = {
        # Map old names to new names if any API changes
        # Example: "old_function": "new_function"
    }

    if name in deprecated:
        import warnings

        warnings.warn(
            f"'{name}' is deprecated. Use '{deprecated[name]}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals().get(deprecated[name])

    # Provide helpful error for common mistakes
    if name == "DRLCox":
        raise AttributeError(
            f"'{name}' not found. Did you mean 'DRLCoxEstimator' or 'fit_drl_cox'?"
        )

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Run initialization checks
try:
    _check_dependencies()
except ImportError:
    pass  # Let the error propagate naturally

# Clean up namespace
del _check_dependencies
del _HAS_PARALLEL_CV


# Convenience function for quick start
def get_demo_data(n: int = 200, d: int = 10, seed: int = 42) -> SurvivalDataset:
    """
    Get demo survival data for quick testing.

    Parameters
    ----------
    n : int, default=200
        Number of samples
    d : int, default=10
        Number of features
    seed : int, default=42
        Random seed

    Returns
    -------
    SurvivalDataset
        Synthetic survival data

    Examples
    --------
    >>> from drl_cox import get_demo_data, DRLCoxEstimator
    >>> data = get_demo_data()
    >>> model = DRLCoxEstimator(epsilon=0.1)
    >>> model.fit(data.X, data.y, data.zeta)
    """
    return simulate_cox_data(n=n, d=d, seed=seed)


# Export convenience function
__all__.append("get_demo_data")
