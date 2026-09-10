"""
Scikit-learn compatible estimator for DRL-Cox.

This module provides a scikit-learn compatible API for the DRL-Cox model,
enabling seamless integration with scikit-learn pipelines and tools.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data

from .drl_cox import (
    DEFAULT_SOLVER,
    SurvivalDataset,
    fit_drl_cox,
    risk_linear_predictor,
)
from .metrics import concordance_index, time_dependent_auc_iAUC


class DRLCoxEstimator(BaseEstimator):
    """
    Distributionally Robust Cox Proportional Hazards Model (scikit-learn compatible).

    This estimator implements the Wasserstein distributionally robust Cox model,
    providing robustness against data contamination and distributional uncertainty.

    The model solves the following optimization problem:

        min_{β,α,s} ε||[β,α]||_q + (1/n)Σ_i ζ_i s_i

    subject to risk set constraints that ensure the robustness guarantees.

    Parameters
    ----------
    epsilon : float, default=0.1
        Wasserstein radius (robustness parameter). Larger values provide more
        robustness at the cost of potential underfitting.
        - epsilon = 0.0: No robustness (standard Cox model)
        - epsilon > 0.0: Increasing robustness
        Must be >= 0.0.

    p : float, default=2.0
        Norm parameter for Wasserstein distance. Must be >= 1.0.
        Common choices: 1.0 (total variation), 2.0 (Euclidean).

    gamma : int, default=3
        Number of risk set constraints per observation. Higher values provide
        stronger guarantees but increase computational cost.
        Must be >= 1.

    solver : str, default="CLARABEL"
        CVXPY solver to use. Any solver supporting the exponential cone works:
        - "CLARABEL": open-source interior-point solver (default, installed with CVXPY)
        - "SCS": first-order solver, scalable for large problems
        - "ECOS": legacy interior-point solver (``pip install drl-cox[ecos]``, Python < 3.13)
        - "MOSEK": commercial solver (if installed)

    solver_opts : dict or None, default=None
        Solver-specific options passed to ``cvxpy.Problem.solve``.
        Example for Clarabel: {"max_iter": 500, "tol_gap_abs": 1e-8}

    Attributes
    ----------
    beta_ : ndarray of shape (n_features,)
        Fitted coefficient vector.

    alpha_ : float
        Fitted time-scale parameter.

    s_ : ndarray of shape (n_samples,)
        Fitted slack variables (one per training observation).

    objective_value_ : float
        Final objective function value after optimization.

    status_ : str
        Solver status ("optimal", "optimal_inaccurate", etc.).

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit (only if X has feature names).

    result_ : DRLCoxResult
        Complete result object from fit_drl_cox.

    Examples
    --------
    >>> from drl_cox import DRLCoxEstimator, simulate_cox_data
    >>>
    >>> # Generate synthetic data
    >>> data = simulate_cox_data(n=200, d=10, seed=42)
    >>> X, y, zeta = data.X, data.y, data.zeta
    >>>
    >>> # Fit DRL-Cox model
    >>> model = DRLCoxEstimator(epsilon=0.1)
    >>> model.fit(X, y, zeta)
    >>>
    >>> # Make predictions (risk scores)
    >>> risk_scores = model.predict(X)
    >>>
    >>> # Evaluate
    >>> from drl_cox import concordance_index
    >>> c_index = concordance_index(risk_scores, y, zeta)
    >>> print(f"C-index: {c_index:.3f}")

    >>> # Use with scikit-learn pipelines
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> pipeline = Pipeline(
    ...     [("scaler", StandardScaler()), ("drl_cox", DRLCoxEstimator(epsilon=0.1))]
    ... )
    >>> pipeline.fit(X, y, drl_cox__zeta=zeta)
    >>> risk_scores = pipeline.predict(X)

    >>> # Hyperparameter tuning with GridSearchCV
    >>> from sklearn.model_selection import GridSearchCV
    >>>
    >>> param_grid = {"epsilon": [0.0, 0.05, 0.1, 0.2], "gamma": [2, 3, 4]}
    >>> grid_search = GridSearchCV(
    ...     DRLCoxEstimator(),
    ...     param_grid,
    ...     cv=5,
    ...     scoring="neg_mean_squared_error",  # Custom scorer recommended
    ... )
    >>> # Note: GridSearchCV requires custom scorer for survival data

    Notes
    -----
    - This estimator requires three arrays for fitting: X, y, and zeta
    - X: Feature matrix (n_samples, n_features)
    - y: Survival times (n_samples,)
    - zeta: Event indicators (n_samples,), 1=event, 0=censored

    - For pipeline integration, pass zeta as a fit parameter:
      pipeline.fit(X, y, estimator_name__zeta=zeta)

    - Standard scikit-learn scoring functions don't support survival data.
      Use custom scorers based on concordance_index or time_dependent_auc_iAUC.

    See Also
    --------
    fit_drl_cox : Low-level function for fitting DRL-Cox
    concordance_index : C-index for survival data
    time_dependent_auc_iAUC : Time-dependent AUC for survival data

    References
    ----------
    .. [1] "Distributionally Robust Learning in Survival Analysis"
           See paper/ directory for full reference.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        p: float = 2.0,
        gamma: int = 3,
        solver: str = DEFAULT_SOLVER,
        solver_opts: dict[str, Any] | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.p = p
        self.gamma = gamma
        self.solver = solver
        self.solver_opts = solver_opts

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        zeta: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> DRLCoxEstimator:
        """
        Fit the DRL-Cox model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.

        y : array-like of shape (n_samples,)
            Survival times. Must be positive.

        zeta : array-like of shape (n_samples,) or None
            Event indicators (1 = event observed, 0 = censored).
            If None, assumes all events are observed.

        sample_weight : array-like of shape (n_samples,) or None, default=None
            Individual weights for each sample. Currently not supported
            (reserved for future implementation).

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid or input arrays have incompatible shapes.
        RuntimeError
            If optimization fails to converge.
        """
        # Validate parameters
        self._validate_params()

        # Convert inputs to numpy arrays
        X = validate_data(self, X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2)
        y = self._check_array(y, name="y", ndim=1, dtype=np.float64)

        # Check sample weight (not supported yet)
        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not supported and will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Validate zeta
        if zeta is None:
            warnings.warn(
                "zeta not provided; assuming all events are observed (zeta=1). "
                "For censored data, pass zeta explicitly.",
                UserWarning,
                stacklevel=2,
            )
            zeta = np.ones(len(y), dtype=int)
        else:
            zeta = self._check_array(zeta, name="zeta", ndim=1, dtype=int)

        # Check shapes
        n_samples = X.shape[0]
        if len(y) != n_samples:
            raise ValueError(
                f"X and y have inconsistent shapes: "
                f"X has {n_samples} samples, y has {len(y)} samples."
            )
        if len(zeta) != n_samples:
            raise ValueError(
                f"X and zeta have inconsistent shapes: "
                f"X has {n_samples} samples, zeta has {len(zeta)} samples."
            )

        # Validate survival data
        if not np.all(y > 0):
            raise ValueError("All survival times must be positive.")
        if not np.all(np.isin(zeta, [0, 1])):
            raise ValueError("zeta must contain only 0 (censored) or 1 (event).")

        # Store number of features
        self.n_features_in_ = X.shape[1]

        # Create SurvivalDataset
        data = SurvivalDataset(X=X, y=y, zeta=zeta)

        # Fit DRL-Cox model
        try:
            result = fit_drl_cox(
                data,
                epsilon=self.epsilon,
                p=self.p,
                gamma=self.gamma,
                solver=self.solver,
                solver_opts=self.solver_opts or {},
            )
        except Exception as e:
            raise RuntimeError(
                f"DRL-Cox optimization failed: {str(e)}\n"
                "Try adjusting epsilon, gamma, or solver_opts."
            ) from e

        # Check if optimization was successful
        if result.status not in ["optimal", "optimal_inaccurate"]:
            warnings.warn(
                f"Solver returned status '{result.status}'. "
                "Results may be unreliable. Try adjusting parameters.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Store fitted parameters
        self.beta_ = result.beta
        self.alpha_ = result.alpha
        self.s_ = result.s
        self.objective_value_ = result.objective_value
        self.status_ = result.status
        self.result_ = result

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores for samples in X.

        Higher risk scores indicate higher hazard (worse prognosis).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which to predict risk scores.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores (X @ beta_).

        Raises
        ------
        ValueError
            If X has wrong number of features.
        NotFittedError
            If estimator is not fitted.
        """
        # Check if fitted
        check_is_fitted(self, ["beta_", "n_features_in_"])

        # Validate input
        X = validate_data(self, X, dtype=np.float64, ensure_2d=True, reset=False)

        # Compute risk scores
        risk_scores = risk_linear_predictor(X, self.beta_)

        return risk_scores

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        zeta: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """
        Return the concordance index (C-index) on the given test data.

        The C-index measures the model's discriminative ability for survival data.
        Values range from 0.5 (random) to 1.0 (perfect discrimination).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        y : array-like of shape (n_samples,)
            True survival times.

        zeta : array-like of shape (n_samples,) or None
            Event indicators (1 = event, 0 = censored).
            If None, assumes all events are observed.

        sample_weight : array-like of shape (n_samples,) or None, default=None
            Sample weights. Currently not supported.

        Returns
        -------
        score : float
            Concordance index (higher is better).

        Raises
        ------
        ValueError
            If input arrays have incompatible shapes.
        NotFittedError
            If estimator is not fitted.
        """
        # Validate inputs
        X = self._check_array(X, name="X", ndim=2, dtype=np.float64)
        y = self._check_array(y, name="y", ndim=1, dtype=np.float64)

        if zeta is None:
            zeta = np.ones(len(y), dtype=int)
        else:
            zeta = self._check_array(zeta, name="zeta", ndim=1, dtype=int)

        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not supported and will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Predict risk scores
        risk_scores = self.predict(X)

        # Compute C-index
        c_index = concordance_index(risk_scores, y, zeta)

        return c_index

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if not isinstance(self.epsilon, (int, float)):
            raise TypeError(f"epsilon must be a number, got {type(self.epsilon).__name__}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")

        if not isinstance(self.p, (int, float)):
            raise TypeError(f"p must be a number, got {type(self.p).__name__}")
        if self.p < 1:
            raise ValueError(f"p must be >= 1, got {self.p}")

        if not isinstance(self.gamma, int):
            raise TypeError(f"gamma must be an integer, got {type(self.gamma).__name__}")
        if self.gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {self.gamma}")

        if not isinstance(self.solver, str):
            raise TypeError(f"solver must be a string, got {type(self.solver).__name__}")

        if self.solver_opts is not None and not isinstance(self.solver_opts, dict):
            raise TypeError(
                f"solver_opts must be a dict or None, got {type(self.solver_opts).__name__}"
            )

    def _check_array(
        self,
        X: Any,
        name: str = "array",
        ndim: int | None = None,
        dtype: type | None = None,
    ) -> np.ndarray:
        """Validate and convert array input."""
        arr: np.ndarray
        if isinstance(X, np.ndarray):
            arr = X
        else:
            try:
                arr = np.asarray(X, dtype=dtype)
            except (ValueError, TypeError) as e:
                raise TypeError(f"{name} must be array-like, got {type(X).__name__}") from e

        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)

        if ndim is not None and arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}-dimensional, got {arr.ndim}-dimensional array")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN or inf)")

        return arr

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "epsilon": self.epsilon,
            "p": self.p,
            "gamma": self.gamma,
            "solver": self.solver,
            "solver_opts": self.solver_opts,
        }

    def set_params(self, **params: Any) -> DRLCoxEstimator:
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {list(self.get_params().keys())}"
                )

        # Validate new parameters
        self._validate_params()

        return self

    def __repr__(self) -> str:
        """String representation of the estimator."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

    def __sklearn_is_fitted__(self) -> bool:
        """Check if estimator is fitted (for scikit-learn compatibility)."""
        return hasattr(self, "beta_")


def make_drl_cox_scorer(
    metric: Literal["cindex", "iauc"] = "cindex", **metric_kwargs: Any
) -> Callable[..., float]:
    """
    Create a scikit-learn compatible scorer for DRL-Cox models.

    Standard scikit-learn scorers don't support survival data (X, y, zeta).
    This function creates custom scorers that work with GridSearchCV and
    cross_val_score.

    Parameters
    ----------
    metric : {"cindex", "iauc"}, default="cindex"
        Metric to use for scoring:
        - "cindex": Concordance index (Harrell's C-index)
        - "iauc": Time-dependent integrated AUC

    **metric_kwargs : dict
        Additional keyword arguments passed to the metric function.
        For iauc: can pass 'average' parameter.

    Returns
    -------
    scorer : callable
        Scorer function compatible with scikit-learn.

    Examples
    --------
    >>> from drl_cox import DRLCoxEstimator, make_drl_cox_scorer
    >>> from sklearn.model_selection import cross_val_score
    >>>
    >>> # Create scorer
    >>> cindex_scorer = make_drl_cox_scorer(metric="cindex")
    >>>
    >>> # Use with cross-validation
    >>> estimator = DRLCoxEstimator(epsilon=0.1)
    >>>
    >>> # Note: Need to pass zeta as fit_params
    >>> # This is a limitation of scikit-learn's cross_val_score
    >>> # Better to use custom CV or GridSearchCV with custom scoring

    >>> # Use with GridSearchCV
    >>> from sklearn.model_selection import GridSearchCV
    >>>
    >>> param_grid = {"epsilon": [0.0, 0.1, 0.2]}
    >>>
    >>> # Create custom CV splitter that handles zeta
    >>> # Or use the estimator's score method directly
    >>> grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring=cindex_scorer)

    Notes
    -----
    Due to scikit-learn's API limitations, using this scorer with standard
    CV functions requires workarounds. Consider using the estimator's
    score() method directly or implementing custom CV loops.
    """

    def scorer(
        estimator: DRLCoxEstimator,
        X: np.ndarray,
        y: np.ndarray,
        zeta: np.ndarray | None = None,
    ) -> float:
        """Score function for survival data."""
        if zeta is None:
            zeta = np.ones(len(y), dtype=int)

        risk_scores = estimator.predict(X)

        if metric == "cindex":
            score = concordance_index(risk_scores, y, zeta)
        elif metric == "iauc":
            score = time_dependent_auc_iAUC(risk_scores, y, zeta, **metric_kwargs)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return score

    return scorer
