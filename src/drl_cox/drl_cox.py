"""Core DRL-Cox solver, survival dataset container and epsilon cross-validation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import cvxpy as cp
import numpy as np
import pandas as pd

from .metrics import concordance_index, time_dependent_auc_iAUC

__all__ = [
    "DEFAULT_SOLVER",
    "SurvivalDataset",
    "DRLCoxResult",
    "fit_drl_cox",
    "risk_linear_predictor",
    "kfold_indices",
    "cross_validate_epsilon",
]

DEFAULT_SOLVER = "CLARABEL"
"""Default CVXPY solver.

Clarabel is an open-source interior-point solver that supports the exponential cone
required by DRL-Cox and is installed together with CVXPY on every supported Python
version. ``"SCS"`` is a first-order alternative for large problems; ``"ECOS"`` can be
used on Python < 3.13 after installing the ``drl-cox[ecos]`` extra.
"""


def _assert_ndarray(name: str, arr: np.ndarray, ndim: int | None = None) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")


def _dual_p(p: float) -> float:
    if p <= 0:
        raise ValueError("p must be >= 1")
    if math.isinf(p):
        return 1.0
    if abs(p - 1.0) < 1e-12:
        return float("inf")
    return 1.0 / (1.0 - 1.0 / p)


def _norm_with_p(x: cp.Expression, q: float) -> cp.Expression:
    return cp.norm_inf(x) if math.isinf(q) else cp.norm(x, q)


@dataclass(frozen=True)
class SurvivalDataset:
    """Right-censored survival data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Covariates.
    y : np.ndarray of shape (n_samples,)
        Observed durations; must be finite and strictly positive.
    zeta : np.ndarray of shape (n_samples,)
        Event indicator: ``1`` if the event was observed, ``0`` if censored.
    validate : bool, default=True
        Check that ``X`` and ``y`` are finite, ``y`` is positive and ``zeta`` is binary.
        Pass ``False`` to wrap raw data (for example with missing values) that is
        destined for the :mod:`drl_cox.preprocessing` utilities. Shapes are always
        checked.
    """

    X: np.ndarray
    y: np.ndarray
    zeta: np.ndarray
    validate: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        _assert_ndarray("X", self.X, 2)
        _assert_ndarray("y", self.y, 1)
        _assert_ndarray("zeta", self.zeta, 1)
        n = self.X.shape[0]
        if self.y.shape[0] != n or self.zeta.shape[0] != n:
            raise ValueError("Shapes mismatch: X, y, zeta must agree.")
        if not self.validate:
            return
        if np.any(~np.isfinite(self.X)):
            raise ValueError("X contains non-finite values.")
        if np.any(~np.isfinite(self.y)) or np.any(self.y <= 0):
            raise ValueError("y must be finite and strictly positive.")
        if set(np.unique(self.zeta)) - {0, 1}:
            raise ValueError("zeta must be binary (0/1).")


@dataclass
class DRLCoxResult:
    """Output of :func:`fit_drl_cox`."""

    beta: np.ndarray
    alpha: float
    s: np.ndarray
    objective_value: float
    status: str
    info: dict[str, Any]


def fit_drl_cox(
    data: SurvivalDataset,
    epsilon: float,
    p: float = 2.0,
    gamma: int = 3,
    solver: str = DEFAULT_SOLVER,
    solver_opts: dict[str, Any] | None = None,
) -> DRLCoxResult:
    """Fit the Wasserstein distributionally robust Cox model.

    Parameters
    ----------
    data : SurvivalDataset
        Training data.
    epsilon : float
        Wasserstein radius (``0`` recovers the regularisation-free problem).
    p : float, default=2.0
        Order of the norm defining the Wasserstein ground metric.
    gamma : int, default=3
        Number of risk-set constraints per observation.
    solver : str, default=DEFAULT_SOLVER
        Name of a CVXPY solver supporting the exponential cone.
    solver_opts : dict, optional
        Keyword arguments forwarded to ``cvxpy.Problem.solve``.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0.")
    if gamma < 1:
        raise ValueError("gamma must be >= 1.")

    X, y, z = data.X.copy(), data.y.copy(), data.zeta.copy()
    N, d = X.shape

    order = np.argsort(-y)
    Xs = X[order]
    ys = y[order]
    zs = z[order]

    beta = cp.Variable(d)
    alpha = cp.Variable(1)
    s = cp.Variable(N)

    q = _dual_p(p)
    beta_dot_X = Xs @ beta

    constraints: list[cp.Constraint] = []
    for i in range(N):
        i_to = min(N - 1, i + gamma - 1)
        for k in range(i, i_to + 1):
            concat = cp.hstack([beta_dot_X[i], beta_dot_X[: (k + 1)]])
            lse_expr = cp.log_sum_exp(concat)
            rhs = lse_expr - beta_dot_X[i] - alpha * (ys[i] - ys[k])
            constraints.append(s[i] >= rhs)

    reg = epsilon * _norm_with_p(cp.hstack([beta, alpha]), q)
    empirical = (1.0 / N) * cp.sum(cp.multiply(zs, s))
    problem = cp.Problem(cp.Minimize(reg + empirical), constraints)

    problem.solve(solver=solver, **(solver_opts or {}))

    alpha_value = float("nan")
    if alpha.value is not None:
        alpha_value = float(np.asarray(alpha.value, dtype=float).reshape(-1)[0])

    return DRLCoxResult(
        beta=np.asarray(beta.value, dtype=float).reshape(-1),
        alpha=alpha_value,
        s=np.asarray(s.value, dtype=float).reshape(-1),
        objective_value=float(problem.value) if problem.value is not None else float("nan"),
        status=str(problem.status),
        info={"solver": solver, "q": q, "epsilon": epsilon, "gamma": gamma},
    )


def risk_linear_predictor(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Linear risk score ``X @ beta`` (higher means higher hazard)."""
    _assert_ndarray("X", X, 2)
    _assert_ndarray("beta", beta, 1)
    if X.shape[1] != beta.shape[0]:
        raise ValueError("X.shape[1] must match beta.shape[0]")
    return np.asarray(X @ beta)


def kfold_indices(n: int, k: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Shuffled, sorted validation index sets for ``k`` folds."""
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [np.sort(f) for f in folds]


def cross_validate_epsilon(
    data: SurvivalDataset,
    epsilons: Iterable[float],
    p: float = 2.0,
    gamma: int = 3,
    kfolds: int = 5,
    metric: Literal["cindex", "iauc"] = "cindex",
    solver: str = DEFAULT_SOLVER,
    solver_opts: dict[str, Any] | None = None,
    iauc_average: Literal["uniform", "event"] = "event",
) -> pd.DataFrame:
    """Sequential k-fold cross-validation over a grid of ``epsilon`` values.

    Returns a data frame with one row per ``(epsilon, fold)`` pair and the
    validation ``score`` for the chosen metric. See
    :func:`drl_cox.parallel_cv.cross_validate_epsilon` for a parallel variant.
    """
    N = data.X.shape[0]
    folds = kfold_indices(N, k=kfolds, seed=123)
    rows: list[dict[str, float | int]] = []

    for eps in epsilons:
        for fold_id in range(kfolds):
            val_idx = folds[fold_id]
            train_idx = np.setdiff1d(np.arange(N), val_idx)

            train = SurvivalDataset(data.X[train_idx], data.y[train_idx], data.zeta[train_idx])
            val_X, val_y, val_z = data.X[val_idx], data.y[val_idx], data.zeta[val_idx]

            res = fit_drl_cox(
                train, epsilon=eps, p=p, gamma=gamma, solver=solver, solver_opts=solver_opts
            )
            r_val = risk_linear_predictor(val_X, res.beta)

            if metric == "cindex":
                score = concordance_index(r_val, val_y, val_z)
            else:
                score = time_dependent_auc_iAUC(
                    r_val, val_y, val_z, times=None, average=iauc_average
                )

            rows.append({"epsilon": float(eps), "fold": int(fold_id), "score": float(score)})

    return pd.DataFrame(rows)
