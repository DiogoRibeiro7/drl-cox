"""
Adaptive cross-validation utilities for DRL-Cox.

The main entrypoint, :func:`auto_select_epsilon`, performs Bayesian optimisation
over the Wasserstein radius ``epsilon`` by repeatedly calling
``cross_validate_epsilon`` and modelling the response surface with a Gaussian
process.  This provides data-driven epsilon selection together with
uncertainty estimates and diagnostic visualisations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .drl_cox import DEFAULT_SOLVER, SurvivalDataset, cross_validate_epsilon

try:  # pragma: no cover - optional dependency checked at runtime
    from skopt import Optimizer
    from skopt.space import Real
except ImportError:  # pragma: no cover
    Optimizer = None
    Real = None

__all__ = ["AutoSelectionResult", "auto_select_epsilon"]


@dataclass(slots=True)
class AutoSelectionResult:
    """Container returned by :func:`auto_select_epsilon`.

    Attributes
    ----------
    best_epsilon:
        Epsilon value expected to maximise the chosen metric.
    best_score:
        Estimated cross-validation score at ``best_epsilon``.
    confidence_interval:
        Tuple ``(lower, upper)`` with a 95% credible interval produced from the
        Gaussian process posterior.
    trials:
        Data frame with every optimisation step (epsilon, score, duration).
    figure:
        Optional Matplotlib figure describing the response surface.
    surrogate_model:
        The final surrogate model fitted by scikit-optimize (typically a GP).
    """

    best_epsilon: float
    best_score: float
    confidence_interval: tuple[float, float]
    trials: pd.DataFrame
    figure: Figure | None
    surrogate_model: Any | None


def _default_epsilon_bounds(data: SurvivalDataset) -> tuple[float, float]:
    """Heuristic epsilon bounds derived from the feature scale."""
    norms = np.linalg.norm(data.X, axis=1)
    median = float(np.median(norms))
    iqr = float(np.subtract(*np.percentile(norms, [75, 25])))
    upper = max(0.05, (median + iqr) * 0.25)
    lower = max(1e-4, upper * 0.02)
    if lower >= upper:
        upper = lower * 10.0
    return (lower, upper)


def auto_select_epsilon(
    data: SurvivalDataset,
    *,
    metric: Literal["cindex", "iauc"] = "cindex",
    p: float = 2.0,
    gamma: int = 3,
    solver: str = DEFAULT_SOLVER,
    solver_opts: dict[str, Any] | None = None,
    iauc_average: Literal["uniform", "event"] = "event",
    epsilon_bounds: tuple[float, float] | None = None,
    n_calls: int = 20,
    n_initial_points: int = 5,
    random_state: int = 42,
    visualize: bool = True,
) -> AutoSelectionResult:
    """Select ``epsilon`` via Bayesian optimisation.

    Parameters
    ----------
    data:
        Survival dataset used for cross-validation.
    metric:
        Optimisation target.  ``"cindex"`` (default) or ``"iauc"``.
    p, gamma, solver, solver_opts, iauc_average:
        Forwarded to :func:`drl_cox.cross_validate_epsilon`.
    epsilon_bounds:
        Optional tuple ``(lower, upper)`` describing the search domain.  If not
        provided boundaries are inferred from the data scale.
    n_calls:
        Total number of optimisation evaluations (including initial points).
    n_initial_points:
        Number of quasi-random evaluations before using the acquisition
        function.  Must be smaller than ``n_calls``.
    random_state:
        Deterministic seed for reproducibility.
    visualize:
        If ``True`` (default) produce a Matplotlib figure depicting the
        surrogate mean, confidence band, and observed evaluations.

    Returns
    -------
    AutoSelectionResult
        Dataclass summarising the optimisation outcome.

    Examples
    --------
    >>> from drl_cox import simulate_cox_data
    >>> from drl_cox.cross_validation import auto_select_epsilon
    >>> dataset = simulate_cox_data(n=120, d=12, seed=0)
    >>> result = auto_select_epsilon(dataset, n_calls=10, random_state=0)
    >>> result.best_epsilon  # doctest: +SKIP
    0.072
    >>> result.confidence_interval  # doctest: +SKIP
    (0.041, 0.11)
    """
    if Optimizer is None or Real is None:  # pragma: no cover - runtime guard
        raise ImportError(
            "scikit-optimize is required for auto_select_epsilon. "
            "Install with `pip install scikit-optimize`."
        )
    if metric not in {"cindex", "iauc"}:
        raise ValueError("metric must be 'cindex' or 'iauc'.")
    if n_calls <= 0:
        raise ValueError("n_calls must be positive.")
    if not (0 < n_initial_points < n_calls):
        raise ValueError("n_initial_points must be in (0, n_calls).")

    bounds = epsilon_bounds or _default_epsilon_bounds(data)
    space = [Real(bounds[0], bounds[1], prior="log-uniform")]
    optimizer = Optimizer(
        dimensions=space,
        base_estimator="GP",
        acq_func="gp_hedge",
        acq_optimizer="sampling",
        n_initial_points=n_initial_points,
        random_state=random_state,
    )

    evaluations: list[dict[str, float]] = []

    for _ in range(n_calls):
        epsilon = float(optimizer.ask()[0])
        scores_df = cross_validate_epsilon(
            data,
            epsilons=[epsilon],
            p=p,
            gamma=gamma,
            kfolds=5,
            metric=metric,
            solver=solver,
            solver_opts=solver_opts,
            iauc_average=iauc_average,
        )
        score = float(scores_df["score"].mean())
        if not np.isfinite(score):
            objective = 1e2
        else:
            objective = -score

        optimizer.tell([epsilon], objective)
        evaluations.append({"epsilon": epsilon, "score": score})

    trials = pd.DataFrame(evaluations)
    best_idx = int(trials["score"].idxmax())
    best_epsilon = float(trials["epsilon"].to_numpy()[best_idx])
    best_score = float(trials["score"].to_numpy()[best_idx])

    figure: Figure | None = None
    ci_low, ci_high = bounds

    if optimizer.models:
        surrogate = optimizer.models[-1]
        grid = np.linspace(bounds[0], bounds[1], 200).reshape(-1, 1)
        mean_obj, std_pred = surrogate.predict(grid, return_std=True)
        mean_obj = mean_obj.astype(float)
        std_pred = std_pred.astype(float)
        crit = 1.96

        mean_score = -mean_obj
        lower_score = mean_score - crit * std_pred
        upper_score = mean_score + crit * std_pred

        idx_best_grid = int(np.abs(grid.flatten() - best_epsilon).argmin())
        best_lower = lower_score[idx_best_grid]
        credible_mask = upper_score >= best_lower

        if np.any(credible_mask):
            eps_region = grid.flatten()[credible_mask]
            ci_low = float(np.min(eps_region))
            ci_high = float(np.max(eps_region))
        else:
            ci_low = ci_high = best_epsilon

        if visualize:
            figure, ax = plt.subplots(figsize=(6, 4))
            ax.plot(grid.flatten(), mean_score, label="GP mean (score)", color="tab:blue")
            ax.fill_between(
                grid.flatten(),
                lower_score,
                upper_score,
                color="tab:blue",
                alpha=0.2,
                label="95% credible interval",
            )
            ax.scatter(trials["epsilon"], trials["score"], color="tab:orange", label="Evaluations")
            ax.axvline(best_epsilon, color="tab:green", linestyle="--", label="Best epsilon")
            ax.set_xlabel("epsilon")
            ax.set_ylabel(metric)
            ax.set_title("Bayesian optimisation of epsilon")
            ax.legend()
            ax.set_xlim(bounds)
            ax.grid(alpha=0.3)
    else:
        surrogate = None
        figure = None

    confidence_interval = (ci_low, ci_high)
    return AutoSelectionResult(
        best_epsilon=best_epsilon,
        best_score=best_score,
        confidence_interval=confidence_interval,
        trials=trials,
        figure=figure,
        surrogate_model=surrogate,
    )
