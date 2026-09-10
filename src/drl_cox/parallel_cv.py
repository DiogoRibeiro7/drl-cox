"""
Optimized cross_validate_epsilon with parallel execution support.

This module provides an optimized version of the cross-validation function
with parallel processing, progress bars, and proper memory management.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .drl_cox import (
    DEFAULT_SOLVER,
    SurvivalDataset,
    fit_drl_cox,
    kfold_indices,
    risk_linear_predictor,
)
from .metrics import concordance_index, time_dependent_auc_iAUC


def _fit_single_fold(
    train_data: SurvivalDataset,
    val_data: SurvivalDataset,
    epsilon: float,
    p: float,
    gamma: int,
    metric: Literal["cindex", "iauc"],
    solver: str,
    solver_opts: dict[str, Any] | None,
    iauc_average: Literal["uniform", "event"],
    fold_id: int,
    random_seed: int,
) -> dict[str, Any]:
    """
    Fit DRL-Cox on a single fold and compute validation score.

    This function is designed to be called in parallel across folds.

    Parameters
    ----------
    train_data : SurvivalDataset
        Training data for this fold
    val_data : SurvivalDataset
        Validation data for this fold
    epsilon : float
        Robustness parameter
    p : float
        Norm parameter
    gamma : int
        Number of risk set constraints
    metric : Literal["cindex", "iauc"]
        Evaluation metric
    solver : str
        CVXPY solver name
    solver_opts : Optional[Dict[str, Any]]
        Solver options
    iauc_average : Literal["uniform", "event"]
        Averaging method for iAUC
    fold_id : int
        Fold identifier
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Dictionary containing epsilon, fold_id, score, and status
    """
    # Set random seed for this fold to ensure reproducibility
    np.random.seed(random_seed + fold_id)

    try:
        # Fit DRL-Cox on training data
        result = fit_drl_cox(
            train_data,
            epsilon=epsilon,
            p=p,
            gamma=gamma,
            solver=solver,
            solver_opts=solver_opts or {},
        )

        # Check if optimization was successful
        if result.status not in ["optimal", "optimal_inaccurate"]:
            warnings.warn(
                f"Fold {fold_id}, ε={epsilon}: Solver status '{result.status}'",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"epsilon": epsilon, "fold": fold_id, "score": np.nan, "status": result.status}

        # Compute validation risk scores
        beta = result.beta
        risk_val = risk_linear_predictor(val_data.X, beta)

        # Compute validation metric
        if metric == "cindex":
            score = concordance_index(risk_val, val_data.y, val_data.zeta)
        else:  # iauc
            score = time_dependent_auc_iAUC(
                risk_val, val_data.y, val_data.zeta, times=None, average=iauc_average
            )

        return {"epsilon": epsilon, "fold": fold_id, "score": score, "status": result.status}

    except Exception as e:
        warnings.warn(
            f"Fold {fold_id}, ε={epsilon}: Error during fitting: {str(e)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "epsilon": epsilon,
            "fold": fold_id,
            "score": np.nan,
            "status": f"error: {str(e)[:50]}",
        }


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
    n_jobs: int = 1,
    verbose: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Cross-validate DRL-Cox across multiple epsilon values with parallel execution.

    This optimized version supports parallel processing across folds and epsilon
    values, with progress bars and proper memory management.

    Parameters
    ----------
    data : SurvivalDataset
        Complete dataset for cross-validation
    epsilons : Iterable[float]
        Epsilon values to test
    p : float, default=2.0
        Norm parameter for Wasserstein distance
    gamma : int, default=3
        Number of risk set constraints per observation
    kfolds : int, default=5
        Number of cross-validation folds
    metric : Literal["cindex", "iauc"], default="cindex"
        Evaluation metric
    solver : str, default=DEFAULT_SOLVER
        CVXPY solver name
    solver_opts : Optional[Dict[str, Any]], default=None
        Solver-specific options
    iauc_average : Literal["uniform", "event"], default="event"
        Averaging method for iAUC (only used if metric="iauc")
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all available cores,
        -2 for all but one, etc. Use 1 for sequential execution.
    verbose : bool, default=True
        Show progress bars
    random_seed : int, default=42
        Random seed for fold generation and reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: epsilon, fold, score, status

    Examples
    --------
    >>> from drl_cox import simulate_cox_data, cross_validate_epsilon
    >>> data = simulate_cox_data(n=200, d=10, seed=42)
    >>>
    >>> # Sequential execution
    >>> results = cross_validate_epsilon(data, epsilons=[0.0, 0.1, 0.2], kfolds=5, n_jobs=1)
    >>>
    >>> # Parallel execution with 4 cores
    >>> results = cross_validate_epsilon(data, epsilons=[0.0, 0.1, 0.2], kfolds=5, n_jobs=4)
    >>>
    >>> # Use all available cores
    >>> results = cross_validate_epsilon(data, epsilons=[0.0, 0.1, 0.2], kfolds=5, n_jobs=-1)

    Notes
    -----
    - Parallelization is done across (epsilon, fold) combinations
    - Each worker gets an independent random seed for reproducibility
    - Memory management is handled by joblib with max_nbytes parameter
    - Progress bars show completion status for each epsilon value
    - NaN scores indicate solver failures or errors
    """
    # Convert epsilons to list for consistency
    epsilon_list = list(epsilons)

    # Generate fold indices (deterministic with random_seed)
    N = data.X.shape[0]
    folds = kfold_indices(N, k=kfolds, seed=random_seed)

    # Create all (epsilon, fold) combinations
    tasks = []
    for eps in epsilon_list:
        for fold_id in range(kfolds):
            # Get train/val indices
            val_idx = folds[fold_id]
            train_idx = np.setdiff1d(np.arange(N), val_idx)

            # Create train and validation datasets
            train_data = SurvivalDataset(data.X[train_idx], data.y[train_idx], data.zeta[train_idx])
            val_data = SurvivalDataset(data.X[val_idx], data.y[val_idx], data.zeta[val_idx])

            tasks.append(
                {
                    "train_data": train_data,
                    "val_data": val_data,
                    "epsilon": eps,
                    "fold_id": fold_id,
                }
            )

    # Determine number of jobs
    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1
    elif n_jobs < -1:
        import os

        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs)

    # Prepare progress bar
    total_tasks = len(tasks)
    desc = f"CV: {len(epsilon_list)} epsilons × {kfolds} folds"

    if verbose:
        print("Starting cross-validation:")
        print(f"  - Epsilon values: {len(epsilon_list)}")
        print(f"  - Folds: {kfolds}")
        print(f"  - Total tasks: {total_tasks}")
        print(f"  - Parallel jobs: {n_jobs}")
        print(f"  - Metric: {metric}")
        print()

    # Execute in parallel with progress bar
    if n_jobs == 1:
        # Sequential execution
        results = []
        for task in tqdm(tasks, desc=desc, disable=not verbose):
            result = _fit_single_fold(
                train_data=task["train_data"],
                val_data=task["val_data"],
                epsilon=task["epsilon"],
                p=p,
                gamma=gamma,
                metric=metric,
                solver=solver,
                solver_opts=solver_opts,
                iauc_average=iauc_average,
                fold_id=task["fold_id"],
                random_seed=random_seed,
            )
            results.append(result)
    else:
        # Parallel execution
        results = Parallel(
            n_jobs=n_jobs,
            verbose=0,
            max_nbytes="100M",  # Limit memory usage for large arrays
            backend="loky",  # Use loky backend for better process isolation
        )(
            delayed(_fit_single_fold)(
                train_data=task["train_data"],
                val_data=task["val_data"],
                epsilon=task["epsilon"],
                p=p,
                gamma=gamma,
                metric=metric,
                solver=solver,
                solver_opts=solver_opts,
                iauc_average=iauc_average,
                fold_id=task["fold_id"],
                random_seed=random_seed,
            )
            for task in tqdm(tasks, desc=desc, disable=not verbose)
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Report any failures
    if verbose:
        n_failures = results_df["score"].isna().sum()
        if n_failures > 0:
            print(f"\n⚠️  Warning: {n_failures}/{total_tasks} tasks failed")
            print("   Check 'status' column for details")
        else:
            print(f"\n✓ All {total_tasks} tasks completed successfully")

    return results_df


# Benchmark function
def benchmark_parallel_cv(
    n: int = 500,
    d: int = 20,
    kfolds: int = 10,
    epsilons: list[float] | None = None,
    n_jobs_list: list[int] | None = None,
) -> pd.DataFrame:
    """
    Benchmark parallel cross-validation performance.

    Parameters
    ----------
    n : int, default=500
        Number of samples
    d : int, default=20
        Number of features
    kfolds : int, default=10
        Number of folds
    epsilons : Optional[list[float]], default=None
        Epsilon values to test (default: [0.0, 0.1, 0.2, 0.3])
    n_jobs_list : Optional[list[int]], default=None
        List of n_jobs values to benchmark (default: [1, 2, 4, -1])

    Returns
    -------
    pd.DataFrame
        Benchmark results with columns: n_jobs, time_seconds, speedup

    Examples
    --------
    >>> from drl_cox import benchmark_parallel_cv
    >>>
    >>> # Benchmark with default settings
    >>> results = benchmark_parallel_cv()
    >>> print(results)
    >>>
    >>> # Custom benchmark
    >>> results = benchmark_parallel_cv(
    ...     n=1000, d=30, kfolds=5, epsilons=[0.0, 0.1, 0.2], n_jobs_list=[1, 2, 4, 8]
    ... )
    """
    import time

    from .datasets import simulate_cox_data

    if epsilons is None:
        epsilons = [0.0, 0.1, 0.2, 0.3]

    if n_jobs_list is None:
        n_jobs_list = [1, 2, 4, -1]

    print("=" * 80)
    print("PARALLEL CROSS-VALIDATION BENCHMARK")
    print("=" * 80)
    print(f"\nDataset: n={n}, d={d}")
    print(f"CV setup: {len(epsilons)} epsilons × {kfolds} folds = {len(epsilons) * kfolds} tasks")
    print(f"Testing n_jobs: {n_jobs_list}")
    print()

    # Generate synthetic data
    print("Generating synthetic data...")
    data = simulate_cox_data(n=n, d=d, seed=42, baseline_hazard=0.02, censor_rate=0.35)
    print(f"Data generated: {data.X.shape[0]} samples, {data.X.shape[1]} features")
    print()

    # Benchmark each n_jobs setting
    benchmark_results = []

    for n_jobs in n_jobs_list:
        print(f"Testing n_jobs={n_jobs}...")

        start_time = time.time()

        results = cross_validate_epsilon(
            data,
            epsilons=epsilons,
            p=2.0,
            gamma=3,
            kfolds=kfolds,
            metric="cindex",
            solver=DEFAULT_SOLVER,
            solver_opts={"max_iter": 200},
            n_jobs=n_jobs,
            verbose=False,
        )

        elapsed_time = time.time() - start_time

        # Check for failures
        n_failures = results["score"].isna().sum()
        mean_score = results["score"].mean()

        benchmark_results.append(
            {
                "n_jobs": n_jobs,
                "time_seconds": elapsed_time,
                "mean_score": mean_score,
                "n_failures": n_failures,
            }
        )

        print(f"   Time: {elapsed_time:.2f}s")
        print(f"   Mean score: {mean_score:.4f}")
        print(f"   Failures: {n_failures}")
        print()

    # Create results DataFrame
    benchmark_df = pd.DataFrame(benchmark_results)

    # Calculate speedup relative to sequential execution
    baseline_time = benchmark_df[benchmark_df["n_jobs"] == 1]["time_seconds"].values[0]
    benchmark_df["speedup"] = baseline_time / benchmark_df["time_seconds"]
    benchmark_df["efficiency"] = benchmark_df["speedup"] / benchmark_df["n_jobs"].abs()

    # Display results
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(benchmark_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    best_row = benchmark_df.loc[benchmark_df["speedup"].idxmax()]
    print(f"Best speedup: {best_row['speedup']:.2f}x with n_jobs={best_row['n_jobs']}")
    print()

    return benchmark_df
