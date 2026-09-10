"""
Benchmark DRL-Cox against classical Cox baselines across synthetic scenarios.

Scenarios
---------
- Clean: no contamination.
- Covariate shift: Gaussian replacement on a subset of features with varying mean shift.
- Outliers: additive noise injected into a random subset of samples.
- High-dimensional: more features than samples, no additional contamination.

For each scenario and repeat the script records:
    * Harrell's C-index
    * time-dependent iAUC
    * training wall-clock time
    * peak memory usage (tracemalloc)

Outputs
-------
- CSV with raw per-run results.
- CSV with aggregated summary (mean/std per scenario & model).
- CSV tables per metric (pivoted for easy documentation inclusion).
- Optional bar plots for metrics, saved as PNG.

Usage
-----
    python examples/benchmark_comparison.py --repeats 3

Use `--help` for more options.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from drl_cox import (
    CoxLasso,
    CoxPartialLikelihood,
    CoxRidge,
    SurvivalDataset,
    concordance_index,
    cross_validate_epsilon,
    fit_drl_cox,
    inject_covariate_shift,
    inject_outliers,
    risk_linear_predictor,
    simulate_cox_data,
    time_dependent_auc_iAUC,
)

# ---------------------------------------------------------------------------#
# Experiment configuration dataclasses
# ---------------------------------------------------------------------------#


@dataclass(frozen=True)
class ScenarioVariant:
    """Single benchmark configuration."""

    scenario: str
    variant: str
    description: str
    generator: Callable[[int], tuple[SurvivalDataset, SurvivalDataset]]


@dataclass(frozen=True)
class ModelSpec:
    """Wrapper for model training callable."""

    name: str
    family: str
    fit: Callable[[SurvivalDataset], tuple[np.ndarray, dict[str, Any]]]


# ---------------------------------------------------------------------------#
# Scenario builders
# ---------------------------------------------------------------------------#


def _generate_base_data(
    n_train: int, n_test: int, d: int, seed: int
) -> tuple[SurvivalDataset, SurvivalDataset]:
    """Simulate train/test data with independent seeds."""
    train = simulate_cox_data(n=n_train, d=d, seed=seed)
    test = simulate_cox_data(n=n_test, d=d, seed=seed + 10_000)
    return train, test


def build_scenarios() -> list[ScenarioVariant]:
    """Define benchmark scenarios."""
    scenarios: list[ScenarioVariant] = []
    base_n_train, base_n_test, base_d = 600, 300, 20

    # Clean scenario
    def _clean_generator(seed: int) -> tuple[SurvivalDataset, SurvivalDataset]:
        return _generate_base_data(base_n_train, base_n_test, base_d, seed)

    scenarios.append(
        ScenarioVariant(
            scenario="Clean",
            variant="no_contamination",
            description="Clean data without contamination (n=600, d=20)",
            generator=_clean_generator,
        )
    )

    # Covariate shift variants
    for mean_shift in (0.5, 1.5, 3.0):
        variant_name = f"shift_mean_{str(mean_shift).replace('.', 'p')}"

        def _cov_shift(
            seed: int, mean: float = mean_shift
        ) -> tuple[SurvivalDataset, SurvivalDataset]:
            train, test = _generate_base_data(base_n_train, base_n_test, base_d, seed)
            feature_indices = range(train.X.shape[1] // 2)
            shifted_X = inject_covariate_shift(
                train.X,
                feature_indices=feature_indices,
                mean=mean,
                std=1.5,
                seed=seed + 500,
            )
            return SurvivalDataset(X=shifted_X, y=train.y, zeta=train.zeta), test

        scenarios.append(
            ScenarioVariant(
                scenario="Covariate shift",
                variant=variant_name,
                description=f"Covariate shift on half the features (mean={mean_shift}, std=1.5)",
                generator=_cov_shift,
            )
        )

    # Outlier contamination variants
    for ratio in (0.05, 0.15, 0.30):
        percent = int(ratio * 100)
        variant_name = f"outliers_{percent}pct"

        def _outliers(
            seed: int, out_ratio: float = ratio
        ) -> tuple[SurvivalDataset, SurvivalDataset]:
            train, test = _generate_base_data(base_n_train, base_n_test, base_d, seed)
            noisy_X = inject_outliers(
                train.X,
                ratio=out_ratio,
                severity_std=5.0,
                seed=seed + 700,
            )
            return SurvivalDataset(X=noisy_X, y=train.y, zeta=train.zeta), test

        scenarios.append(
            ScenarioVariant(
                scenario="Outliers",
                variant=variant_name,
                description=f"Additive Gaussian outliers on {percent}% of samples (std=5.0)",
                generator=_outliers,
            )
        )

    # High-dimensional scenario
    def _high_dim(seed: int) -> tuple[SurvivalDataset, SurvivalDataset]:
        n_train, n_test, d = 120, 300, 200
        return _generate_base_data(n_train, n_test, d, seed)

    scenarios.append(
        ScenarioVariant(
            scenario="High-dimensional",
            variant="d200_gt_n120",
            description="High-dimensional regime with d=200 > n_train=120",
            generator=_high_dim,
        )
    )

    return scenarios


# ---------------------------------------------------------------------------#
# Model builders
# ---------------------------------------------------------------------------#


def make_drl_model(
    epsilon_grid: Iterable[float],
    *,
    p: float = 2.0,
    gamma: int = 3,
    solver: str = "CLARABEL",
    solver_opts: dict[str, Any] | None = None,
    cv_kfolds: int = 3,
) -> ModelSpec:
    """Create DRL-Cox model with simple epsilon selection via cross-validation."""
    eps_values = sorted(set(float(eps) for eps in epsilon_grid))
    if not eps_values:
        raise ValueError("epsilon_grid must contain at least one value.")
    solver_opts = solver_opts or {"max_iter": 200}

    def _fit(train_data: SurvivalDataset) -> tuple[np.ndarray, dict[str, Any]]:
        cv_start = time.perf_counter()
        cv_df = cross_validate_epsilon(
            train_data,
            epsilons=eps_values,
            p=p,
            gamma=gamma,
            kfolds=cv_kfolds,
            metric="cindex",
            solver=solver,
            solver_opts=solver_opts,
        )
        cv_elapsed = time.perf_counter() - cv_start
        summary = cv_df.groupby("epsilon")["score"].mean().dropna()
        if summary.empty:
            raise RuntimeError("Cross-validation failed for all epsilon values.")
        best_eps = float(summary.idxmax())
        best_score = float(summary.loc[best_eps])

        result = fit_drl_cox(
            train_data,
            epsilon=best_eps,
            p=p,
            gamma=gamma,
            solver=solver,
            solver_opts=solver_opts,
        )
        if result.beta is None:
            raise RuntimeError(f"Solver failed with status: {result.status}")
        info = {
            "best_epsilon": best_eps,
            "cv_mean_score": best_score,
            "cv_time": cv_elapsed,
            "solver_status": result.status,
            "objective_value": result.objective_value,
        }
        return result.beta, info

    return ModelSpec(name="DRL-Cox", family="drl", fit=_fit)


def make_partial_likelihood_model(max_iter: int = 40, tol: float = 1e-6) -> ModelSpec:
    def _fit(train_data: SurvivalDataset) -> tuple[np.ndarray, dict[str, Any]]:
        estimator = CoxPartialLikelihood(max_iter=max_iter, tol=tol)
        beta = estimator.fit(train_data.X, train_data.y, train_data.zeta)
        return beta, {"max_iter": max_iter, "tol": tol}

    return ModelSpec(name="Cox-PL", family="baseline", fit=_fit)


def make_ridge_model(alpha: float = 1.0, max_iter: int = 50, tol: float = 1e-6) -> ModelSpec:
    def _fit(train_data: SurvivalDataset) -> tuple[np.ndarray, dict[str, Any]]:
        estimator = CoxRidge(alpha=alpha, max_iter=max_iter, tol=tol)
        beta = estimator.fit(train_data.X, train_data.y, train_data.zeta)
        return beta, {"alpha": alpha}

    return ModelSpec(name=f"Cox-Ridge (alpha={alpha})", family="baseline", fit=_fit)


def make_lasso_model(alpha: float = 0.05, max_iter: int = 120, tol: float = 1e-6) -> ModelSpec:
    def _fit(train_data: SurvivalDataset) -> tuple[np.ndarray, dict[str, Any]]:
        estimator = CoxLasso(alpha=alpha, max_iter=max_iter, tol=tol)
        beta = estimator.fit(train_data.X, train_data.y, train_data.zeta)
        return beta, {"alpha": alpha}

    return ModelSpec(name=f"Cox-Lasso (alpha={alpha})", family="baseline", fit=_fit)


# ---------------------------------------------------------------------------#
# Evaluation utilities
# ---------------------------------------------------------------------------#


def evaluate_model(
    model_spec: ModelSpec,
    train_data: SurvivalDataset,
    test_data: SurvivalDataset,
) -> dict[str, Any]:
    """Fit a model and compute metrics on the test set."""
    tracemalloc.start()
    start = time.perf_counter()
    status = "ok"
    error_message = ""
    info: dict[str, Any] = {}
    beta: np.ndarray | None = None

    try:
        beta, info = model_spec.fit(train_data)
    except Exception as exc:  # noqa: BLE001 - capture any failure
        status = "error"
        error_message = str(exc)
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / (1024 * 1024)

    row: dict[str, Any] = {
        "model": model_spec.name,
        "family": model_spec.family,
        "train_time": elapsed,
        "memory_mb": memory_mb,
        "status": status,
        "error_message": error_message,
    }

    row.update(info)

    if status != "ok" or beta is None or not np.all(np.isfinite(beta)):
        row["c_index"] = np.nan
        row["iauc"] = np.nan
        return row

    risk_scores = risk_linear_predictor(test_data.X, beta)
    c_index = concordance_index(risk_scores, test_data.y, test_data.zeta)
    iauc = time_dependent_auc_iAUC(risk_scores, test_data.y, test_data.zeta, average="event")

    row["c_index"] = float(c_index)
    row["iauc"] = float(iauc)
    return row


def run_benchmark(
    scenarios: Sequence[ScenarioVariant],
    models: Sequence[ModelSpec],
    *,
    repeats: int,
) -> pd.DataFrame:
    """Run all scenario/model combinations."""
    rows: list[dict[str, Any]] = []

    for scenario_idx, variant in enumerate(scenarios):
        base_seed = 10_000 * (scenario_idx + 1)
        for repeat in range(repeats):
            seed = base_seed + repeat
            train_data, test_data = variant.generator(seed)
            for model in models:
                result_row = evaluate_model(model, train_data, test_data)
                result_row.update(
                    {
                        "scenario": variant.scenario,
                        "variant": variant.variant,
                        "scenario_description": variant.description,
                        "repeat": repeat,
                        "seed": seed,
                        "n_train": train_data.X.shape[0],
                        "n_test": test_data.X.shape[0],
                        "n_features": train_data.X.shape[1],
                    }
                )
                rows.append(result_row)

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per scenario/model summary statistics."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    summary = (
        ok.groupby(["scenario", "variant", "model"], as_index=False)
        .agg(
            c_index_mean=("c_index", "mean"),
            c_index_std=("c_index", "std"),
            iauc_mean=("iauc", "mean"),
            iauc_std=("iauc", "std"),
            train_time_mean=("train_time", "mean"),
            train_time_std=("train_time", "std"),
            memory_mb_mean=("memory_mb", "mean"),
            memory_mb_std=("memory_mb", "std"),
        )
        .sort_values(["scenario", "variant", "model"])
    )
    return summary


def export_metric_tables(summary: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Create pivot tables per metric and save as CSV."""
    tables: dict[str, Path] = {}
    if summary.empty:
        return tables

    metric_map = {
        "c_index_mean": "table_c_index_mean.csv",
        "iauc_mean": "table_iauc_mean.csv",
        "train_time_mean": "table_train_time_mean.csv",
        "memory_mb_mean": "table_memory_mean.csv",
    }
    for metric_col, filename in metric_map.items():
        pivot = summary.pivot_table(
            index=["scenario", "variant"],
            columns="model",
            values=metric_col,
        )
        csv_path = output_dir / filename
        pivot.to_csv(csv_path)
        tables[metric_col] = csv_path
    return tables


def plot_metric(
    results: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    yscale: str | None = None,
) -> Path | None:
    """Create a facet bar plot for a specific metric."""
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return None

    g = sns.catplot(
        data=ok,
        kind="bar",
        x="variant",
        y=metric,
        hue="model",
        col="scenario",
        errorbar="sd",
        height=4.2,
        aspect=1.2,
        sharey=False,
    )
    g.set_axis_labels("Variant", ylabel)
    g.set_titles("{col_name}")
    if yscale:
        for ax in g.axes.flatten():
            ax.set_yscale(yscale)
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=30)
    g.fig.suptitle(title, fontsize=14)
    g.tight_layout()
    g.fig.subplots_adjust(top=0.88)
    g.savefig(output_path, dpi=300)
    plt.close(g.fig)
    return output_path


# ---------------------------------------------------------------------------#
# CLI
# ---------------------------------------------------------------------------#


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DRL-Cox against classical baselines.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of random repeats per scenario (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory to store CSV outputs and plots (default: examples/results).",
    )
    parser.add_argument(
        "--epsilon-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3],
        help="Candidate epsilon values for DRL-Cox cross-validation.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=3,
        help="Neighborhood size (gamma) for DRL-Cox robust optimization.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=2.0,
        help="Norm parameter p for the Wasserstein ball in DRL-Cox.",
    )
    parser.add_argument(
        "--cv-kfolds",
        type=int,
        default=3,
        help="Number of folds for DRL-Cox epsilon cross-validation.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generation of plots (useful for headless or quick runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")

    scenarios = build_scenarios()
    models = [
        make_drl_model(
            args.epsilon_grid,
            p=args.p,
            gamma=args.gamma,
            cv_kfolds=args.cv_kfolds,
        ),
        make_partial_likelihood_model(),
        make_ridge_model(alpha=1.0),
        make_lasso_model(alpha=0.05),
    ]

    print(
        f"Running benchmark for {len(scenarios)} scenarios, {len(models)} models, "
        f"{args.repeats} repeats..."
    )
    results_df = run_benchmark(scenarios, models, repeats=args.repeats)

    raw_path = output_dir / "benchmark_comparison_results.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"Raw results saved to {raw_path}")

    summary_df = summarize_results(results_df)
    summary_path = output_dir / "benchmark_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    tables = export_metric_tables(summary_df, output_dir)
    for metric_col, path in tables.items():
        print(f"Table for {metric_col} written to {path}")

    if not args.skip_plots:
        plot_metric(
            results_df,
            metric="c_index",
            ylabel="C-index",
            title="C-index comparison",
            output_path=output_dir / "plot_c_index.png",
        )
        plot_metric(
            results_df,
            metric="iauc",
            ylabel="iAUC",
            title="Time-dependent iAUC comparison",
            output_path=output_dir / "plot_iauc.png",
        )
        plot_metric(
            results_df,
            metric="train_time",
            ylabel="Training time (s)",
            title="Training time comparison",
            output_path=output_dir / "plot_train_time.png",
            yscale="log",
        )
        plot_metric(
            results_df,
            metric="memory_mb",
            ylabel="Peak memory (MB)",
            title="Peak memory usage comparison",
            output_path=output_dir / "plot_memory.png",
        )

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
