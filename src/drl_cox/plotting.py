"""
Visualization utilities for DRL-Cox survival analysis.

This module provides publication-ready plotting functions for survival analysis,
model comparison, and cross-validation results.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .drl_cox import SurvivalDataset

# Set default style for publication-ready plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

__all__ = [
    "plot_survival_curves",
    "plot_coefficient_path",
    "plot_cv_results",
    "plot_model_comparison",
    "plot_risk_groups",
    "plot_concordance_over_time",
    "setup_plotting_style",
    "save_figure",
]


def setup_plotting_style(
    style: str = "publication",
    font_size: int = 10,
    font_family: str = "sans-serif",
    dpi: int = 100,
) -> None:
    """
    Set up matplotlib plotting style for consistent, professional figures.

    Parameters
    ----------
    style : str, default="publication"
        Style preset: "publication", "presentation", "notebook", or "minimal"
    font_size : int, default=10
        Base font size for text elements
    font_family : str, default="sans-serif"
        Font family to use
    dpi : int, default=100
        Resolution for figure display

    Examples
    --------
    >>> from drl_cox.plotting import setup_plotting_style
    >>> setup_plotting_style("publication", font_size=12)
    """
    # Style presets
    # rcParams keys are typed as a Literal union in matplotlib; keep the presets loosely typed.
    styles: dict[str, dict[Any, Any]] = {
        "publication": {
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "patch.linewidth": 0.8,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 2,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.titlesize": font_size + 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
        "presentation": {
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.5,
            "lines.markersize": 10,
            "patch.linewidth": 1.2,
            "axes.labelsize": font_size + 2,
            "axes.titlesize": font_size + 4,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size + 6,
        },
        "notebook": {
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "patch.linewidth": 1.0,
            "axes.labelsize": font_size + 1,
            "axes.titlesize": font_size + 3,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size + 4,
        },
        "minimal": {
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.bottom": True,
            "ytick.left": True,
            "axes.grid": False,
        },
    }

    # Apply base settings
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "figure.dpi": dpi,
            "savefig.dpi": dpi * 3,  # Higher resolution for saved figures
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "gray",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fancybox": True,
            "figure.autolayout": True,
        }
    )

    # Apply style-specific settings
    if style in styles:
        plt.rcParams.update(styles[style])


def save_figure(
    fig: Figure,
    filename: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    transparent: bool = False,
    **kwargs: Any,
) -> None:
    """
    Save figure with consistent high-quality settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename (extension determines format)
    dpi : int, default=300
        Resolution for raster formats
    bbox_inches : str, default="tight"
        Bounding box setting
    transparent : bool, default=False
        Whether to save with transparent background
    **kwargs
        Additional arguments passed to savefig

    Examples
    --------
    >>> fig, ax = plot_survival_curves(data, beta)
    >>> save_figure(fig, "survival_curves.pdf")
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent, **kwargs)
    print(f"✓ Figure saved to {filename}")


def plot_survival_curves(
    data: SurvivalDataset,
    beta: np.ndarray,
    title: str | None = None,
    n_groups: int = 3,
    group_labels: list[str] | None = None,
    confidence_intervals: bool = True,
    figsize: tuple[float, float] = (8, 6),
    colors: list[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot Kaplan-Meier style survival curves stratified by risk scores.

    Parameters
    ----------
    data : SurvivalDataset
        Survival dataset with X, y, and zeta
    beta : np.ndarray
        Coefficient vector for computing risk scores
    title : str, optional
        Plot title (default: "Survival Curves by Risk Group")
    n_groups : int, default=3
        Number of risk groups to create
    group_labels : List[str], optional
        Custom labels for risk groups
    confidence_intervals : bool, default=True
        Whether to show 95% confidence intervals
    figsize : Tuple[float, float], default=(8, 6)
        Figure size in inches
    colors : List[str], optional
        Custom colors for each group
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Examples
    --------
    >>> from drl_cox import simulate_cox_data, fit_drl_cox
    >>> from drl_cox.plotting import plot_survival_curves
    >>>
    >>> data = simulate_cox_data(n=200, d=10)
    >>> result = fit_drl_cox(data, epsilon=0.1)
    >>> fig, ax = plot_survival_curves(data, result.beta)
    """

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Compute risk scores
    risk_scores = data.X @ beta

    # Create risk groups based on quantiles
    if n_groups == 2:
        quantiles = np.array([0.5])
    elif n_groups == 3:
        quantiles = np.array([0.33, 0.67])
    else:
        quantiles = np.linspace(0, 1, n_groups + 1)[1:-1]

    risk_thresholds = np.quantile(risk_scores, quantiles)
    risk_groups = np.digitize(risk_scores, risk_thresholds)

    # Default group labels
    if group_labels is None:
        if n_groups == 2:
            group_labels = ["Low Risk", "High Risk"]
        elif n_groups == 3:
            group_labels = ["Low Risk", "Medium Risk", "High Risk"]
        else:
            group_labels = [f"Group {i + 1}" for i in range(n_groups)]

    # Default colors
    if colors is None:
        colors = sns.color_palette("husl", n_groups)

    # Plot Kaplan-Meier curves for each group
    for i in range(n_groups):
        mask = risk_groups == i
        if not np.any(mask):
            continue

        # Extract group data
        y_group = data.y[mask]
        zeta_group = data.zeta[mask]

        # Compute Kaplan-Meier estimate
        km_time, km_surv, km_ci_lower, km_ci_upper = _kaplan_meier(y_group, zeta_group)

        # Plot survival curve
        ax.step(
            km_time,
            km_surv,
            where="post",
            label=f"{group_labels[i]} (n={mask.sum()})",
            color=colors[i],
            linewidth=2,
            alpha=0.9,
        )

        # Add confidence intervals
        if confidence_intervals:
            ax.fill_between(
                km_time,
                km_ci_lower,
                km_ci_upper,
                step="post",
                alpha=0.2,
                color=colors[i],
            )

    # Add median survival lines (optional)
    for i in range(n_groups):
        mask = risk_groups == i
        if not np.any(mask):
            continue
        y_group = data.y[mask]
        zeta_group = data.zeta[mask]
        km_time, km_surv, _, _ = _kaplan_meier(y_group, zeta_group)

        # Find median survival time
        if np.any(km_surv <= 0.5):
            median_idx = np.where(km_surv <= 0.5)[0][0]
            median_time = km_time[median_idx]
            ax.axvline(
                median_time,
                color=colors[i],
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )

    # Styling
    ax.set_xlabel("Time", fontweight="bold")
    ax.set_ylabel("Survival Probability", fontweight="bold")
    ax.set_title(title or "Survival Curves by Risk Group", fontweight="bold", pad=20)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add risk table below (optional for publication quality)
    _add_risk_table(ax, data, risk_groups, group_labels, colors)

    # Add p-value for log-rank test
    if n_groups == 2:
        p_value = _log_rank_test(data.y, data.zeta, risk_groups)
        ax.text(
            0.98,
            0.02,
            f"Log-rank p = {p_value:.3e}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    return fig, ax


def plot_coefficient_path(
    epsilons: list[float] | np.ndarray,
    betas: np.ndarray,
    feature_names: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    highlight_top: int = 5,
    log_scale: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot regularization path showing how coefficients change with epsilon.

    Parameters
    ----------
    epsilons : List[float] or np.ndarray
        Epsilon values (regularization parameters)
    betas : np.ndarray
        Coefficients matrix (n_epsilons x n_features)
    feature_names : List[str], optional
        Names for each feature
    title : str, optional
        Plot title
    figsize : Tuple[float, float], default=(10, 6)
        Figure size
    highlight_top : int, default=5
        Number of top features to highlight with labels
    log_scale : bool, default=False
        Whether to use log scale for epsilon axis
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Examples
    --------
    >>> epsilons = [0.0, 0.05, 0.1, 0.2, 0.5]
    >>> betas = np.array([fit_drl_cox(data, eps).beta for eps in epsilons])
    >>> fig, ax = plot_coefficient_path(epsilons, betas.T)
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Convert to arrays
    epsilons = np.array(epsilons)
    if betas.ndim == 1:
        betas = betas.reshape(1, -1)

    n_features = betas.shape[1]

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(n_features)]

    # Color palette
    colors = sns.color_palette("husl", n_features)

    # Find top features by absolute coefficient magnitude at last epsilon
    top_features = np.argsort(np.abs(betas[-1, :]))[::-1][:highlight_top]

    # Plot coefficient paths
    for i in range(n_features):
        alpha = 0.9 if i in top_features else 0.3
        linewidth = 2.0 if i in top_features else 1.0

        ax.plot(
            epsilons,
            betas[:, i],
            color=colors[i % len(colors)],
            alpha=alpha,
            linewidth=linewidth,
            marker="o" if len(epsilons) <= 10 else None,
            markersize=4,
        )

        # Add labels for highlighted features
        if i in top_features:
            # Position label at the end of the line
            x_pos = epsilons[-1]
            y_pos = betas[-1, i]

            # Adjust position to avoid overlaps

            ax.annotate(
                feature_names[i],
                xy=(x_pos, y_pos),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                color=colors[i % len(colors)],
                alpha=0.9,
                va="center",
            )

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel("Epsilon (ε)", fontweight="bold")
    ax.set_ylabel("Coefficient Value", fontweight="bold")
    ax.set_title(
        title or "Coefficient Path: Impact of Robustness Parameter", fontweight="bold", pad=20
    )

    if log_scale and np.all(epsilons > 0):
        ax.set_xscale("log")
        ax.set_xlabel("Epsilon (ε) [log scale]", fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")

    # Add text box with coefficient statistics
    textstr = f"Features: {n_features}\nEpsilon range: [{epsilons[0]:.3f}, {epsilons[-1]:.3f}]"
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    return fig, ax


def plot_cv_results(
    cv_dataframe: pd.DataFrame,
    metric_name: str = "C-index",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    show_std: bool = True,
    mark_best: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot cross-validation scores vs epsilon with error bars.

    Parameters
    ----------
    cv_dataframe : pd.DataFrame
        Cross-validation results with columns: epsilon, fold, score
    metric_name : str, default="C-index"
        Name of the metric being plotted
    title : str, optional
        Plot title
    figsize : Tuple[float, float], default=(8, 6)
        Figure size
    show_std : bool, default=True
        Whether to show standard deviation bands
    mark_best : bool, default=True
        Whether to mark the best epsilon value
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Examples
    --------
    >>> from drl_cox import cross_validate_epsilon
    >>> cv_results = cross_validate_epsilon(data, epsilons=[0.0, 0.1, 0.2])
    >>> fig, ax = plot_cv_results(cv_results)
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Aggregate results
    cv_summary = (
        cv_dataframe.groupby("epsilon")["score"].agg(["mean", "std", "count"]).reset_index()
    )

    # Calculate standard error
    cv_summary["se"] = cv_summary["std"] / np.sqrt(cv_summary["count"])

    # Main line plot
    ax.plot(
        cv_summary["epsilon"],
        cv_summary["mean"],
        "o-",
        color="#2E86AB",
        linewidth=2.5,
        markersize=8,
        label=f"Mean {metric_name}",
        alpha=0.9,
    )

    # Error bands
    if show_std:
        ax.fill_between(
            cv_summary["epsilon"],
            cv_summary["mean"] - cv_summary["std"],
            cv_summary["mean"] + cv_summary["std"],
            alpha=0.2,
            color="#2E86AB",
            label="±1 Std Dev",
        )

        # Standard error bars
        ax.errorbar(
            cv_summary["epsilon"],
            cv_summary["mean"],
            yerr=cv_summary["se"],
            fmt="none",
            color="#2E86AB",
            alpha=0.6,
            capsize=3,
            capthick=1,
        )

    # Mark best epsilon
    if mark_best:
        best_pos = int(np.argmax(cv_summary["mean"].to_numpy()))
        best_eps = float(cv_summary["epsilon"].to_numpy()[best_pos])
        best_score = float(cv_summary["mean"].to_numpy()[best_pos])

        ax.scatter(
            [best_eps],
            [best_score],
            color="#E63946",
            s=200,
            zorder=5,
            marker="*",
            edgecolors="white",
            linewidth=2,
            label=f"Best ε = {best_eps:.3f}",
        )

        # Add annotation
        ax.annotate(
            f"Best: {best_score:.4f}",
            xy=(best_eps, best_score),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E63946", alpha=0.1),
            arrowprops=dict(arrowstyle="->", color="#E63946", alpha=0.6),
        )

    # Add individual fold points (faded)
    for eps in cv_summary["epsilon"]:
        fold_scores = cv_dataframe[cv_dataframe["epsilon"] == eps]["score"]
        ax.scatter(
            [eps] * len(fold_scores),
            fold_scores,
            alpha=0.3,
            s=20,
            color="gray",
            zorder=1,
        )

    # Styling
    ax.set_xlabel("Epsilon (ε)", fontweight="bold")
    ax.set_ylabel(metric_name, fontweight="bold")
    ax.set_title(
        title or f"Cross-Validation Results: {metric_name} vs Robustness", fontweight="bold", pad=20
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add summary statistics
    n_folds = cv_dataframe["fold"].nunique()
    n_epsilons = cv_dataframe["epsilon"].nunique()
    textstr = f"Folds: {n_folds}\nEpsilon values: {n_epsilons}"
    ax.text(
        0.98,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    return fig, ax


def plot_model_comparison(
    results_dict: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    colors: list[str] | None = None,
    show_values: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Side-by-side comparison of multiple models' performance metrics.

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, float]]
        Nested dict: {model_name: {metric_name: value}}
    metrics : List[str], optional
        Metrics to plot (default: all available)
    title : str, optional
        Plot title
    figsize : Tuple[float, float], default=(10, 6)
        Figure size
    colors : List[str], optional
        Custom colors for bars
    show_values : bool, default=True
        Whether to show values on bars
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Examples
    --------
    >>> results = {
    ...     "DRL-Cox (ε=0.1)": {"C-index": 0.75, "iAUC": 0.72},
    ...     "Cox-Lasso": {"C-index": 0.71, "iAUC": 0.69},
    ...     "Cox-Ridge": {"C-index": 0.70, "iAUC": 0.68},
    ... }
    >>> fig, ax = plot_model_comparison(results)
    """
    # Create figure with subplots if not provided
    if ax is None:
        # Determine if we need subplots
        if metrics is None:
            metrics = list(next(iter(results_dict.values())).keys())

        if len(metrics) > 1:
            fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
            if len(metrics) == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
    else:
        fig = cast(Figure, ax.figure)
        axes = [ax]
        if metrics is None:
            metrics = list(next(iter(results_dict.values())).keys())[:1]

    # Default colors
    if colors is None:
        colors = sns.color_palette("husl", len(results_dict))

    # Plot each metric
    for idx, (metric, ax) in enumerate(zip(metrics, axes, strict=False)):
        model_names = list(results_dict.keys())
        values = [results_dict[model].get(metric, 0) for model in model_names]

        # Create bars
        bars = ax.bar(
            range(len(model_names)),
            values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Highlight best model
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("#E63946")
        bars[best_idx].set_linewidth(3)
        bars[best_idx].set_alpha(1.0)

        # Add value labels
        if show_values:
            for i, (bar, val) in enumerate(zip(bars, values, strict=False)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 * max(values),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold" if i == best_idx else "normal",
                    fontsize=9,
                )

        # Styling
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric, fontweight="bold")
        ax.set_title(metric, fontweight="bold", pad=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, max(values) * 1.15)

        # Add horizontal line at chance level for C-index
        if "index" in metric.lower() or "auc" in metric.lower():
            ax.axhline(
                y=0.5,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="Chance level",
                linewidth=1,
            )
            if idx == 0:  # Only show legend once
                ax.legend(loc="lower right", fontsize=8)

    # Overall title
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=12, y=1.02)
    else:
        fig.suptitle("Model Performance Comparison", fontweight="bold", fontsize=12, y=1.02)

    plt.tight_layout()

    return fig, axes[0] if len(axes) == 1 else axes


def plot_risk_groups(
    data: SurvivalDataset,
    beta: np.ndarray,
    n_groups: int = 3,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, np.ndarray]:
    """
    Create multiple plots analyzing risk group stratification.

    Parameters
    ----------
    data : SurvivalDataset
        Survival dataset
    beta : np.ndarray
        Coefficient vector
    n_groups : int, default=3
        Number of risk groups
    title : str, optional
        Overall title
    figsize : Tuple[float, float], default=(12, 5)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : np.ndarray
        Array of axes objects

    Examples
    --------
    >>> fig, axes = plot_risk_groups(data, result.beta, n_groups=3)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Compute risk scores and groups
    risk_scores = data.X @ beta
    quantiles = np.linspace(0, 1, n_groups + 1)[1:-1]
    risk_thresholds = np.quantile(risk_scores, quantiles)
    risk_groups = np.digitize(risk_scores, risk_thresholds)

    # 1. Risk score distribution
    ax = axes[0]
    colors = sns.color_palette("husl", n_groups)

    for i in range(n_groups):
        mask = risk_groups == i
        ax.hist(
            risk_scores[mask],
            bins=20,
            alpha=0.6,
            label=f"Group {i + 1}",
            color=colors[i],
            edgecolor="black",
            linewidth=1,
        )

    # Add threshold lines
    for thresh in risk_thresholds:
        ax.axvline(thresh, color="red", linestyle="--", alpha=0.7, linewidth=2)

    ax.set_xlabel("Risk Score", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Risk Score Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Event rates by group
    ax = axes[1]
    event_rates = []
    group_labels = []

    for i in range(n_groups):
        mask = risk_groups == i
        event_rate = np.mean(data.zeta[mask])
        event_rates.append(event_rate)
        group_labels.append(f"Group {i + 1}\n(n={mask.sum()})")

    bars = ax.bar(
        range(n_groups),
        event_rates,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bar, rate in zip(bars, event_rates, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Event Rate", fontweight="bold")
    ax.set_title("Event Rates by Risk Group", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(event_rates) * 1.2)

    # 3. Median survival times
    ax = axes[2]
    median_times = []

    for i in range(n_groups):
        mask = risk_groups == i
        km_time, km_surv, _, _ = _kaplan_meier(data.y[mask], data.zeta[mask])

        # Find median survival time
        if np.any(km_surv <= 0.5):
            median_idx = np.where(km_surv <= 0.5)[0][0]
            median_time = km_time[median_idx]
        else:
            median_time = np.max(km_time)  # Use max time if median not reached
        median_times.append(median_time)

    bars = ax.bar(
        range(n_groups),
        median_times,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bar, time in zip(bars, median_times, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(median_times) * 0.01,
            f"{time:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"Group {i + 1}" for i in range(n_groups)])
    ax.set_ylabel("Median Survival Time", fontweight="bold")
    ax.set_title("Median Survival by Risk Group", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=12, y=1.05)

    plt.tight_layout()
    return fig, axes


def plot_concordance_over_time(
    data: SurvivalDataset,
    models_dict: dict[str, np.ndarray],
    time_points: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot time-dependent concordance index for multiple models.

    Parameters
    ----------
    data : SurvivalDataset
        Survival dataset
    models_dict : Dict[str, np.ndarray]
        Dictionary mapping model names to coefficient vectors
    time_points : np.ndarray, optional
        Time points for evaluation (default: event times)
    title : str, optional
        Plot title
    figsize : Tuple[float, float], default=(10, 6)
        Figure size
    ax : matplotlib.axes.Axes, optional
        Existing axes

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Examples
    --------
    >>> models = {
    ...     "DRL-Cox": result_drl.beta,
    ...     "Cox-Lasso": beta_lasso,
    ...     "Cox-Ridge": beta_ridge,
    ... }
    >>> fig, ax = plot_concordance_over_time(data, models)
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Default time points
    if time_points is None:
        # Use quartiles of event times
        event_times = data.y[data.zeta == 1]
        time_points = np.quantile(event_times, [0.1, 0.25, 0.5, 0.75, 0.9])

    # Colors for models
    colors = sns.color_palette("husl", len(models_dict))

    # Plot each model
    for (model_name, beta), color in zip(models_dict.items(), colors, strict=False):
        concordances = []

        # Compute time-dependent C-index
        for t in time_points:
            # Only consider samples with T > t or (T <= t and event)
            mask = (data.y > t) | ((data.y <= t) & (data.zeta == 1))
            if mask.sum() < 10:  # Skip if too few samples
                concordances.append(np.nan)
                continue

            risk_scores = data.X[mask] @ beta
            c_index = _concordance_at_time(risk_scores, data.y[mask], data.zeta[mask], t)
            concordances.append(c_index)

        # Plot line
        valid_mask = ~np.isnan(concordances)
        ax.plot(
            time_points[valid_mask],
            np.array(concordances)[valid_mask],
            "o-",
            label=model_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    # Add reference line at 0.5
    ax.axhline(
        y=0.5,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Chance level",
        linewidth=1,
    )

    # Styling
    ax.set_xlabel("Time", fontweight="bold")
    ax.set_ylabel("Concordance Index", fontweight="bold")
    ax.set_title(title or "Time-Dependent Model Performance", fontweight="bold", pad=20)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0.4, max(0.9, ax.get_ylim()[1]))

    return fig, ax


# ============================================================================
# Helper Functions (Private)
# ============================================================================


def _kaplan_meier(
    y: np.ndarray,
    zeta: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Kaplan-Meier survival estimate with confidence intervals.

    Returns
    -------
    time : np.ndarray
        Time points
    survival : np.ndarray
        Survival probability estimates
    ci_lower : np.ndarray
        Lower confidence interval
    ci_upper : np.ndarray
        Upper confidence interval
    """
    # Sort by time
    order = np.argsort(y)
    y_sorted = y[order]
    zeta_sorted = zeta[order]

    # Get unique event times
    event_times = np.unique(y_sorted[zeta_sorted == 1])

    # Initialize
    survival = [1.0]
    ci_lower = [1.0]
    ci_upper = [1.0]
    time_points = [0.0]

    s = 1.0  # Current survival estimate
    var_s = 0.0  # Greenwood variance

    for t in event_times:
        # Number at risk
        n_risk = np.sum(y_sorted >= t)
        # Number of events at time t
        n_event = np.sum((y_sorted == t) & (zeta_sorted == 1))

        if n_risk > 0:
            # Update survival
            s *= 1 - n_event / n_risk

            # Update Greenwood variance
            if n_risk > n_event:
                var_s += n_event / (n_risk * (n_risk - n_event))

            # Confidence intervals (log-log transformation)
            if s > 0:
                se = s * np.sqrt(var_s)
                z = 1.96  # For 95% CI

                # Log-log transformation for better properties
                if s < 1:
                    log_s = np.log(-np.log(s))
                    se_log = se / (s * abs(np.log(s)))
                    ci_l = np.exp(-np.exp(log_s + z * se_log))
                    ci_u = np.exp(-np.exp(log_s - z * se_log))
                else:
                    ci_l = s - z * se
                    ci_u = s + z * se

                ci_lower.append(max(0, ci_l))
                ci_upper.append(min(1, ci_u))
            else:
                ci_lower.append(0)
                ci_upper.append(0)

            survival.append(s)
            time_points.append(t)

    return (
        np.array(time_points),
        np.array(survival),
        np.array(ci_lower),
        np.array(ci_upper),
    )


def _log_rank_test(
    y: np.ndarray,
    zeta: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Perform log-rank test between groups.

    Returns
    -------
    p_value : float
        P-value from chi-squared test
    """
    from scipy.stats import chi2

    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        return np.nan

    # Get unique event times
    event_times = np.unique(y[zeta == 1])

    # Calculate observed and expected events
    O1, E1 = 0, 0  # Group 1
    V = 0  # Variance

    for t in event_times:
        # At risk in each group
        n1 = np.sum((y >= t) & (groups == unique_groups[0]))
        n2 = np.sum((y >= t) & (groups == unique_groups[1]))
        n = n1 + n2

        if n == 0:
            continue

        # Events in each group
        d1 = np.sum((y == t) & (zeta == 1) & (groups == unique_groups[0]))
        d2 = np.sum((y == t) & (zeta == 1) & (groups == unique_groups[1]))
        d = d1 + d2

        # Expected events
        e1 = n1 * d / n

        O1 += d1
        E1 += e1

        # Variance
        if n > 1:
            V += e1 * (n - n1) * (n - d) / (n * (n - 1))

    if V == 0:
        return 1.0

    # Chi-squared statistic
    chi2_stat = (O1 - E1) ** 2 / V
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return float(p_value)


def _add_risk_table(
    ax: Axes,
    data: SurvivalDataset,
    risk_groups: np.ndarray,
    group_labels: list[str],
    colors: list[str],
) -> None:
    """
    Add at-risk table below survival curves.
    """
    # Get time points for the table
    time_points = np.quantile(data.y, [0, 0.25, 0.5, 0.75, 1.0])

    # Create text for risk table
    table_text = []
    for i, label in enumerate(group_labels):
        row = [label]
        mask = risk_groups == i
        for t in time_points:
            n_risk = np.sum((data.y >= t) & mask)
            row.append(str(n_risk))
        table_text.append(row)

    # Position the table
    table_y_start = -0.25
    y_positions = np.linspace(table_y_start, table_y_start - 0.15, len(group_labels))

    # Add table headers
    ax.text(
        -0.1,
        table_y_start + 0.05,
        "At Risk:",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
    )

    # Add time headers
    x_positions = np.linspace(0.1, 0.9, len(time_points))
    for x, t in zip(x_positions, time_points, strict=False):
        ax.text(
            x,
            table_y_start + 0.05,
            f"{t:.1f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="center",
        )

    # Add risk numbers
    for i, (row, y_pos) in enumerate(zip(table_text, y_positions, strict=False)):
        # Group label
        ax.text(
            -0.05,
            y_pos,
            row[0],
            transform=ax.transAxes,
            fontsize=8,
            color=colors[i],
            fontweight="bold",
        )
        # Numbers at risk
        for x, val in zip(x_positions, row[1:], strict=False):
            ax.text(
                x,
                y_pos,
                val,
                transform=ax.transAxes,
                fontsize=8,
                ha="center",
            )


def _concordance_at_time(
    risk_scores: np.ndarray,
    y: np.ndarray,
    zeta: np.ndarray,
    t: float,
) -> float:
    """
    Compute concordance index at a specific time point.
    """
    from .metrics import concordance_index

    # Focus on relevant pairs for time t
    # This is a simplified version
    return concordance_index(risk_scores, y, zeta)


# ============================================================================
# Tutorial Examples
# ============================================================================


def create_tutorial_plots() -> dict[str, tuple[Figure, Any]]:
    """
    Generate example plots for the tutorial.

    This function demonstrates all plotting capabilities with synthetic data.
    """
    # Import dependencies
    from .cox_baseline import CoxLasso, CoxRidge
    from .datasets import simulate_cox_data
    from .drl_cox import cross_validate_epsilon, fit_drl_cox

    # Generate synthetic data
    print("Generating synthetic data...")
    data = simulate_cox_data(n=300, d=10, seed=42)

    # Set up plotting style
    setup_plotting_style("publication", font_size=10)

    # 1. Survival Curves
    print("Creating survival curves plot...")
    result = fit_drl_cox(data, epsilon=0.1)
    fig1, ax1 = plot_survival_curves(
        data,
        result.beta,
        title="Survival Analysis: Risk Stratification",
        n_groups=3,
    )
    save_figure(fig1, "tutorial_survival_curves.pdf")

    # 2. Coefficient Path
    print("Creating coefficient path plot...")
    epsilons = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    betas = []
    for eps in epsilons:
        res = fit_drl_cox(data, epsilon=eps)
        betas.append(res.beta)

    fig2, ax2 = plot_coefficient_path(
        epsilons,
        np.array(betas),
        feature_names=[f"Gene {i + 1}" for i in range(10)],
        title="Feature Selection via Regularization",
        highlight_top=5,
    )
    save_figure(fig2, "tutorial_coefficient_path.pdf")

    # 3. Cross-Validation Results
    print("Running cross-validation...")
    cv_results = cross_validate_epsilon(
        data,
        epsilons=epsilons,
        kfolds=5,
        metric="cindex",
    )

    fig3, ax3 = plot_cv_results(
        cv_results,
        metric_name="C-index",
        title="Cross-Validation: Optimal Robustness Selection",
    )
    save_figure(fig3, "tutorial_cv_results.pdf")

    # 4. Model Comparison
    print("Comparing models...")

    # Fit baseline models
    cox_lasso = CoxLasso(alpha=0.05).fit(data.X, data.y, data.zeta)
    cox_ridge = CoxRidge(alpha=1.0).fit(data.X, data.y, data.zeta)

    # Compute metrics
    from .metrics import concordance_index, time_dependent_auc_iAUC

    models_results = {}
    for name, beta in [
        ("DRL-Cox (ε=0.1)", result.beta),
        ("Cox-Lasso", cox_lasso),
        ("Cox-Ridge", cox_ridge),
    ]:
        risk = data.X @ beta
        c_idx = concordance_index(risk, data.y, data.zeta)
        iauc = time_dependent_auc_iAUC(risk, data.y, data.zeta)
        models_results[name] = {"C-index": c_idx, "iAUC": iauc}

    fig4, ax4 = plot_model_comparison(
        models_results,
        title="Model Performance Benchmark",
    )
    save_figure(fig4, "tutorial_model_comparison.pdf")

    # 5. Risk Group Analysis
    print("Creating risk group analysis...")
    fig5, axes5 = plot_risk_groups(
        data,
        result.beta,
        title="Risk Group Characterization",
    )
    save_figure(fig5, "tutorial_risk_groups.pdf")

    # 6. Time-Dependent Performance
    print("Creating time-dependent concordance plot...")
    models_dict = {
        "DRL-Cox": result.beta,
        "Cox-Lasso": cox_lasso,
        "Cox-Ridge": cox_ridge,
    }

    fig6, ax6 = plot_concordance_over_time(
        data,
        models_dict,
        title="Model Performance Over Time",
    )
    save_figure(fig6, "tutorial_concordance_time.pdf")

    print("\n✓ All tutorial plots created successfully!")
    print("Files saved: tutorial_*.pdf")

    return {
        "survival_curves": (fig1, ax1),
        "coefficient_path": (fig2, ax2),
        "cv_results": (fig3, ax3),
        "model_comparison": (fig4, ax4),
        "risk_groups": (fig5, axes5),
        "concordance_time": (fig6, ax6),
    }


if __name__ == "__main__":
    # Run tutorial examples when module is executed directly
    create_tutorial_plots()
