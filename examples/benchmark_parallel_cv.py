"""
Benchmark script for parallel cross-validation performance.

This script demonstrates the performance improvements from parallelization
and provides visualization of the results.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import the optimized CV function
# Note: In production, this would be imported from drl_cox
# For now, we'll define it inline or import from the optimization module
from drl_cox import (
    cross_validate_epsilon,
    simulate_cox_data,
)

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def run_comprehensive_benchmark():
    """
    Run comprehensive benchmarks across different configurations.
    """
    print("=" * 80)
    print("COMPREHENSIVE PARALLEL CV BENCHMARK")
    print("=" * 80)
    print()

    # Configuration
    configs = [
        # (n, d, kfolds, description)
        (500, 20, 10, "Standard (n=500, d=20, k=10)"),
        (1000, 30, 5, "Large dataset (n=1000, d=30, k=5)"),
        (200, 10, 10, "Small dataset (n=200, d=10, k=10)"),
    ]

    epsilons = [0.0, 0.05, 0.1, 0.2]
    n_jobs_list = [1, 2, 4, -1]

    all_results = []

    for n, d, kfolds, description in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {description}")
        print(f"{'=' * 80}")
        print(f"Dataset: n={n}, d={d}")
        print(f"CV: {len(epsilons)} epsilons × {kfolds} folds = {len(epsilons) * kfolds} tasks")
        print()

        # Generate data
        print("Generating synthetic data...")
        data = simulate_cox_data(n=n, d=d, seed=42, baseline_hazard=0.02, censor_rate=0.35)
        print("✓ Data generated")
        print()

        # Benchmark each n_jobs setting
        for n_jobs in n_jobs_list:
            print(f"Testing n_jobs={n_jobs}...")

            start_time = time.time()

            try:
                results = cross_validate_epsilon(
                    data,
                    epsilons=epsilons,
                    p=2.0,
                    gamma=3,
                    kfolds=kfolds,
                    metric="cindex",
                    solver="CLARABEL",
                    solver_opts={"max_iter": 200},
                    n_jobs=n_jobs,
                    verbose=False,
                )

                elapsed_time = time.time() - start_time
                n_failures = results["score"].isna().sum()
                mean_score = results["score"].mean()

                all_results.append(
                    {
                        "config": description,
                        "n": n,
                        "d": d,
                        "kfolds": kfolds,
                        "n_jobs": n_jobs,
                        "time_seconds": elapsed_time,
                        "mean_score": mean_score,
                        "n_failures": n_failures,
                        "total_tasks": len(epsilons) * kfolds,
                    }
                )

                print(f"   ✓ Completed in {elapsed_time:.2f}s")
                print(f"     Mean C-index: {mean_score:.4f}")
                if n_failures > 0:
                    print(f"     ⚠️  Failures: {n_failures}")
                print()

            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
                print()
                continue

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate speedup for each configuration
    for config in results_df["config"].unique():
        mask = results_df["config"] == config
        baseline_time = results_df[mask & (results_df["n_jobs"] == 1)]["time_seconds"].values
        if len(baseline_time) > 0:
            baseline_time = baseline_time[0]
            results_df.loc[mask, "speedup"] = baseline_time / results_df.loc[mask, "time_seconds"]
            results_df.loc[mask, "efficiency"] = (
                results_df.loc[mask, "speedup"] / results_df.loc[mask, "n_jobs"].abs()
            )

    return results_df


def visualize_benchmark_results(results_df: pd.DataFrame):
    """
    Create visualizations of benchmark results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Parallel Cross-Validation Performance Benchmark", fontsize=16, fontweight="bold")

    configs = results_df["config"].unique()
    colors = sns.color_palette("husl", len(configs))

    # 1. Execution time vs n_jobs
    ax = axes[0, 0]
    for i, config in enumerate(configs):
        data = results_df[results_df["config"] == config]
        ax.plot(
            data["n_jobs"],
            data["time_seconds"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=config,
            color=colors[i],
        )
    ax.set_xlabel("Number of Jobs")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Execution Time vs Parallelization")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(results_df["n_jobs"].unique())

    # 2. Speedup vs n_jobs
    ax = axes[0, 1]
    for i, config in enumerate(configs):
        data = results_df[results_df["config"] == config]
        ax.plot(
            data["n_jobs"],
            data["speedup"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=config,
            color=colors[i],
        )
    # Add ideal speedup line
    n_jobs_range = results_df["n_jobs"].unique()
    n_jobs_range = n_jobs_range[n_jobs_range > 0]
    ax.plot(n_jobs_range, n_jobs_range, "k--", linewidth=2, alpha=0.5, label="Ideal (linear)")
    ax.set_xlabel("Number of Jobs")
    ax.set_ylabel("Speedup Factor")
    ax.set_title("Speedup vs Parallelization")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(results_df["n_jobs"].unique())

    # 3. Efficiency vs n_jobs
    ax = axes[0, 2]
    for i, config in enumerate(configs):
        data = results_df[results_df["config"] == config]
        ax.plot(
            data["n_jobs"],
            data["efficiency"] * 100,
            marker="^",
            linewidth=2,
            markersize=8,
            label=config,
            color=colors[i],
        )
    ax.axhline(100, color="black", linestyle="--", linewidth=2, alpha=0.5)
    ax.set_xlabel("Number of Jobs")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Parallel Efficiency")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(results_df["n_jobs"].unique())

    # 4. Bar chart: Time comparison
    ax = axes[1, 0]
    width = 0.2
    x = np.arange(len(configs))
    n_jobs_vals = sorted([j for j in results_df["n_jobs"].unique() if j > 0])
    for i, n_jobs in enumerate(n_jobs_vals):
        times = [
            results_df[(results_df["config"] == cfg) & (results_df["n_jobs"] == n_jobs)][
                "time_seconds"
            ].values[0]
            for cfg in configs
        ]
        ax.bar(x + i * width, times, width, label=f"n_jobs={n_jobs}", alpha=0.8)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Execution Time by Configuration")
    ax.set_xticks(x + width * (len(n_jobs_vals) - 1) / 2)
    ax.set_xticklabels([c.split("(")[0].strip() for c in configs], rotation=20, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 5. Heatmap: Speedup
    ax = axes[1, 1]
    pivot = results_df.pivot_table(
        values="speedup", index="n_jobs", columns="config", aggfunc="mean"
    )
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([c.split("(")[0].strip() for c in pivot.columns], rotation=20, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Number of Jobs")
    ax.set_title("Speedup Heatmap")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}x",
                    ha="center",
                    va="center",
                    color="black" if val < 3 else "white",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.colorbar(im, ax=ax, label="Speedup Factor")

    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis("off")

    summary_data = []
    for config in configs:
        data = results_df[results_df["config"] == config]
        seq_time = data[data["n_jobs"] == 1]["time_seconds"].values[0]
        best_parallel = data[data["n_jobs"] > 1]["time_seconds"].min()
        best_speedup = data["speedup"].max()
        best_n_jobs = data.loc[data["speedup"].idxmax(), "n_jobs"]

        summary_data.append(
            [
                config.split("(")[0].strip(),
                f"{seq_time:.1f}s",
                f"{best_parallel:.1f}s",
                f"{best_speedup:.1f}x",
                f"{best_n_jobs}",
            ]
        )

    table = ax.table(
        cellText=summary_data,
        colLabels=["Config", "Sequential", "Best Parallel", "Max Speedup", "Best n_jobs"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style rows
    for i in range(1, len(summary_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax.set_title("Summary Statistics", fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def print_summary_report(results_df: pd.DataFrame):
    """
    Print a detailed summary report.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 80)

    for config in results_df["config"].unique():
        data = results_df[results_df["config"] == config]

        print(f"\n{config}")
        print("-" * 80)

        # Sequential baseline
        seq_data = data[data["n_jobs"] == 1].iloc[0]
        print("Sequential (n_jobs=1):")
        print(f"  Time: {seq_data['time_seconds']:.2f}s")
        print(f"  Tasks: {seq_data['total_tasks']}")
        print(f"  Time per task: {seq_data['time_seconds'] / seq_data['total_tasks']:.2f}s")
        print(f"  Mean C-index: {seq_data['mean_score']:.4f}")

        # Best parallel result
        best_idx = data["speedup"].idxmax()
        best_data = data.loc[best_idx]
        print(f"\nBest Parallel (n_jobs={int(best_data['n_jobs'])}):")
        print(f"  Time: {best_data['time_seconds']:.2f}s")
        print(f"  Speedup: {best_data['speedup']:.2f}x")
        print(f"  Efficiency: {best_data['efficiency'] * 100:.1f}%")
        print(f"  Time saved: {seq_data['time_seconds'] - best_data['time_seconds']:.2f}s")

        # All parallel results
        print("\nAll Results:")
        display_cols = ["n_jobs", "time_seconds", "speedup", "efficiency"]
        display_data = data[display_cols].copy()
        display_data["efficiency"] = display_data["efficiency"] * 100
        print(display_data.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Overall statistics
    max_speedup = results_df["speedup"].max()
    max_speedup_config = results_df.loc[results_df["speedup"].idxmax(), "config"]
    max_speedup_njobs = results_df.loc[results_df["speedup"].idxmax(), "n_jobs"]

    print(f"\n✓ Maximum speedup achieved: {max_speedup:.2f}x")
    print(f"  Configuration: {max_speedup_config}")
    print(f"  n_jobs: {int(max_speedup_njobs)}")

    # Average speedup by n_jobs
    print("\n✓ Average speedup by parallelization level:")
    for n_jobs in sorted(results_df["n_jobs"].unique()):
        if n_jobs > 1:
            avg_speedup = results_df[results_df["n_jobs"] == n_jobs]["speedup"].mean()
            print(f"  n_jobs={int(n_jobs):2d}: {avg_speedup:.2f}x average speedup")

    # Efficiency analysis
    print("\n✓ Parallel efficiency:")
    for n_jobs in sorted(results_df["n_jobs"].unique()):
        if n_jobs > 1:
            avg_eff = results_df[results_df["n_jobs"] == n_jobs]["efficiency"].mean() * 100
            print(f"  n_jobs={int(n_jobs):2d}: {avg_eff:.1f}% average efficiency")

    print()


def save_results(results_df: pd.DataFrame, fig, output_dir: str = "benchmark_results"):
    """
    Save benchmark results and plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save DataFrame
    csv_path = output_path / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")

    # Save plot
    plot_path = output_path / "benchmark_plots.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Plots saved to {plot_path}")

    # Save summary report
    report_path = output_path / "benchmark_report.txt"
    with open(report_path, "w") as f:
        # Redirect print to file
        import sys

        old_stdout = sys.stdout
        sys.stdout = f
        print_summary_report(results_df)
        sys.stdout = old_stdout
    print(f"✓ Report saved to {report_path}")


def quick_benchmark():
    """
    Run a quick benchmark with standard settings.
    """
    print("=" * 80)
    print("QUICK BENCHMARK: Standard Configuration")
    print("=" * 80)
    print()

    # Standard configuration
    n, d, kfolds = 500, 20, 10
    epsilons = [0.0, 0.1, 0.2, 0.3]
    n_jobs_list = [1, 2, 4, -1]

    print(f"Dataset: n={n}, d={d}")
    print(f"CV: {len(epsilons)} epsilons × {kfolds} folds = {len(epsilons) * kfolds} tasks")
    print()

    # Generate data
    print("Generating synthetic data...")
    data = simulate_cox_data(n=n, d=d, seed=42, baseline_hazard=0.02, censor_rate=0.35)
    print(f"✓ Data generated: {data.X.shape[0]} samples, {data.X.shape[1]} features")
    print()

    results = []

    for n_jobs in n_jobs_list:
        print(f"Testing n_jobs={n_jobs}...")

        start_time = time.time()

        cv_results = cross_validate_epsilon(
            data,
            epsilons=epsilons,
            p=2.0,
            gamma=3,
            kfolds=kfolds,
            metric="cindex",
            solver="CLARABEL",
            solver_opts={"max_iter": 200},
            n_jobs=n_jobs,
            verbose=True,
        )

        elapsed_time = time.time() - start_time
        n_failures = cv_results["score"].isna().sum()
        mean_score = cv_results["score"].mean()

        results.append(
            {
                "n_jobs": n_jobs,
                "time_seconds": elapsed_time,
                "mean_score": mean_score,
                "n_failures": n_failures,
            }
        )

        print(f"✓ Completed in {elapsed_time:.2f}s")
        print(f"  Mean C-index: {mean_score:.4f}")
        print()

    # Calculate speedup
    results_df = pd.DataFrame(results)
    baseline_time = results_df[results_df["n_jobs"] == 1]["time_seconds"].values[0]
    results_df["speedup"] = baseline_time / results_df["time_seconds"]
    results_df["efficiency"] = results_df["speedup"] / results_df["n_jobs"].abs()

    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    # Summary
    best_idx = results_df["speedup"].idxmax()
    best = results_df.loc[best_idx]
    print(f"Best speedup: {best['speedup']:.2f}x with n_jobs={int(best['n_jobs'])}")
    saved = baseline_time - best["time_seconds"]
    pct_faster = (1 - best["time_seconds"] / baseline_time) * 100
    print(f"Time saved: {saved:.2f}s ({pct_faster:.1f}% faster)")
    print()

    return results_df


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick benchmark
        results_df = quick_benchmark()

    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        # Comprehensive benchmark
        results_df = run_comprehensive_benchmark()

        # Print summary report
        print_summary_report(results_df)

        # Create visualizations
        print("\nGenerating visualizations...")
        fig = visualize_benchmark_results(results_df)

        # Save results
        print("\nSaving results...")
        save_results(results_df, fig)

        print("\n✓ Benchmark complete!")
        plt.show()

    else:
        # Default: run quick benchmark
        print("Usage:")
        print("  python benchmark_parallel_cv.py          # Quick benchmark")
        print("  python benchmark_parallel_cv.py quick    # Quick benchmark")
        print("  python benchmark_parallel_cv.py full     # Comprehensive benchmark")
        print()
        print("Running quick benchmark...\n")
        results_df = quick_benchmark()
