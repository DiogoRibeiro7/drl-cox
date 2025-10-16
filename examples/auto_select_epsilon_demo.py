"""
Demonstration of adaptive epsilon selection using Bayesian optimisation.

Run with:
    poetry run python examples/auto_select_epsilon_demo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from drl_cox import simulate_cox_data
from drl_cox.cross_validation import auto_select_epsilon


def main() -> None:
    data = simulate_cox_data(n=150, d=12, seed=7)
    result = auto_select_epsilon(
        data,
        metric="cindex",
        n_calls=12,
        n_initial_points=4,
        random_state=7,
    )

    print(f"Best epsilon: {result.best_epsilon:.4f}")
    print(f"Estimated score: {result.best_score:.4f}")
    low, high = result.confidence_interval
    print(f"95% credible interval: [{low:.4f}, {high:.4f}]")
    print(result.trials.head())

    if result.figure is not None:
        plt.show()


if __name__ == "__main__":
    main()
