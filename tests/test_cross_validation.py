from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from drl_cox import cross_validate_epsilon, simulate_cox_data
from drl_cox.cross_validation import auto_select_epsilon


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_auto_select_epsilon_basic():
    data = simulate_cox_data(n=60, d=6, seed=123)
    result = auto_select_epsilon(
        data,
        metric="cindex",
        n_calls=6,
        n_initial_points=2,
        random_state=0,
        visualize=False,
        solver="SCS",
        solver_opts={"max_iters": 100},
    )

    assert np.isfinite(result.best_epsilon)
    assert result.best_epsilon > 0
    low, high = result.confidence_interval
    assert low <= result.best_epsilon <= high
    assert len(result.trials) == 6
    assert np.isfinite(result.best_score)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_auto_select_respects_bounds_and_visualizes():
    data = simulate_cox_data(n=40, d=4, seed=21)
    bounds = (0.02, 0.05)
    result = auto_select_epsilon(
        data,
        metric="cindex",
        n_calls=5,
        n_initial_points=2,
        random_state=1,
        visualize=True,
        epsilon_bounds=bounds,
        solver="SCS",
        solver_opts={"max_iters": 80},
    )
    assert bounds[0] <= result.best_epsilon <= bounds[1]
    assert result.figure is not None
    # ensure trials are within bounds
    assert result.trials["epsilon"].between(bounds[0], bounds[1]).all()
    plt.close(result.figure)


def test_auto_select_input_validation(simple_dataset):
    with pytest.raises(ValueError):
        auto_select_epsilon(simple_dataset, metric="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        auto_select_epsilon(simple_dataset, n_calls=0)
    with pytest.raises(ValueError):
        auto_select_epsilon(simple_dataset, n_calls=4, n_initial_points=4)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cross_validate_epsilon_integration(simple_dataset):
    epsilons = [0.0, 0.1]
    df = cross_validate_epsilon(
        simple_dataset,
        epsilons=epsilons,
        kfolds=3,
        metric="cindex",
        solver="SCS",
        solver_opts={"max_iters": 80},
    )
    assert set(df.columns) == {"epsilon", "fold", "score"}
    assert sorted(df["epsilon"].unique()) == epsilons
    assert len(df) == len(epsilons) * 3


@pytest.fixture
def simple_dataset():
    return simulate_cox_data(n=40, d=4, seed=99)
