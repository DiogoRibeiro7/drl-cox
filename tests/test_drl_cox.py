"""Tests for DRL-Cox solver."""

from __future__ import annotations

import numpy as np
import pytest

from drl_cox import SurvivalDataset, fit_drl_cox, simulate_cox_data


def test_simulate_cox_data():
    """Test synthetic data generation."""
    data = simulate_cox_data(n=50, d=5, seed=42)
    assert data.X.shape == (50, 5)
    assert data.y.shape == (50,)
    assert data.zeta.shape == (50,)
    assert np.all(data.y > 0)
    assert np.all((data.zeta == 0) | (data.zeta == 1))


def test_fit_drl_cox_basic():
    """Test basic DRL-Cox fitting."""
    data = simulate_cox_data(n=30, d=3, seed=0)
    result = fit_drl_cox(
        data,
        epsilon=0.1,
        p=2.0,
        gamma=2,
        solver="CLARABEL",
        solver_opts={"max_iter": 100},
    )
    assert result.beta.shape == (3,)
    assert isinstance(result.alpha, float)
    assert result.status == "optimal"


def test_survival_dataset_validation():
    """Test SurvivalDataset validation."""
    with pytest.raises(ValueError):
        SurvivalDataset(
            X=np.array([[1, 2], [3, 4]]),
            y=np.array([1.0, -1.0]),  # negative duration
            zeta=np.array([1, 1]),
        )

    with pytest.raises(ValueError):
        SurvivalDataset(
            X=np.array([[1, 2], [3, 4]]),
            y=np.array([1.0, 2.0]),
            zeta=np.array([1, 2]),  # invalid event indicator
        )


def test_epsilon_zero():
    """Test DRL-Cox with epsilon=0 (no robustness)."""
    data = simulate_cox_data(n=40, d=4, seed=123)
    result = fit_drl_cox(data, epsilon=0.0, gamma=2)
    assert result.beta.shape == (4,)
    assert result.status == "optimal"


def test_formulations_reach_the_same_optimum():
    """The event-only formulation drops zero-weight rows without changing the solution."""
    data = simulate_cox_data(n=40, d=3, seed=5)
    full = fit_drl_cox(data, epsilon=0.05, gamma=2, formulation="full")
    events = fit_drl_cox(data, epsilon=0.05, gamma=2, formulation="events")

    assert full.info["formulation"] == "full"
    assert events.info["formulation"] == "events"
    np.testing.assert_allclose(events.beta, full.beta, atol=1e-4)
    assert events.alpha == pytest.approx(full.alpha, abs=1e-4)
    assert events.objective_value == pytest.approx(full.objective_value, abs=1e-6)
    np.testing.assert_allclose(events.s, full.s, atol=1e-4)


def test_slack_values_follow_input_rows():
    """Slack values are aligned with the input rows and vanish for censored rows."""
    data = simulate_cox_data(n=40, d=3, seed=6)
    result = fit_drl_cox(data, epsilon=0.05, gamma=2)

    assert result.s.shape == (40,)
    assert np.all(result.s[data.zeta == 0] == 0.0)
    assert np.all(result.s[data.zeta == 1] > 0.0)


def test_auto_formulation_follows_the_solver():
    """Clarabel gets the full formulation, SCS the event-only one."""
    data = simulate_cox_data(n=30, d=3, seed=7)
    clarabel = fit_drl_cox(data, epsilon=0.05, gamma=2, solver="CLARABEL")
    scs = fit_drl_cox(data, epsilon=0.05, gamma=2, solver="SCS", solver_opts={"max_iters": 500})

    assert clarabel.info["formulation"] == "full"
    assert scs.info["formulation"] == "events"


def test_invalid_formulation_is_rejected():
    data = simulate_cox_data(n=20, d=2, seed=8)
    with pytest.raises(ValueError, match="formulation"):
        fit_drl_cox(data, epsilon=0.05, gamma=2, formulation="fast")
