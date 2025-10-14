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
        solver="ECOS",
        solver_opts={"max_iters": 100}
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
            zeta=np.array([1, 1])
        )
    
    with pytest.raises(ValueError):
        SurvivalDataset(
            X=np.array([[1, 2], [3, 4]]),
            y=np.array([1.0, 2.0]),
            zeta=np.array([1, 2])  # invalid event indicator
        )


def test_epsilon_zero():
    """Test DRL-Cox with epsilon=0 (no robustness)."""
    data = simulate_cox_data(n=40, d=4, seed=123)
    result = fit_drl_cox(data, epsilon=0.0, gamma=2)
    assert result.beta.shape == (4,)
    assert result.status == "optimal"
