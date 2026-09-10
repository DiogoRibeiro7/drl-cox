"""Tests for survival metrics."""

from __future__ import annotations

import numpy as np

from drl_cox import concordance_index, time_dependent_auc_iAUC


def test_concordance_index_perfect():
    """Test C-index with perfect predictions."""
    # Higher risk must correspond to an earlier event for perfect concordance.
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    zeta = np.array([1, 1, 1, 1])
    c = concordance_index(risk, y, zeta)
    assert c == 1.0


def test_concordance_index_random():
    """Test C-index with random predictions."""
    np.random.seed(42)
    risk = np.random.randn(50)
    y = np.random.exponential(2.0, 50)
    zeta = np.random.binomial(1, 0.7, 50)
    c = concordance_index(risk, y, zeta)
    assert 0.0 <= c <= 1.0


def test_iauc_basic():
    """Test iAUC computation."""
    np.random.seed(0)
    risk = np.random.randn(40)
    y = np.random.exponential(2.0, 40)
    zeta = np.random.binomial(1, 0.6, 40)
    iauc = time_dependent_auc_iAUC(risk, y, zeta)
    assert isinstance(iauc, float)
    if not np.isnan(iauc):
        assert 0.0 <= iauc <= 1.0
