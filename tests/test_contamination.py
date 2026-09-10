"""Tests for contamination utilities."""

from __future__ import annotations

import numpy as np

from drl_cox import inject_covariate_shift, inject_outliers


def test_inject_covariate_shift():
    """Test covariate shift injection."""
    X = np.random.randn(50, 10)
    X_shifted = inject_covariate_shift(X, feature_indices=[0, 1, 2], mean=5.0, std=2.0, seed=42)
    assert X_shifted.shape == X.shape
    # Shifted features should differ
    assert not np.allclose(X[:, :3], X_shifted[:, :3])
    # Other features unchanged
    assert np.allclose(X[:, 3:], X_shifted[:, 3:])


def test_inject_outliers():
    """Test outlier injection."""
    X = np.random.randn(100, 5)
    X_noisy = inject_outliers(X, ratio=0.2, severity_std=5.0, seed=0)
    assert X_noisy.shape == X.shape
    # Some rows should differ significantly
    row_diffs = np.linalg.norm(X - X_noisy, axis=1)
    assert np.sum(row_diffs > 1.0) >= 10  # at least some corrupted
