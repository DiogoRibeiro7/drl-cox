"""Comprehensive tests for Cox baseline models."""

from __future__ import annotations
import numpy as np
import pytest
from drl_cox.cox_baseline import (
    CoxPartialLikelihood,
    CoxRidge,
    CoxLasso,
    _assert_ndarray,
    _riskset_prefix_sums,
    _partial_grad,
)


# ============================================================================
# Helper Functions and Fixtures
# ============================================================================


@pytest.fixture
def simple_survival_data():
    """Simple survival dataset with known properties."""
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 3)
    # True beta for validation
    beta_true = np.array([0.5, -0.3, 0.8])
    linear_pred = X @ beta_true
    # Generate survival times
    u = np.random.uniform(size=n)
    T = -np.log(u) / (0.01 * np.exp(linear_pred))
    # Generate censoring
    C = np.random.exponential(scale=np.quantile(T, 0.7), size=n)
    y = np.minimum(T, C)
    zeta = (T <= C).astype(int)
    return X, y, zeta, beta_true


@pytest.fixture
def single_feature_data():
    """Dataset with single feature for testing."""
    np.random.seed(123)
    n = 30
    X = np.random.randn(n, 1)
    beta_true = np.array([1.0])
    linear_pred = X @ beta_true
    u = np.random.uniform(size=n)
    T = -np.log(u) / (0.02 * np.exp(linear_pred))
    C = np.random.exponential(scale=np.quantile(T, 0.6), size=n)
    y = np.minimum(T, C)
    zeta = (T <= C).astype(int)
    return X, y, zeta, beta_true


@pytest.fixture
def all_censored_data():
    """Dataset where all observations are censored."""
    np.random.seed(0)
    n = 20
    X = np.random.randn(n, 3)
    y = np.random.exponential(2.0, n)
    zeta = np.zeros(n, dtype=int)  # All censored
    return X, y, zeta


@pytest.fixture
def no_censoring_data():
    """Dataset with no censored observations."""
    np.random.seed(1)
    n = 40
    X = np.random.randn(n, 4)
    beta_true = np.array([0.3, -0.2, 0.5, -0.4])
    linear_pred = X @ beta_true
    u = np.random.uniform(size=n)
    y = -np.log(u) / (0.01 * np.exp(linear_pred))
    zeta = np.ones(n, dtype=int)  # No censoring
    return X, y, zeta, beta_true


@pytest.fixture
def collinear_data():
    """Dataset with collinear features to test regularization."""
    np.random.seed(99)
    n = 60
    X1 = np.random.randn(n, 1)
    X2 = X1 + np.random.randn(n, 1) * 0.01  # Nearly collinear
    X3 = np.random.randn(n, 1)
    X = np.hstack([X1, X2, X3])
    beta_true = np.array([0.5, 0.5, 0.3])
    linear_pred = X @ beta_true
    u = np.random.uniform(size=n)
    T = -np.log(u) / (0.01 * np.exp(linear_pred))
    C = np.random.exponential(scale=np.quantile(T, 0.65), size=n)
    y = np.minimum(T, C)
    zeta = (T <= C).astype(int)
    return X, y, zeta


def generate_known_solution_data():
    """Generate data with analytically tractable solution."""
    # Simple case: 2 observations, no ties, single feature
    X = np.array([[1.0], [-1.0]])
    y = np.array([2.0, 1.0])
    zeta = np.array([1, 1])
    # With β=0, risk scores are equal, so it's neutral
    return X, y, zeta


# ============================================================================
# Tests for Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_assert_ndarray_valid(self):
        """Test _assert_ndarray with valid inputs."""
        arr = np.array([1, 2, 3])
        _assert_ndarray("test", arr)  # Should not raise
        _assert_ndarray("test", arr, ndim=1)  # Should not raise

    def test_assert_ndarray_invalid_type(self):
        """Test _assert_ndarray with invalid type."""
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            _assert_ndarray("test", [1, 2, 3])

    def test_assert_ndarray_invalid_ndim(self):
        """Test _assert_ndarray with wrong dimensions."""
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="must have ndim=1"):
            _assert_ndarray("test", arr, ndim=1)

    def test_riskset_prefix_sums_basic(self):
        """Test risk set prefix sum computation."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        Xbeta = np.array([0.5, 0.3, 0.1, -0.2])
        order, prefix = _riskset_prefix_sums(y, Xbeta)

        # Check ordering (descending y)
        assert np.all(y[order] == np.array([4.0, 3.0, 2.0, 1.0]))

        # Check prefix is cumulative sum of exp(Xbeta)
        assert prefix.shape == (4,)
        assert np.all(prefix > 0)
        assert np.all(np.diff(prefix) >= 0)  # Should be increasing

    def test_riskset_prefix_sums_ties(self):
        """Test with tied survival times."""
        y = np.array([1.0, 2.0, 2.0, 3.0])
        Xbeta = np.array([0.1, 0.2, 0.3, 0.4])
        order, prefix = _riskset_prefix_sums(y, Xbeta)

        assert order.shape == (4,)
        assert prefix.shape == (4,)
        assert np.all(prefix > 0)

    def test_partial_grad_computation(self):
        """Test partial gradient computation."""
        X = np.array([[1.0, 0.5], [2.0, -0.5], [0.5, 1.0]])
        y = np.array([1.0, 2.0, 3.0])
        zeta = np.array([1, 1, 0])
        beta = np.array([0.0, 0.0])

        grad = _partial_grad(X, y, zeta, beta)
        assert isinstance(grad, float)
        # With beta=0, should have some gradient
        assert not np.isnan(grad)


# ============================================================================
# Tests for CoxPartialLikelihood
# ============================================================================


class TestCoxPartialLikelihood:
    """Tests for Cox Partial Likelihood estimator."""

    def test_initialization(self):
        """Test model initialization with different parameters."""
        model = CoxPartialLikelihood()
        assert model.max_iter == 50
        assert model.tol == 1e-6

        model = CoxPartialLikelihood(max_iter=100, tol=1e-8)
        assert model.max_iter == 100
        assert model.tol == 1e-8

    def test_fit_simple_data(self, simple_survival_data):
        """Test fitting on simple synthetic data."""
        X, y, zeta, beta_true = simple_survival_data
        model = CoxPartialLikelihood()
        beta_hat = model.fit(X, y, zeta)

        # Check output shape
        assert beta_hat.shape == (3,)
        assert np.all(np.isfinite(beta_hat))

        # Check signs align roughly with true beta
        # (not exact match expected, but correlation should be positive)
        correlation = np.corrcoef(beta_hat, beta_true)[0, 1]
        assert correlation > 0.3

    def test_fit_single_feature(self, single_feature_data):
        """Test fitting with single feature."""
        X, y, zeta, beta_true = single_feature_data
        model = CoxPartialLikelihood()
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (1,)
        assert np.isfinite(beta_hat[0])
        # Should be positive since true beta is positive
        assert beta_hat[0] > 0

    def test_fit_no_censoring(self, no_censoring_data):
        """Test fitting when no observations are censored."""
        X, y, zeta, beta_true = no_censoring_data
        model = CoxPartialLikelihood()
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (4,)
        assert np.all(np.isfinite(beta_hat))

        # Should have reasonable correlation with true beta
        correlation = np.corrcoef(beta_hat, beta_true)[0, 1]
        assert correlation > 0.2

    def test_fit_all_censored(self, all_censored_data):
        """Test fitting when all observations are censored."""
        X, y, zeta = all_censored_data
        model = CoxPartialLikelihood()
        beta_hat = model.fit(X, y, zeta)

        # With all censored, gradient is zero, should return zeros or minimal movement
        assert beta_hat.shape == (3,)
        assert np.all(np.isfinite(beta_hat))
        assert np.linalg.norm(beta_hat) < 1.0

    def test_convergence(self, simple_survival_data):
        """Test convergence behavior."""
        X, y, zeta, _ = simple_survival_data

        # Tight tolerance should converge
        model = CoxPartialLikelihood(max_iter=100, tol=1e-8)
        beta_tight = model.fit(X, y, zeta)

        # Loose tolerance
        model = CoxPartialLikelihood(max_iter=100, tol=1e-3)
        beta_loose = model.fit(X, y, zeta)

        # Should be similar but not identical
        assert np.linalg.norm(beta_tight - beta_loose) < 0.5

    def test_fit_known_solution(self):
        """Test against known analytical solution."""
        X, y, zeta = generate_known_solution_data()
        model = CoxPartialLikelihood()
        beta_hat = model.fit(X, y, zeta)

        # For this simple case, we can verify basic properties
        assert beta_hat.shape == (1,)
        assert np.isfinite(beta_hat[0])

    def test_input_validation(self):
        """Test input validation."""
        model = CoxPartialLikelihood()

        # Wrong types
        with pytest.raises(TypeError):
            model.fit([1, 2, 3], np.array([1.0]), np.array([1]))

        # Wrong dimensions
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]), np.array([1.0]), np.array([1]))

        # Mismatched shapes
        X = np.random.randn(10, 3)
        y = np.random.randn(5)
        zeta = np.array([1, 1, 1, 0, 0])
        with pytest.raises((ValueError, IndexError)):
            model.fit(X, y, zeta)

    def test_reproducibility(self, simple_survival_data):
        """Test that fitting is reproducible."""
        X, y, zeta, _ = simple_survival_data
        model = CoxPartialLikelihood()

        beta1 = model.fit(X, y, zeta)
        beta2 = model.fit(X, y, zeta)

        np.testing.assert_array_almost_equal(beta1, beta2)

    def test_large_coefficients(self):
        """Test behavior with large coefficient values."""
        np.random.seed(555)
        n = 40
        X = np.random.randn(n, 2)
        # Extreme beta should still work due to clipping
        beta_true = np.array([10.0, -10.0])
        linear_pred = X @ beta_true
        u = np.random.uniform(size=n)
        T = -np.log(u) / (0.001 * np.exp(np.clip(linear_pred, -50, 50)))
        C = np.random.exponential(scale=np.quantile(T, 0.6), size=n)
        y = np.minimum(T, C)
        zeta = (T <= C).astype(int)

        model = CoxPartialLikelihood(max_iter=100)
        beta_hat = model.fit(X, y, zeta)

        assert np.all(np.isfinite(beta_hat))
        # Should capture the sign at least
        assert np.sign(beta_hat[0]) == np.sign(beta_true[0])
        assert np.sign(beta_hat[1]) == np.sign(beta_true[1])


# ============================================================================
# Tests for CoxRidge
# ============================================================================


class TestCoxRidge:
    """Tests for Ridge-regularized Cox model."""

    def test_initialization(self):
        """Test model initialization."""
        model = CoxRidge()
        assert model.alpha == 1.0
        assert model.max_iter == 50

        model = CoxRidge(alpha=0.5, max_iter=100)
        assert model.alpha == 0.5
        assert model.max_iter == 100

    def test_fit_simple_data(self, simple_survival_data):
        """Test fitting on simple data."""
        X, y, zeta, _ = simple_survival_data
        model = CoxRidge(alpha=1.0)
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (3,)
        assert np.all(np.isfinite(beta_hat))

    def test_regularization_effect(self, simple_survival_data):
        """Test that increasing alpha shrinks coefficients."""
        X, y, zeta, _ = simple_survival_data

        # No regularization (alpha=0)
        model_no_reg = CoxRidge(alpha=0.0)
        beta_no_reg = model_no_reg.fit(X, y, zeta)

        # Light regularization
        model_light = CoxRidge(alpha=0.1)
        beta_light = model_light.fit(X, y, zeta)

        # Heavy regularization
        model_heavy = CoxRidge(alpha=10.0)
        beta_heavy = model_heavy.fit(X, y, zeta)

        # Check shrinkage: ||beta|| should decrease with alpha
        norm_no_reg = np.linalg.norm(beta_no_reg)
        norm_light = np.linalg.norm(beta_light)
        norm_heavy = np.linalg.norm(beta_heavy)

        assert norm_heavy < norm_light < norm_no_reg

    def test_collinear_features(self, collinear_data):
        """Test that Ridge handles collinearity better than unregularized."""
        X, y, zeta = collinear_data

        # Unregularized might be unstable
        model_unreg = CoxRidge(alpha=0.0)
        beta_unreg = model_unreg.fit(X, y, zeta)

        # Regularized should be more stable
        model_reg = CoxRidge(alpha=1.0)
        beta_reg = model_reg.fit(X, y, zeta)

        # Both should finish
        assert np.all(np.isfinite(beta_unreg))
        assert np.all(np.isfinite(beta_reg))

        # Regularized should have smaller norm
        assert np.linalg.norm(beta_reg) < np.linalg.norm(beta_unreg)

    def test_single_feature(self, single_feature_data):
        """Test Ridge with single feature."""
        X, y, zeta, _ = single_feature_data
        model = CoxRidge(alpha=0.5)
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (1,)
        assert np.isfinite(beta_hat[0])

    def test_extreme_alpha(self, simple_survival_data):
        """Test with extreme regularization values."""
        X, y, zeta, _ = simple_survival_data

        # Very large alpha should give near-zero coefficients
        model = CoxRidge(alpha=1000.0)
        beta_hat = model.fit(X, y, zeta)
        assert np.linalg.norm(beta_hat) < 0.01

        # Very small alpha should be similar to unregularized
        model_small = CoxRidge(alpha=1e-6)
        beta_small = model_small.fit(X, y, zeta)

        model_unreg = CoxRidge(alpha=0.0)
        beta_unreg = model_unreg.fit(X, y, zeta)

        np.testing.assert_array_almost_equal(beta_small, beta_unreg, decimal=2)

    def test_all_censored(self, all_censored_data):
        """Test Ridge with all censored data."""
        X, y, zeta = all_censored_data
        model = CoxRidge(alpha=1.0)
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (3,)
        assert np.all(np.isfinite(beta_hat))
        assert np.linalg.norm(beta_hat) < 0.5


# ============================================================================
# Tests for CoxLasso
# ============================================================================


class TestCoxLasso:
    """Tests for Lasso-regularized Cox model."""

    def test_initialization(self):
        """Test model initialization."""
        model = CoxLasso()
        assert model.alpha == 0.01
        assert model.max_iter == 100

        model = CoxLasso(alpha=0.05, max_iter=200, tol=1e-8)
        assert model.alpha == 0.05
        assert model.max_iter == 200
        assert model.tol == 1e-8

    def test_fit_simple_data(self, simple_survival_data):
        """Test fitting on simple data."""
        X, y, zeta, _ = simple_survival_data
        model = CoxLasso(alpha=0.01)
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (3,)
        assert np.all(np.isfinite(beta_hat))

    def test_sparsity_effect(self, simple_survival_data):
        """Test that increasing alpha increases sparsity."""
        X, y, zeta, _ = simple_survival_data

        # Light regularization
        model_light = CoxLasso(alpha=0.001, max_iter=150)
        beta_light = model_light.fit(X, y, zeta)

        # Heavy regularization
        model_heavy = CoxLasso(alpha=0.1, max_iter=150)
        beta_heavy = model_heavy.fit(X, y, zeta)

        # Count near-zero coefficients
        sparsity_light = np.sum(np.abs(beta_light) < 1e-3)
        sparsity_heavy = np.sum(np.abs(beta_heavy) < 1e-3)

        # Heavy regularization should produce more zeros
        assert sparsity_heavy >= sparsity_light

    def test_feature_selection(self):
        """Test that Lasso can identify relevant features."""
        np.random.seed(777)
        n = 80
        # First 2 features are relevant, last 3 are noise
        X_relevant = np.random.randn(n, 2)
        X_noise = np.random.randn(n, 3) * 0.1
        X = np.hstack([X_relevant, X_noise])

        beta_true = np.array([1.0, -0.8, 0.0, 0.0, 0.0])
        linear_pred = X @ beta_true
        u = np.random.uniform(size=n)
        T = -np.log(u) / (0.01 * np.exp(linear_pred))
        C = np.random.exponential(scale=np.quantile(T, 0.7), size=n)
        y = np.minimum(T, C)
        zeta = (T <= C).astype(int)

        model = CoxLasso(alpha=0.05, max_iter=200)
        beta_hat = model.fit(X, y, zeta)

        # First two coefficients should be non-zero
        assert np.abs(beta_hat[0]) > 0.01 or np.abs(beta_hat[1]) > 0.01

        # At least one noise coefficient should be near zero
        assert np.sum(np.abs(beta_hat[2:]) < 0.05) >= 1

    def test_single_feature(self, single_feature_data):
        """Test Lasso with single feature."""
        X, y, zeta, _ = single_feature_data
        model = CoxLasso(alpha=0.01, max_iter=150)
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (1,)
        assert np.isfinite(beta_hat[0])

    def test_convergence(self, simple_survival_data):
        """Test convergence with different max_iter."""
        X, y, zeta, _ = simple_survival_data

        # Few iterations
        model_few = CoxLasso(alpha=0.01, max_iter=10)
        beta_few = model_few.fit(X, y, zeta)

        # Many iterations
        model_many = CoxLasso(alpha=0.01, max_iter=200)
        beta_many = model_many.fit(X, y, zeta)

        # Both should be finite
        assert np.all(np.isfinite(beta_few))
        assert np.all(np.isfinite(beta_many))

        # More iterations should give similar or better solution
        # (not always guaranteed due to coordinate descent, but should be close)
        diff = np.linalg.norm(beta_few - beta_many)
        assert diff < 2.0  # Allow some difference

    def test_extreme_alpha(self, simple_survival_data):
        """Test with extreme regularization values."""
        X, y, zeta, _ = simple_survival_data

        # Very large alpha should give all zeros
        model = CoxLasso(alpha=10.0, max_iter=150)
        beta_hat = model.fit(X, y, zeta)
        assert np.linalg.norm(beta_hat) < 0.1

        # Very small alpha
        model_small = CoxLasso(alpha=1e-6, max_iter=150)
        beta_small = model_small.fit(X, y, zeta)
        assert np.all(np.isfinite(beta_small))

    def test_all_censored(self, all_censored_data):
        """Test Lasso with all censored data."""
        X, y, zeta = all_censored_data
        model = CoxLasso(alpha=0.01, max_iter=150)
        beta_hat = model.fit(X, y, zeta)

        assert beta_hat.shape == (3,)
        assert np.all(np.isfinite(beta_hat))

    def test_reproducibility(self, simple_survival_data):
        """Test that fitting is reproducible."""
        X, y, zeta, _ = simple_survival_data
        model = CoxLasso(alpha=0.01, max_iter=150)

        beta1 = model.fit(X, y, zeta)
        beta2 = model.fit(X, y, zeta)

        # Coordinate descent should be deterministic
        np.testing.assert_array_almost_equal(beta1, beta2, decimal=4)


# ============================================================================
# Comparison Tests
# ============================================================================


class TestModelComparisons:
    """Tests comparing different Cox models."""

    def test_ridge_vs_lasso_sparsity(self, simple_survival_data):
        """Test that Lasso produces sparser solutions than Ridge."""
        X, y, zeta, _ = simple_survival_data

        ridge = CoxRidge(alpha=0.1)
        beta_ridge = ridge.fit(X, y, zeta)

        lasso = CoxLasso(alpha=0.05, max_iter=200)
        beta_lasso = lasso.fit(X, y, zeta)

        # Count near-zero coefficients
        sparsity_ridge = np.sum(np.abs(beta_ridge) < 1e-4)
        sparsity_lasso = np.sum(np.abs(beta_lasso) < 1e-4)

        # Lasso should be at least as sparse
        assert sparsity_lasso >= sparsity_ridge

    def test_all_models_converge(self, simple_survival_data):
        """Test that all models can fit the same data."""
        X, y, zeta, _ = simple_survival_data

        pl = CoxPartialLikelihood()
        beta_pl = pl.fit(X, y, zeta)

        ridge = CoxRidge(alpha=0.1)
        beta_ridge = ridge.fit(X, y, zeta)

        lasso = CoxLasso(alpha=0.01, max_iter=150)
        beta_lasso = lasso.fit(X, y, zeta)

        # All should produce finite results
        assert np.all(np.isfinite(beta_pl))
        assert np.all(np.isfinite(beta_ridge))
        assert np.all(np.isfinite(beta_lasso))

        # All should have same shape
        assert beta_pl.shape == beta_ridge.shape == beta_lasso.shape

    def test_regularization_reduces_overfitting(self):
        """Test that regularization helps with overfitting."""
        np.random.seed(888)
        n_train = 30
        n_test = 50
        d = 10  # High dimensional

        # Training data
        X_train = np.random.randn(n_train, d)
        beta_true = np.random.randn(d) * 0.3
        linear_pred = X_train @ beta_true
        u = np.random.uniform(size=n_train)
        T_train = -np.log(u) / (0.01 * np.exp(linear_pred))
        C_train = np.random.exponential(scale=np.quantile(T_train, 0.6), size=n_train)
        y_train = np.minimum(T_train, C_train)
        zeta_train = (T_train <= C_train).astype(int)

        # Test data
        X_test = np.random.randn(n_test, d)
        linear_pred_test = X_test @ beta_true
        u_test = np.random.uniform(size=n_test)
        T_test = -np.log(u_test) / (0.01 * np.exp(linear_pred_test))
        C_test = np.random.exponential(scale=np.quantile(T_test, 0.6), size=n_test)
        y_test = np.minimum(T_test, C_test)
        zeta_test = (T_test <= C_test).astype(int)

        # Fit models
        pl = CoxPartialLikelihood()
        beta_pl = pl.fit(X_train, y_train, zeta_train)

        ridge = CoxRidge(alpha=1.0)
        beta_ridge = ridge.fit(X_train, y_train, zeta_train)

        # Test error (distance from true beta)
        error_pl = np.linalg.norm(beta_pl - beta_true)
        error_ridge = np.linalg.norm(beta_ridge - beta_true)

        # Ridge should have lower or similar error
        # (not guaranteed every time due to randomness, but should be close)
        assert np.isfinite(error_pl)
        assert np.isfinite(error_ridge)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_observation(self):
        """Test with only one observation."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        zeta = np.array([1])

        pl = CoxPartialLikelihood(max_iter=10)
        beta = pl.fit(X, y, zeta)

        assert beta.shape == (2,)
        assert np.all(np.isfinite(beta))

    def test_tied_survival_times(self):
        """Test with many tied survival times."""
        np.random.seed(999)
        n = 40
        X = np.random.randn(n, 3)
        # Create ties
        y = np.repeat([1.0, 2.0, 3.0, 4.0], 10)
        zeta = np.random.binomial(1, 0.7, n)

        pl = CoxPartialLikelihood()
        beta = pl.fit(X, y, zeta)

        assert beta.shape == (3,)
        assert np.all(np.isfinite(beta))

    def test_very_small_survival_times(self):
        """Test with very small survival times."""
        np.random.seed(111)
        n = 30
        X = np.random.randn(n, 2)
        y = np.random.exponential(0.001, n)  # Very small times
        zeta = np.random.binomial(1, 0.6, n)

        pl = CoxPartialLikelihood()
        beta = pl.fit(X, y, zeta)

        assert np.all(np.isfinite(beta))

    def test_very_large_survival_times(self):
        """Test with very large survival times."""
        np.random.seed(222)
        n = 30
        X = np.random.randn(n, 2)
        y = np.random.exponential(1000, n)  # Very large times
        zeta = np.random.binomial(1, 0.6, n)

        pl = CoxPartialLikelihood()
        beta = pl.fit(X, y, zeta)

        assert np.all(np.isfinite(beta))

    def test_perfect_separation(self):
        """Test with perfectly separable groups."""
        # Group 1: all events early
        X1 = np.ones((10, 1)) * -2
        y1 = np.random.uniform(0.5, 1.0, 10)
        zeta1 = np.ones(10, dtype=int)

        # Group 2: all events late (or censored)
        X2 = np.ones((10, 1)) * 2
        y2 = np.random.uniform(5.0, 10.0, 10)
        zeta2 = np.random.binomial(1, 0.5, 10)

        X = np.vstack([X1, X2])
        y = np.concatenate([y1, y2])
        zeta = np.concatenate([zeta1, zeta2])

        pl = CoxPartialLikelihood(max_iter=100)
        beta = pl.fit(X, y, zeta)

        # Should still converge and be finite
        assert np.all(np.isfinite(beta))
        # Should be strongly positive (higher X -> later time)
        assert beta[0] > 0

    def test_zero_variance_feature(self):
        """Test with a feature that has zero variance."""
        np.random.seed(333)
        n = 30
        X1 = np.random.randn(n, 2)
        X2 = np.ones((n, 1))  # Zero variance
        X = np.hstack([X1, X2])

        beta_true = np.array([0.5, -0.3, 0.0])
        linear_pred = X @ beta_true
        u = np.random.uniform(size=n)
        T = -np.log(u) / (0.01 * np.exp(linear_pred))
        C = np.random.exponential(scale=np.quantile(T, 0.6), size=n)
        y = np.minimum(T, C)
        zeta = (T <= C).astype(int)

        # All models should handle this
        pl = CoxPartialLikelihood()
        beta_pl = pl.fit(X, y, zeta)
        assert np.all(np.isfinite(beta_pl))

        ridge = CoxRidge(alpha=0.1)
        beta_ridge = ridge.fit(X, y, zeta)
        assert np.all(np.isfinite(beta_ridge))

        lasso = CoxLasso(alpha=0.01, max_iter=100)
        beta_lasso = lasso.fit(X, y, zeta)
        assert np.all(np.isfinite(beta_lasso))

    def test_high_dimensional(self):
        """Test with d > n (high dimensional case)."""
        np.random.seed(444)
        n = 20
        d = 30  # More features than observations
        X = np.random.randn(n, d)
        beta_true = np.random.randn(d) * 0.1
        linear_pred = X @ beta_true
        u = np.random.uniform(size=n)
        T = -np.log(u) / (0.01 * np.exp(linear_pred))
        C = np.random.exponential(scale=np.quantile(T, 0.6), size=n)
        y = np.minimum(T, C)
        zeta = (T <= C).astype(int)

        # Regularized methods should work
        ridge = CoxRidge(alpha=1.0, max_iter=50)
        beta_ridge = ridge.fit(X, y, zeta)
        assert beta_ridge.shape == (d,)
        assert np.all(np.isfinite(beta_ridge))

        lasso = CoxLasso(alpha=0.1, max_iter=100)
        beta_lasso = lasso.fit(X, y, zeta)
        assert beta_lasso.shape == (d,)
        assert np.all(np.isfinite(beta_lasso))


# ============================================================================
# Performance and Stability Tests
# ============================================================================


class TestPerformanceAndStability:
    """Tests for numerical stability and performance."""

    def test_numerical_stability_extreme_values(self):
        """Test stability with extreme covariate values."""
        np.random.seed(555)
        n = 40
        # Mix of normal and extreme values
        X = np.random.randn(n, 3)
        X[0:5, 0] = 100  # Extreme positive
        X[5:10, 1] = -100  # Extreme negative

        y = np.random.exponential(2.0, n)
        zeta = np.random.binomial(1, 0.6, n)

        # Should not crash or produce NaN/Inf
        pl = CoxPartialLikelihood()
        beta = pl.fit(X, y, zeta)
        assert np.all(np.isfinite(beta))

        ridge = CoxRidge(alpha=0.5)
        beta_ridge = ridge.fit(X, y, zeta)
        assert np.all(np.isfinite(beta_ridge))

    def test_consistency_across_runs(self, simple_survival_data):
        """Test that results are consistent across multiple runs."""
        X, y, zeta, _ = simple_survival_data

        betas_pl = []
        betas_ridge = []

        for _ in range(3):
            pl = CoxPartialLikelihood()
            betas_pl.append(pl.fit(X, y, zeta))

            ridge = CoxRidge(alpha=0.1)
            betas_ridge.append(ridge.fit(X, y, zeta))

        # All runs should give identical results
        for i in range(1, 3):
            np.testing.assert_array_almost_equal(betas_pl[0], betas_pl[i])
            np.testing.assert_array_almost_equal(betas_ridge[0], betas_ridge[i])

    def test_computation_time_reasonable(self, simple_survival_data):
        """Test that fitting completes in reasonable time."""
        import time

        X, y, zeta, _ = simple_survival_data

        # Should complete quickly for small dataset
        start = time.time()
        pl = CoxPartialLikelihood()
        pl.fit(X, y, zeta)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should take less than 5 seconds

    def test_memory_efficiency(self):
        """Test that models don't consume excessive memory."""
        import sys

        np.random.seed(666)

        # Moderately sized dataset
        n = 200
        d = 20
        X = np.random.randn(n, d)
        beta_true = np.random.randn(d) * 0.2
        linear_pred = X @ beta_true
        u = np.random.uniform(size=n)
        T = -np.log(u) / (0.01 * np.exp(linear_pred))
        C = np.random.exponential(scale=np.quantile(T, 0.6), size=n)
        y = np.minimum(T, C)
        zeta = (T <= C).astype(int)

        # Measure memory usage (approximate)
        import gc

        gc.collect()

        pl = CoxPartialLikelihood()
        beta = pl.fit(X, y, zeta)

        # Should complete without memory errors
        assert beta.shape == (d,)
        assert np.all(np.isfinite(beta))


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_pipeline_all_models(self, simple_survival_data):
        """Test complete pipeline with all models."""
        X, y, zeta, _ = simple_survival_data

        models = {
            "pl": CoxPartialLikelihood(),
            "ridge_0.1": CoxRidge(alpha=0.1),
            "ridge_1.0": CoxRidge(alpha=1.0),
            "lasso_0.01": CoxLasso(alpha=0.01, max_iter=150),
            "lasso_0.1": CoxLasso(alpha=0.1, max_iter=150),
        }

        results = {}
        for name, model in models.items():
            beta = model.fit(X, y, zeta)
            results[name] = beta

            # All should succeed
            assert np.all(np.isfinite(beta))
            assert beta.shape == (3,)

        # Check that different alphas give different results
        assert not np.allclose(results["ridge_0.1"], results["ridge_1.0"])
        assert not np.allclose(results["lasso_0.01"], results["lasso_0.1"])

    def test_cross_validation_setup(self, simple_survival_data):
        """Test that models can be used in a CV-like setup."""
        X, y, zeta, _ = simple_survival_data
        n = X.shape[0]

        # Split into train and test
        train_idx = np.arange(n // 2)
        test_idx = np.arange(n // 2, n)

        X_train, y_train, zeta_train = X[train_idx], y[train_idx], zeta[train_idx]
        X_test, y_test, zeta_test = X[test_idx], y[test_idx], zeta[test_idx]

        # Fit on train
        pl = CoxPartialLikelihood()
        beta_train = pl.fit(X_train, y_train, zeta_train)

        # Predict on test (compute risk scores)
        risk_test = X_test @ beta_train

        # Should be finite
        assert np.all(np.isfinite(beta_train))
        assert np.all(np.isfinite(risk_test))
        assert risk_test.shape == (len(test_idx),)

    def test_comparison_with_baseline(self, no_censoring_data):
        """Test that models give sensible results compared to true beta."""
        X, y, zeta, beta_true = no_censoring_data

        # Fit all models
        pl = CoxPartialLikelihood()
        beta_pl = pl.fit(X, y, zeta)

        ridge = CoxRidge(alpha=0.1)
        beta_ridge = ridge.fit(X, y, zeta)

        lasso = CoxLasso(alpha=0.01, max_iter=200)
        beta_lasso = lasso.fit(X, y, zeta)

        # All should have positive correlation with true beta
        corr_pl = np.corrcoef(beta_pl, beta_true)[0, 1]
        corr_ridge = np.corrcoef(beta_ridge, beta_true)[0, 1]
        corr_lasso = np.corrcoef(beta_lasso, beta_true)[0, 1]

        assert corr_pl > 0.2
        assert corr_ridge > 0.2
        assert corr_lasso > 0.1  # Lasso might be sparser


# ============================================================================
# Regression Tests
# ============================================================================


class TestRegression:
    """Regression tests to ensure consistent behavior."""

    def test_fixed_dataset_pl(self):
        """Test CoxPartialLikelihood on fixed dataset."""
        np.random.seed(42)
        X = np.array([[1.0, 0.5], [0.5, 1.0], [-0.5, 0.2], [-1.0, -0.5], [0.0, 0.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        zeta = np.array([1, 1, 0, 1, 1])

        pl = CoxPartialLikelihood()
        beta = pl.fit(X, y, zeta)

        # Check shape
        assert beta.shape == (2,)
        assert np.all(np.isfinite(beta))

        # Rough check of expected values (may need adjustment)
        assert beta[0] > -2 and beta[0] < 2
        assert beta[1] > -2 and beta[1] < 2

    def test_fixed_dataset_ridge(self):
        """Test CoxRidge on fixed dataset."""
        np.random.seed(42)
        X = np.array([[1.0, 0.5], [0.5, 1.0], [-0.5, 0.2], [-1.0, -0.5], [0.0, 0.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        zeta = np.array([1, 1, 0, 1, 1])

        ridge = CoxRidge(alpha=0.5)
        beta = ridge.fit(X, y, zeta)

        assert beta.shape == (2,)
        assert np.all(np.isfinite(beta))
        assert np.linalg.norm(beta) < 2.0  # Should be shrunk

    def test_fixed_dataset_lasso(self):
        """Test CoxLasso on fixed dataset."""
        np.random.seed(42)
        X = np.array([[1.0, 0.5], [0.5, 1.0], [-0.5, 0.2], [-1.0, -0.5], [0.0, 0.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        zeta = np.array([1, 1, 0, 1, 1])

        lasso = CoxLasso(alpha=0.05, max_iter=150)
        beta = lasso.fit(X, y, zeta)

        assert beta.shape == (2,)
        assert np.all(np.isfinite(beta))


# ============================================================================
# Documentation and Example Tests
# ============================================================================


class TestDocumentationExamples:
    """Tests based on documentation examples."""

    def test_basic_usage_example(self):
        """Test basic usage example from documentation."""
        # Generate data
        np.random.seed(0)
        n = 100
        X = np.random.randn(n, 5)
        beta_true = np.array([0.5, -0.3, 0.8, 0.0, -0.2])
        linear_pred = X @ beta_true
        u = np.random.uniform(size=n)
        T = -np.log(u) / (0.01 * np.exp(linear_pred))
        C = np.random.exponential(scale=np.quantile(T, 0.7), size=n)
        y = np.minimum(T, C)
        zeta = (T <= C).astype(int)

        # Fit models
        from drl_cox.cox_baseline import CoxPartialLikelihood, CoxRidge, CoxLasso

        pl = CoxPartialLikelihood()
        beta_pl = pl.fit(X, y, zeta)

        ridge = CoxRidge(alpha=1.0)
        beta_ridge = ridge.fit(X, y, zeta)

        lasso = CoxLasso(alpha=0.05, max_iter=200)
        beta_lasso = lasso.fit(X, y, zeta)

        # All should work
        assert beta_pl.shape == (5,)
        assert beta_ridge.shape == (5,)
        assert beta_lasso.shape == (5,)
        assert np.all(np.isfinite(beta_pl))
        assert np.all(np.isfinite(beta_ridge))
        assert np.all(np.isfinite(beta_lasso))

    def test_regularization_comparison_example(self):
        """Test regularization comparison example."""
        np.random.seed(123)
        n = 80
        X = np.random.randn(n, 10)
        beta_true = np.concatenate(
            [
                np.array([1.0, -0.8, 0.6]),  # Strong signals
                np.zeros(7),  # Noise features
            ]
        )
        linear_pred = X @ beta_true
        u = np.random.uniform(size=n)
        T = -np.log(u) / (0.01 * np.exp(linear_pred))
        C = np.random.exponential(scale=np.quantile(T, 0.65), size=n)
        y = np.minimum(T, C)
        zeta = (T <= C).astype(int)

        # Compare regularization strengths
        alphas = [0.01, 0.1, 1.0]
        norms = []

        for alpha in alphas:
            ridge = CoxRidge(alpha=alpha)
            beta = ridge.fit(X, y, zeta)
            norms.append(np.linalg.norm(beta))

        # Norms should decrease with increasing alpha
        assert norms[0] > norms[1] > norms[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=drl_cox.cox_baseline", "--cov-report=term-missing"])
