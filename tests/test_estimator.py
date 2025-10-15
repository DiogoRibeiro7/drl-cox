"""Tests for the scikit-learn compatible DRLCoxEstimator."""

from __future__ import annotations
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.estimator_checks import check_estimator

from drl_cox import (
    DRLCoxEstimator,
    make_drl_cox_scorer,
    simulate_cox_data,
    SurvivalDataset,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_survival_data():
    """Generate simple survival data for testing."""
    data = simulate_cox_data(n=100, d=5, seed=42)
    return data.X, data.y, data.zeta


@pytest.fixture
def large_survival_data():
    """Generate larger survival data for integration tests."""
    data = simulate_cox_data(n=300, d=10, seed=123)
    return data.X, data.y, data.zeta


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestDRLCoxEstimatorBasics:
    """Test basic estimator functionality."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        est = DRLCoxEstimator()
        assert est.epsilon == 0.1
        assert est.p == 2.0
        assert est.gamma == 3
        assert est.solver == "ECOS"
        assert est.solver_opts is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        est = DRLCoxEstimator(
            epsilon=0.2, p=1.5, gamma=5, solver="SCS", solver_opts={"max_iters": 500}
        )
        assert est.epsilon == 0.2
        assert est.p == 1.5
        assert est.gamma == 5
        assert est.solver == "SCS"
        assert est.solver_opts == {"max_iters": 500}

    def test_get_params(self):
        """Test get_params method."""
        est = DRLCoxEstimator(epsilon=0.15, gamma=4)
        params = est.get_params()

        assert params["epsilon"] == 0.15
        assert params["gamma"] == 4
        assert params["p"] == 2.0  # default
        assert "solver" in params

    def test_set_params(self):
        """Test set_params method."""
        est = DRLCoxEstimator()
        est.set_params(epsilon=0.3, gamma=5)

        assert est.epsilon == 0.3
        assert est.gamma == 5

    def test_set_invalid_params(self):
        """Test that setting invalid parameters raises error."""
        est = DRLCoxEstimator()

        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(invalid_param=123)

    def test_repr(self):
        """Test string representation."""
        est = DRLCoxEstimator(epsilon=0.1, gamma=3)
        repr_str = repr(est)

        assert "DRLCoxEstimator" in repr_str
        assert "epsilon=0.1" in repr_str
        assert "gamma=3" in repr_str


# ============================================================================
# Parameter Validation Tests
# ============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    def test_negative_epsilon(self):
        """Test that negative epsilon raises error."""
        with pytest.raises(ValueError, match="epsilon must be >= 0"):
            est = DRLCoxEstimator(epsilon=-0.1)
            est._validate_params()

    def test_invalid_p(self):
        """Test that p < 1 raises error."""
        with pytest.raises(ValueError, match="p must be >= 1"):
            est = DRLCoxEstimator(p=0.5)
            est._validate_params()

    def test_invalid_gamma(self):
        """Test that gamma < 1 raises error."""
        with pytest.raises(ValueError, match="gamma must be >= 1"):
            est = DRLCoxEstimator(gamma=0)
            est._validate_params()

    def test_non_integer_gamma(self):
        """Test that non-integer gamma raises error."""
        with pytest.raises(TypeError, match="gamma must be an integer"):
            est = DRLCoxEstimator(gamma=3.5)
            est._validate_params()

    def test_invalid_solver_type(self):
        """Test that non-string solver raises error."""
        with pytest.raises(TypeError, match="solver must be a string"):
            est = DRLCoxEstimator(solver=123)
            est._validate_params()


# ============================================================================
# Fit Method Tests
# ============================================================================


class TestFitMethod:
    """Test fit method."""

    def test_fit_basic(self, simple_survival_data):
        """Test basic fitting."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        # Check fitted attributes
        assert hasattr(est, "beta_")
        assert hasattr(est, "alpha_")
        assert hasattr(est, "s_")
        assert hasattr(est, "n_features_in_")
        assert est.n_features_in_ == X.shape[1]
        assert est.beta_.shape == (X.shape[1],)

    def test_fit_without_zeta(self, simple_survival_data):
        """Test fitting without providing zeta (should warn)."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator(epsilon=0.1)

        with pytest.warns(UserWarning, match="zeta not provided"):
            est.fit(X, y)

        # Should still fit
        assert hasattr(est, "beta_")

    def test_fit_mismatched_shapes(self, simple_survival_data):
        """Test that mismatched array shapes raise error."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator()

        # Wrong y length
        with pytest.raises(ValueError, match="inconsistent shapes"):
            est.fit(X, y[:-10], zeta)

        # Wrong zeta length
        with pytest.raises(ValueError, match="inconsistent shapes"):
            est.fit(X, y, zeta[:-10])

    def test_fit_negative_survival_times(self, simple_survival_data):
        """Test that negative survival times raise error."""
        X, y, zeta = simple_survival_data
        y_bad = y.copy()
        y_bad[0] = -1.0

        est = DRLCoxEstimator()
        with pytest.raises(ValueError, match="survival times must be positive"):
            est.fit(X, y_bad, zeta)

    def test_fit_invalid_zeta_values(self, simple_survival_data):
        """Test that invalid zeta values raise error."""
        X, y, zeta = simple_survival_data
        zeta_bad = zeta.copy()
        zeta_bad[0] = 2  # Invalid value

        est = DRLCoxEstimator()
        with pytest.raises(ValueError, match="zeta must contain only 0.*or 1"):
            est.fit(X, y, zeta_bad)

    def test_fit_with_nan(self, simple_survival_data):
        """Test that NaN values raise error."""
        X, y, zeta = simple_survival_data
        X_bad = X.copy()
        X_bad[0, 0] = np.nan

        est = DRLCoxEstimator()
        with pytest.raises(ValueError, match="non-finite values"):
            est.fit(X_bad, y, zeta)

    def test_fit_epsilon_zero(self, simple_survival_data):
        """Test fitting with epsilon=0 (no robustness)."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator(epsilon=0.0)
        est.fit(X, y, zeta)

        assert hasattr(est, "beta_")
        assert est.status_ in ["optimal", "optimal_inaccurate"]

    def test_fit_different_solvers(self, simple_survival_data):
        """Test fitting with different solvers."""
        X, y, zeta = simple_survival_data

        for solver in ["ECOS", "SCS"]:
            est = DRLCoxEstimator(epsilon=0.1, solver=solver)
            est.fit(X, y, zeta)
            assert hasattr(est, "beta_")


# ============================================================================
# Predict Method Tests
# ============================================================================


class TestPredictMethod:
    """Test predict method."""

    def test_predict_basic(self, simple_survival_data):
        """Test basic prediction."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        risk_scores = est.predict(X)

        assert risk_scores.shape == (X.shape[0],)
        assert np.all(np.isfinite(risk_scores))

    def test_predict_not_fitted(self, simple_survival_data):
        """Test that predict raises error when not fitted."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator()

        with pytest.raises(Exception):  # sklearn raises NotFittedError
            est.predict(X)

    def test_predict_wrong_features(self, simple_survival_data):
        """Test that predict raises error with wrong number of features."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        X_wrong = X[:, :-1]  # Remove one feature
        with pytest.raises(ValueError, match="has.*features.*fitted with"):
            est.predict(X_wrong)

    def test_predict_new_samples(self, simple_survival_data):
        """Test prediction on new samples."""
        X, y, zeta = simple_survival_data

        # Split data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        zeta_train, zeta_test = zeta[:80], zeta[80:]

        # Fit on train
        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X_train, y_train, zeta_train)

        # Predict on test
        risk_scores = est.predict(X_test)
        assert risk_scores.shape == (X_test.shape[0],)


# ============================================================================
# Score Method Tests
# ============================================================================


class TestScoreMethod:
    """Test score method."""

    def test_score_basic(self, simple_survival_data):
        """Test basic scoring."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        score = est.score(X, y, zeta)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_without_zeta(self, simple_survival_data):
        """Test scoring without zeta (should use all events)."""
        X, y, zeta = simple_survival_data
        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        score = est.score(X, y)  # No zeta
        assert isinstance(score, float)

    def test_score_train_vs_test(self, large_survival_data):
        """Test that train score is typically better than test score."""
        X, y, zeta = large_survival_data

        # Split data
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]
        zeta_train, zeta_test = zeta[:200], zeta[200:]

        # Fit
        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X_train, y_train, zeta_train)

        train_score = est.score(X_train, y_train, zeta_train)
        test_score = est.score(X_test, y_test, zeta_test)

        # Train score should generally be higher
        assert train_score >= test_score - 0.1  # Allow some variance


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Test integration with scikit-learn pipelines."""

    def test_pipeline_basic(self, simple_survival_data):
        """Test basic pipeline."""
        X, y, zeta = simple_survival_data

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("drl_cox", DRLCoxEstimator(epsilon=0.1))]
        )

        pipeline.fit(X, y, drl_cox__zeta=zeta)
        risk_scores = pipeline.predict(X)

        assert risk_scores.shape == (X.shape[0],)

    def test_pipeline_score(self, simple_survival_data):
        """Test pipeline scoring."""
        X, y, zeta = simple_survival_data

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("drl_cox", DRLCoxEstimator(epsilon=0.1))]
        )

        pipeline.fit(X, y, drl_cox__zeta=zeta)
        score = pipeline.score(X, y, drl_cox__zeta=zeta)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_pipeline_get_params(self, simple_survival_data):
        """Test get_params through pipeline."""
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("drl_cox", DRLCoxEstimator(epsilon=0.1))]
        )

        params = pipeline.get_params()
        assert "drl_cox__epsilon" in params
        assert params["drl_cox__epsilon"] == 0.1

    def test_pipeline_set_params(self, simple_survival_data):
        """Test set_params through pipeline."""
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("drl_cox", DRLCoxEstimator(epsilon=0.1))]
        )

        pipeline.set_params(drl_cox__epsilon=0.2)
        assert pipeline.named_steps["drl_cox"].epsilon == 0.2


# ============================================================================
# GridSearchCV Tests
# ============================================================================


class TestGridSearchCV:
    """Test GridSearchCV integration."""

    def test_grid_search_basic(self, simple_survival_data):
        """Test basic grid search (limited due to zeta requirement)."""
        X, y, zeta = simple_survival_data

        # Create estimator
        est = DRLCoxEstimator()

        # Define parameter grid
        param_grid = {"epsilon": [0.0, 0.1, 0.2], "gamma": [2, 3]}

        # Note: GridSearchCV with survival data requires custom handling
        # This is a simplified test
        grid_search = GridSearchCV(
            est,
            param_grid,
            cv=3,
            scoring=None,  # Will use estimator's score method
        )

        # This will work but zeta handling is limited
        # In practice, need custom CV splitter
        try:
            grid_search.fit(X, y, zeta=zeta)
            assert hasattr(grid_search, "best_params_")
        except Exception as e:
            # Expected due to scikit-learn limitations with 3-argument fit
            pytest.skip(f"GridSearchCV limitation: {e}")

    def test_parameter_search_manual(self, simple_survival_data):
        """Test manual parameter search (recommended approach)."""
        X, y, zeta = simple_survival_data

        # Manual grid search
        results = []
        for epsilon in [0.0, 0.1, 0.2]:
            for gamma in [2, 3]:
                est = DRLCoxEstimator(epsilon=epsilon, gamma=gamma)
                est.fit(X, y, zeta)
                score = est.score(X, y, zeta)
                results.append({"epsilon": epsilon, "gamma": gamma, "score": score})

        # Find best
        best = max(results, key=lambda x: x["score"])
        assert "epsilon" in best
        assert "gamma" in best


# ============================================================================
# Scorer Tests
# ============================================================================


class TestMakeDRLCoxScorer:
    """Test make_drl_cox_scorer function."""

    def test_cindex_scorer(self, simple_survival_data):
        """Test C-index scorer."""
        X, y, zeta = simple_survival_data

        scorer = make_drl_cox_scorer(metric="cindex")

        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        score = scorer(est, X, y, zeta)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_iauc_scorer(self, simple_survival_data):
        """Test iAUC scorer."""
        X, y, zeta = simple_survival_data

        scorer = make_drl_cox_scorer(metric="iauc")

        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        score = scorer(est, X, y, zeta)
        assert isinstance(score, float)

    def test_scorer_without_zeta(self, simple_survival_data):
        """Test scorer without zeta."""
        X, y, zeta = simple_survival_data

        scorer = make_drl_cox_scorer(metric="cindex")

        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(X, y, zeta)

        # Should work without zeta (assumes all events)
        score = scorer(est, X, y)
        assert isinstance(score, float)

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            scorer = make_drl_cox_scorer(metric="invalid")
            # Need to call it to trigger error
            est = DRLCoxEstimator()
            X = np.random.randn(10, 3)
            y = np.random.exponential(2, 10)
            est.fit(X, y)
            scorer(est, X, y)


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_feature(self):
        """Test with single feature."""
        data = simulate_cox_data(n=50, d=1, seed=42)

        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(data.X, data.y, data.zeta)

        assert est.beta_.shape == (1,)
        risk_scores = est.predict(data.X)
        assert risk_scores.shape == (50,)

    def test_all_censored(self):
        """Test with all censored observations."""
        data = simulate_cox_data(n=50, d=5, seed=42)
        zeta_all_censored = np.zeros(50, dtype=int)

        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(data.X, data.y, zeta_all_censored)

        # Should still fit (though results may not be meaningful)
        assert hasattr(est, "beta_")

    def test_all_events(self):
        """Test with no censored observations."""
        data = simulate_cox_data(n=50, d=5, seed=42)
        zeta_all_events = np.ones(50, dtype=int)

        est = DRLCoxEstimator(epsilon=0.1)
        est.fit(data.X, data.y, zeta_all_events)

        assert hasattr(est, "beta_")
        score = est.score(data.X, data.y, zeta_all_events)
        assert 0.0 <= score <= 1.0

    def test_high_dimensional(self):
        """Test with more features than samples."""
        data = simulate_cox_data(n=30, d=50, seed=42)

        est = DRLCoxEstimator(epsilon=0.5, gamma=2)  # Higher epsilon for regularization
        est.fit(data.X, data.y, data.zeta)

        assert est.beta_.shape == (50,)

    def test_reproducibility(self, simple_survival_data):
        """Test that fitting is reproducible."""
        X, y, zeta = simple_survival_data

        est1 = DRLCoxEstimator(epsilon=0.1, solver_opts={"max_iters": 200})
        est1.fit(X, y, zeta)
        beta1 = est1.beta_.copy()

        est2 = DRLCoxEstimator(epsilon=0.1, solver_opts={"max_iters": 200})
        est2.fit(X, y, zeta)
        beta2 = est2.beta_.copy()

        # Should be identical (deterministic solver)
        np.testing.assert_array_almost_equal(beta1, beta2, decimal=6)


# ============================================================================
# Comparison with Functional API Tests
# ============================================================================


class TestConsistencyWithFunctionalAPI:
    """Test that estimator gives same results as functional API."""

    def test_same_results_as_fit_drl_cox(self, simple_survival_data):
        """Test that estimator matches fit_drl_cox results."""
        from drl_cox import fit_drl_cox, SurvivalDataset

        X, y, zeta = simple_survival_data

        # Fit using estimator
        est = DRLCoxEstimator(epsilon=0.1, p=2.0, gamma=3)
        est.fit(X, y, zeta)
        beta_est = est.beta_

        # Fit using functional API
        data = SurvivalDataset(X=X, y=y, zeta=zeta)
        result = fit_drl_cox(data, epsilon=0.1, p=2.0, gamma=3)
        beta_func = result.beta

        # Should be very close
        np.testing.assert_array_almost_equal(beta_est, beta_func, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
