"""
Comprehensive tests for the preprocessing module.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import warnings
from drl_cox import SurvivalDataset, simulate_cox_data
from drl_cox.preprocessing import (
    SurvivalStandardScaler,
    detect_outliers,
    handle_missing_data,
    train_test_split_survival,
    create_time_dependent_features,
    discretize_time,
    balance_censoring,
    validate_survival_data,
    compute_feature_importance,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_survival_data():
    """Generate simple survival data."""
    np.random.seed(42)
    return simulate_cox_data(n=100, d=5, seed=42)


@pytest.fixture
def data_with_missing():
    """Generate data with missing values."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    # Introduce missing values
    X[X < -2] = np.nan
    y = np.random.exponential(2, 50)
    zeta = np.random.binomial(1, 0.7, 50)
    return SurvivalDataset(X=X, y=y, zeta=zeta)


@pytest.fixture
def data_with_outliers():
    """Generate data with outliers."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # Add outliers
    X[0:5, :] = 10  # Extreme values
    y = np.random.exponential(2, 100)
    zeta = np.random.binomial(1, 0.6, 100)
    return SurvivalDataset(X=X, y=y, zeta=zeta)


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced survival data."""
    np.random.seed(42)
    data = simulate_cox_data(n=200, d=5, censor_rate=0.8, seed=42)
    return data


# ============================================================================
# Tests for SurvivalStandardScaler
# ============================================================================


class TestSurvivalStandardScaler:
    """Tests for SurvivalStandardScaler."""

    def test_init(self):
        """Test initialization."""
        scaler = SurvivalStandardScaler()
        assert scaler.with_mean is True
        assert scaler.with_std is True
        assert scaler.use_ipcw is False
        assert scaler.robust is False

    def test_fit_transform(self, simple_survival_data):
        """Test fit and transform."""
        scaler = SurvivalStandardScaler()
        data_scaled = scaler.fit_transform(simple_survival_data)

        # Check output type
        assert isinstance(data_scaled, SurvivalDataset)

        # Check mean and std
        assert np.allclose(data_scaled.X.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(data_scaled.X.std(axis=0, ddof=0), 1, atol=1e-10)

        # Check y and zeta unchanged
        np.testing.assert_array_equal(data_scaled.y, simple_survival_data.y)
        np.testing.assert_array_equal(data_scaled.zeta, simple_survival_data.zeta)

    def test_transform_new_data(self, simple_survival_data):
        """Test transform on new data."""
        # Split data
        train, test = train_test_split_survival(simple_survival_data, test_size=0.3)

        # Fit on train
        scaler = SurvivalStandardScaler()
        scaler.fit(train)

        # Transform test
        test_scaled = scaler.transform(test)

        assert isinstance(test_scaled, SurvivalDataset)
        assert test_scaled.X.shape == test.X.shape

    def test_inverse_transform(self, simple_survival_data):
        """Test inverse transform."""
        scaler = SurvivalStandardScaler()
        data_scaled = scaler.fit_transform(simple_survival_data)
        data_inv = scaler.inverse_transform(data_scaled)

        # Should recover original X
        np.testing.assert_array_almost_equal(data_inv.X, simple_survival_data.X, decimal=10)

    def test_robust_scaling(self, data_with_outliers):
        """Test robust scaling with outliers."""
        scaler = SurvivalStandardScaler(robust=True)
        data_scaled = scaler.fit_transform(data_with_outliers)

        # Check that scaling worked
        assert data_scaled.X.shape == data_with_outliers.X.shape

        # Median should be close to 0
        medians = np.median(data_scaled.X, axis=0)
        assert np.allclose(medians, 0, atol=0.1)

    def test_ipcw_weighting(self, simple_survival_data):
        """Test IPCW weighting."""
        scaler = SurvivalStandardScaler(use_ipcw=True)
        data_scaled = scaler.fit_transform(simple_survival_data)

        # Check weights were computed
        assert scaler._ipcw_weights is not None
        assert len(scaler._ipcw_weights) == len(simple_survival_data.y)
        assert np.all(scaler._ipcw_weights > 0)

    def test_without_centering(self, simple_survival_data):
        """Test scaling without centering."""
        scaler = SurvivalStandardScaler(with_mean=False)
        data_scaled = scaler.fit_transform(simple_survival_data)

        # Mean should not be zero
        means = data_scaled.X.mean(axis=0)
        assert not np.allclose(means, 0)

        # But std should be 1
        stds = data_scaled.X.std(axis=0, ddof=0)
        assert np.allclose(stds, 1, atol=1e-10)

    def test_without_scaling(self, simple_survival_data):
        """Test centering without scaling."""
        scaler = SurvivalStandardScaler(with_std=False)
        data_scaled = scaler.fit_transform(simple_survival_data)

        # Mean should be zero
        means = data_scaled.X.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-10)

        # But std should not be 1
        stds = data_scaled.X.std(axis=0, ddof=0)
        assert not np.allclose(stds, 1)

    def test_not_fitted_error(self, simple_survival_data):
        """Test error when transforming before fitting."""
        scaler = SurvivalStandardScaler()

        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(simple_survival_data)


# ============================================================================
# Tests for detect_outliers
# ============================================================================


class TestDetectOutliers:
    """Tests for outlier detection."""

    def test_mad_method(self, data_with_outliers):
        """Test MAD outlier detection."""
        outliers = detect_outliers(data_with_outliers, method="mad", threshold=3.0)

        assert isinstance(outliers, np.ndarray)
        assert outliers.dtype == bool
        assert len(outliers) == len(data_with_outliers.y)

        # Should detect at least some of the artificial outliers
        assert outliers[:5].sum() >= 3  # At least 3 of the 5 outliers

    def test_iqr_method(self, data_with_outliers):
        """Test IQR outlier detection."""
        outliers = detect_outliers(data_with_outliers, method="iqr", threshold=1.5)

        assert isinstance(outliers, np.ndarray)
        assert outliers.sum() > 0  # Should find some outliers

    def test_isolation_forest(self, data_with_outliers):
        """Test Isolation Forest outlier detection."""
        try:
            from sklearn.ensemble import IsolationForest

            outliers = detect_outliers(data_with_outliers, method="isolation", contamination=0.1)
            assert outliers.sum() > 0
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_lof_method(self, simple_survival_data):
        """Test Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor

            outliers = detect_outliers(simple_survival_data, method="lof", contamination=0.05)
            assert isinstance(outliers, np.ndarray)
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_return_scores(self, data_with_outliers):
        """Test returning outlier scores."""
        outliers, scores = detect_outliers(data_with_outliers, method="mad", return_scores=True)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(outliers)
        assert np.all(scores >= 0)

        # High scores should correspond to outliers
        assert np.mean(scores[outliers]) > np.mean(scores[~outliers])

    def test_invalid_method(self, simple_survival_data):
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            detect_outliers(simple_survival_data, method="invalid")


# ============================================================================
# Tests for handle_missing_data
# ============================================================================


class TestHandleMissingData:
    """Tests for missing data handling."""

    def test_no_missing_data(self, simple_survival_data):
        """Test with no missing values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_clean, mask = handle_missing_data(simple_survival_data, strategy="median")

        # Should return same data
        np.testing.assert_array_equal(data_clean.X, simple_survival_data.X)

    def test_drop_strategy(self, data_with_missing):
        """Test dropping samples with missing values."""
        data_clean, _ = handle_missing_data(data_with_missing, strategy="drop")

        # Should have fewer samples
        assert len(data_clean.y) < len(data_with_missing.y)

        # No missing values in result
        assert not np.any(np.isnan(data_clean.X))

    def test_mean_imputation(self, data_with_missing):
        """Test mean imputation."""
        data_clean, _ = handle_missing_data(data_with_missing, strategy="mean")

        # Same number of samples
        assert len(data_clean.y) == len(data_with_missing.y)

        # No missing values
        assert not np.any(np.isnan(data_clean.X))

    def test_median_imputation(self, data_with_missing):
        """Test median imputation."""
        data_clean, _ = handle_missing_data(data_with_missing, strategy="median")

        assert not np.any(np.isnan(data_clean.X))

    def test_forward_fill(self, data_with_missing):
        """Test forward fill imputation."""
        data_clean, _ = handle_missing_data(data_with_missing, strategy="forward")

        assert not np.any(np.isnan(data_clean.X))

    def test_mice_imputation(self, data_with_missing):
        """Test MICE imputation."""
        data_clean, _ = handle_missing_data(data_with_missing, strategy="mice", max_iter=5)

        assert not np.any(np.isnan(data_clean.X))

    def test_knn_imputation(self, data_with_missing):
        """Test KNN imputation."""
        try:
            from sklearn.impute import KNNImputer

            data_clean, _ = handle_missing_data(data_with_missing, strategy="knn", n_neighbors=3)
            assert not np.any(np.isnan(data_clean.X))
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_missing_indicator(self, data_with_missing):
        """Test missing indicator creation."""
        data_clean, mask = handle_missing_data(
            data_with_missing, strategy="median", missing_indicator=True
        )

        assert mask is not None
        assert mask.shape == data_with_missing.X.shape
        assert mask.dtype == bool

    def test_missing_y_and_zeta(self):
        """Test handling missing survival times and events."""
        X = np.random.randn(20, 3)
        y = np.random.exponential(2, 20)
        y[0] = np.nan  # Missing survival time
        zeta = np.random.binomial(1, 0.7, 20).astype(float)
        zeta[1] = np.nan  # Missing event indicator

        data = SurvivalDataset(X=X, y=y, zeta=zeta)

        data_clean, _ = handle_missing_data(data, strategy="median")

        assert not np.any(np.isnan(data_clean.y))
        assert not np.any(np.isnan(data_clean.zeta))


# ============================================================================
# Tests for train_test_split_survival
# ============================================================================


class TestTrainTestSplitSurvival:
    """Tests for survival data splitting."""

    def test_basic_split(self, simple_survival_data):
        """Test basic train/test split."""
        train, test = train_test_split_survival(
            simple_survival_data, test_size=0.3, random_state=42
        )

        # Check sizes
        assert len(train.y) == 70
        assert len(test.y) == 30

        # Check no overlap
        assert len(train.y) + len(test.y) == len(simple_survival_data.y)

    def test_event_stratification(self, simple_survival_data):
        """Test stratification by event status."""
        train, test = train_test_split_survival(
            simple_survival_data, test_size=0.3, stratify_by="event", random_state=42
        )

        # Event rates should be similar
        train_rate = train.zeta.mean()
        test_rate = test.zeta.mean()

        assert abs(train_rate - test_rate) < 0.1

    def test_time_stratification(self, simple_survival_data):
        """Test stratification by time."""
        train, test = train_test_split_survival(
            simple_survival_data, test_size=0.3, stratify_by="time", random_state=42
        )

        # Time distributions should be similar
        train_median = np.median(train.y)
        test_median = np.median(test.y)
        overall_median = np.median(simple_survival_data.y)

        assert abs(train_median - overall_median) < 2.0
        assert abs(test_median - overall_median) < 2.0

    def test_both_stratification(self, simple_survival_data):
        """Test stratification by both event and time."""
        train, test = train_test_split_survival(
            simple_survival_data, test_size=0.3, stratify_by="both", random_state=42
        )

        # Should maintain both distributions
        assert abs(train.zeta.mean() - test.zeta.mean()) < 0.15

    def test_no_shuffle(self, simple_survival_data):
        """Test splitting without shuffling."""
        train, test = train_test_split_survival(
            simple_survival_data, test_size=0.3, shuffle=False, stratify_by=None
        )

        # Should take first 70% for train
        np.testing.assert_array_equal(train.X, simple_survival_data.X[:70])

    def test_absolute_sizes(self, simple_survival_data):
        """Test with absolute train/test sizes."""
        train, test = train_test_split_survival(
            simple_survival_data, train_size=60, test_size=30, random_state=42
        )

        assert len(train.y) == 60
        assert len(test.y) == 30

    def test_invalid_sizes(self, simple_survival_data):
        """Test error for invalid sizes."""
        with pytest.raises(ValueError):
            train_test_split_survival(
                simple_survival_data,
                train_size=80,
                test_size=40,  # Total > 100
            )


# ============================================================================
# Tests for Additional Functions
# ============================================================================


class TestCreateTimeDependentFeatures:
    """Tests for time-dependent features."""

    def test_polynomial_features(self, simple_survival_data):
        """Test polynomial time features."""
        data_td = create_time_dependent_features(
            simple_survival_data, feature_type="polynomial", degree=2
        )

        # Should add 2 new features (t and t^2)
        assert data_td.X.shape[1] == simple_survival_data.X.shape[1] + 2

    def test_spline_features(self, simple_survival_data):
        """Test spline features."""
        data_td = create_time_dependent_features(
            simple_survival_data, feature_type="spline", degree=3
        )

        assert data_td.X.shape[1] > simple_survival_data.X.shape[1]

    def test_step_features(self, simple_survival_data):
        """Test step function features."""
        time_points = np.quantile(simple_survival_data.y, [0.25, 0.5, 0.75])
        data_td = create_time_dependent_features(
            simple_survival_data, time_points=time_points, feature_type="step"
        )

        # Should add 3 binary features
        assert data_td.X.shape[1] == simple_survival_data.X.shape[1] + 3


class TestDiscretizeTime:
    """Tests for time discretization."""

    def test_quantile_discretization(self, simple_survival_data):
        """Test quantile-based discretization."""
        data_disc, intervals = discretize_time(
            simple_survival_data, n_intervals=5, strategy="quantile"
        )

        assert len(intervals) == 6  # n_intervals + 1
        assert len(np.unique(data_disc.y)) <= 5

    def test_uniform_discretization(self, simple_survival_data):
        """Test uniform discretization."""
        data_disc, intervals = discretize_time(
            simple_survival_data, n_intervals=5, strategy="uniform"
        )

        # Intervals should be evenly spaced
        widths = np.diff(intervals)
        assert np.allclose(widths, widths[0])


class TestBalanceCensoring:
    """Tests for censoring balance."""

    def test_subsample_method(self, imbalanced_data):
        """Test subsampling for balance."""
        data_balanced = balance_censoring(
            imbalanced_data, target_rate=0.5, method="subsample", random_state=42
        )

        # Event rate should be closer to target
        assert abs(data_balanced.zeta.mean() - 0.5) < 0.1

        # Should have fewer samples
        assert len(data_balanced.y) <= len(imbalanced_data.y)

    def test_synthetic_method(self, imbalanced_data):
        """Test synthetic sample generation."""
        data_balanced = balance_censoring(
            imbalanced_data, target_rate=0.5, method="synthetic", random_state=42
        )

        # Event rate should be closer to target
        assert abs(data_balanced.zeta.mean() - 0.5) < 0.15

    def test_weight_method(self, imbalanced_data):
        """Test sample weighting."""
        data, weights = balance_censoring(imbalanced_data, target_rate=0.5, method="weight")

        # Should return same data with weights
        assert data is imbalanced_data
        assert len(weights) == len(data.y)
        assert np.all(weights > 0)

        # Weights should be normalized
        assert np.abs(weights.mean() - 1.0) < 0.1


class TestValidateSurvivalData:
    """Tests for data validation."""

    def test_valid_data(self, simple_survival_data):
        """Test validation of valid data."""
        is_valid, diagnostics = validate_survival_data(simple_survival_data, verbose=False)

        assert is_valid is True
        assert len(diagnostics["issues"]) == 0

    def test_invalid_times(self):
        """Test detection of invalid survival times."""
        X = np.random.randn(10, 3)
        y = np.array([1, 2, -1, 3, 4, 5, 6, 7, 8, 9])  # Negative time
        zeta = np.ones(10)

        data = SurvivalDataset(X=X, y=y, zeta=zeta)

        is_valid, diagnostics = validate_survival_data(data, verbose=False)

        assert is_valid is False
        assert any("non-positive" in issue for issue in diagnostics["issues"])

    def test_invalid_events(self):
        """Test detection of invalid event indicators."""
        X = np.random.randn(10, 3)
        y = np.random.exponential(2, 10)
        zeta = np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0])  # Invalid value 2

        data = SurvivalDataset(X=X, y=y, zeta=zeta)

        is_valid, diagnostics = validate_survival_data(data, verbose=False)

        assert is_valid is False
        assert any("Invalid event" in issue for issue in diagnostics["issues"])

    def test_zero_variance_detection(self, simple_survival_data):
        """Test zero variance feature detection."""
        # Add zero variance feature
        X_new = np.column_stack([simple_survival_data.X, np.ones(len(simple_survival_data.y))])
        data = SurvivalDataset(X=X_new, y=simple_survival_data.y, zeta=simple_survival_data.zeta)

        is_valid, diagnostics = validate_survival_data(
            data, remove_zero_variance=True, verbose=False
        )

        assert "zero_variance_features" in diagnostics
        assert 5 in diagnostics["zero_variance_features"]  # Last feature


class TestComputeFeatureImportance:
    """Tests for feature importance computation."""

    def test_coefficient_importance(self, simple_survival_data):
        """Test coefficient-based importance."""
        beta = np.array([0.5, -0.3, 0.8, 0.1, -0.2])

        importance = compute_feature_importance(simple_survival_data, beta, method="coefficient")

        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 5
        assert importance.iloc[0]["importance"] == 0.8  # Largest absolute value

    def test_permutation_importance(self, simple_survival_data):
        """Test permutation importance."""
        beta = np.random.randn(5)

        importance = compute_feature_importance(
            simple_survival_data, beta, method="permutation", n_permutations=10, random_state=42
        )

        assert len(importance) == 5
        assert "relative_importance" in importance.columns

    def test_univariate_importance(self, simple_survival_data):
        """Test univariate importance."""
        beta = np.zeros(5)  # Beta not used for univariate

        importance = compute_feature_importance(simple_survival_data, beta, method="univariate")

        assert len(importance) == 5
        assert np.all(importance["importance"] >= 0)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Generate data with issues
        np.random.seed(42)
        X = np.random.randn(200, 10)
        X[X < -2] = np.nan  # Missing values
        X[0:5, :] = 10  # Outliers
        y = np.random.exponential(2, 200)
        zeta = np.random.binomial(1, 0.3, 200)  # Imbalanced

        data = SurvivalDataset(X=X, y=y, zeta=zeta)

        # 1. Validate
        is_valid, _ = validate_survival_data(data, verbose=False)
        assert not is_valid

        # 2. Handle missing data
        data, _ = handle_missing_data(data, strategy="median")

        # 3. Detect and remove outliers
        outliers = detect_outliers(data, method="mad")
        data_clean = SurvivalDataset(
            X=data.X[~outliers], y=data.y[~outliers], zeta=data.zeta[~outliers]
        )

        # 4. Balance censoring
        data_balanced = balance_censoring(data_clean, target_rate=0.5, method="subsample")

        # 5. Scale features
        scaler = SurvivalStandardScaler(robust=True)
        data_scaled = scaler.fit_transform(data_balanced)

        # 6. Split data
        train, test = train_test_split_survival(data_scaled, test_size=0.3, stratify_by="event")

        # Final validation
        is_valid, _ = validate_survival_data(train, verbose=False)
        assert is_valid

        # Check properties
        assert not np.any(np.isnan(train.X))
        assert abs(train.zeta.mean() - 0.5) < 0.2
        assert np.abs(train.X.mean()) < 0.5  # Roughly centered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
