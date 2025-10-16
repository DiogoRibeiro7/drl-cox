"""
Preprocessing utilities for survival analysis with DRL-Cox.

This module provides survival-specific data preprocessing functions that
properly handle censoring, missing data, and stratification while preserving
the SurvivalDataset structure.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union, Dict, Any, Literal
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import median_abs_deviation

from .drl_cox import SurvivalDataset

__all__ = [
    "SurvivalStandardScaler",
    "detect_outliers",
    "handle_missing_data",
    "train_test_split_survival",
    "create_time_dependent_features",
    "discretize_time",
    "balance_censoring",
    "validate_survival_data",
    "compute_feature_importance",
]


@dataclass
class SurvivalStandardScaler:
    """
    StandardScaler specifically designed for survival data.
    
    Handles censoring properly by computing statistics using all data
    but with optional weighting by event status or inverse probability
    of censoring weights (IPCW).
    
    Parameters
    ----------
    with_mean : bool, default=True
        Whether to center the data
    with_std : bool, default=True
        Whether to scale to unit variance
    use_ipcw : bool, default=False
        Whether to use inverse probability of censoring weights
    robust : bool, default=False
        Whether to use robust statistics (median and MAD)
        
    Attributes
    ----------
    mean_ : np.ndarray
        Mean of each feature (computed during fit)
    scale_ : np.ndarray
        Standard deviation of each feature (computed during fit)
    n_features_in_ : int
        Number of features seen during fit
    feature_names_in_ : np.ndarray
        Names of features seen during fit (if available)
        
    Examples
    --------
    >>> from drl_cox import simulate_cox_data
    >>> from drl_cox.preprocessing import SurvivalStandardScaler
    >>> 
    >>> data = simulate_cox_data(n=200, d=10)
    >>> scaler = SurvivalStandardScaler(robust=True)
    >>> data_scaled = scaler.fit_transform(data)
    >>> 
    >>> # Transform new data
    >>> new_data = simulate_cox_data(n=50, d=10)
    >>> new_scaled = scaler.transform(new_data)
    """
    
    with_mean: bool = True
    with_std: bool = True
    use_ipcw: bool = False
    robust: bool = False
    
    def __post_init__(self):
        """Initialize internal state."""
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._ipcw_weights = None
    
    def fit(
        self,
        data: SurvivalDataset,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SurvivalStandardScaler":
        """
        Compute mean and std for scaling.
        
        Parameters
        ----------
        data : SurvivalDataset
            Training data
        sample_weight : np.ndarray, optional
            Sample weights for computing statistics
            
        Returns
        -------
        self : SurvivalStandardScaler
            Fitted scaler
        """
        X = data.X
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Compute IPCW weights if requested
        if self.use_ipcw:
            self._ipcw_weights = self._compute_ipcw_weights(data.y, data.zeta)
            if sample_weight is None:
                sample_weight = self._ipcw_weights
            else:
                sample_weight = sample_weight * self._ipcw_weights
        
        # Compute location and scale
        if self.robust:
            # Use median and MAD for robust scaling
            if self.with_mean:
                if sample_weight is not None:
                    # Weighted median
                    self.mean_ = np.array([
                        self._weighted_median(X[:, j], sample_weight)
                        for j in range(n_features)
                    ])
                else:
                    self.mean_ = np.median(X, axis=0)
            else:
                self.mean_ = np.zeros(n_features)
            
            if self.with_std:
                # Median Absolute Deviation
                self.scale_ = median_abs_deviation(X, axis=0, scale="normal")
                # Prevent division by zero
                self.scale_[self.scale_ == 0] = 1.0
            else:
                self.scale_ = np.ones(n_features)
        else:
            # Standard mean and std
            if self.with_mean:
                if sample_weight is not None:
                    self.mean_ = np.average(X, axis=0, weights=sample_weight)
                else:
                    self.mean_ = np.mean(X, axis=0)
            else:
                self.mean_ = np.zeros(n_features)
            
            if self.with_std:
                if sample_weight is not None:
                    # Weighted variance
                    var = np.average(
                        (X - self.mean_) ** 2,
                        axis=0,
                        weights=sample_weight
                    )
                    self.scale_ = np.sqrt(var)
                else:
                    self.scale_ = np.std(X, axis=0, ddof=0)
                # Prevent division by zero
                self.scale_[self.scale_ == 0] = 1.0
            else:
                self.scale_ = np.ones(n_features)
        
        return self
    
    def transform(self, data: SurvivalDataset) -> SurvivalDataset:
        """
        Scale features of data.
        
        Parameters
        ----------
        data : SurvivalDataset
            Data to transform
            
        Returns
        -------
        data_scaled : SurvivalDataset
            Transformed data with same y and zeta
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        
        X = data.X
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but scaler was fitted with "
                f"{self.n_features_in_} features."
            )
        
        # Scale features
        X_scaled = (X - self.mean_) / self.scale_
        
        # Return new SurvivalDataset
        return SurvivalDataset(X=X_scaled, y=data.y, zeta=data.zeta)
    
    def fit_transform(
        self,
        data: SurvivalDataset,
        sample_weight: Optional[np.ndarray] = None,
    ) -> SurvivalDataset:
        """
        Fit scaler and transform data.
        
        Parameters
        ----------
        data : SurvivalDataset
            Data to fit and transform
        sample_weight : np.ndarray, optional
            Sample weights
            
        Returns
        -------
        data_scaled : SurvivalDataset
            Transformed data
        """
        return self.fit(data, sample_weight).transform(data)
    
    def inverse_transform(self, data: SurvivalDataset) -> SurvivalDataset:
        """
        Inverse transform scaled data back to original scale.
        
        Parameters
        ----------
        data : SurvivalDataset
            Scaled data
            
        Returns
        -------
        data_original : SurvivalDataset
            Data in original scale
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        
        X_original = data.X * self.scale_ + self.mean_
        
        return SurvivalDataset(X=X_original, y=data.y, zeta=data.zeta)
    
    def _compute_ipcw_weights(
        self,
        y: np.ndarray,
        zeta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Inverse Probability of Censoring Weights.
        
        Uses Kaplan-Meier estimate of censoring distribution.
        """
        # Sort by time
        order = np.argsort(y)
        y_sorted = y[order]
        zeta_sorted = zeta[order]
        
        # Kaplan-Meier for censoring (flip event indicator)
        n = len(y)
        weights = np.ones(n)
        
        # Compute survival function for censoring
        unique_times = np.unique(y_sorted)
        G_t = 1.0  # Censoring survival at time t
        
        for t in unique_times:
            at_risk = np.sum(y_sorted >= t)
            n_censored = np.sum((y_sorted == t) & (zeta_sorted == 0))
            
            if at_risk > 0:
                # Update censoring survival
                G_t_prev = G_t
                G_t *= (1 - n_censored / at_risk)
                
                # Set weights for observations at time t
                mask = y == t
                if zeta[mask].any():  # Events
                    weights[mask & (zeta == 1)] = 1.0 / G_t_prev
                if (~zeta[mask]).any():  # Censored
                    weights[mask & (zeta == 0)] = 1.0 / G_t
        
        # Normalize weights
        weights = weights * len(weights) / weights.sum()
        
        return weights
    
    def _weighted_median(
        self,
        x: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute weighted median."""
        sorted_idx = np.argsort(x)
        sorted_x = x[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        cumsum = np.cumsum(sorted_weights)
        cutoff = cumsum[-1] / 2.0
        
        return sorted_x[cumsum >= cutoff][0]


def detect_outliers(
    data: SurvivalDataset,
    method: Literal["iqr", "mad", "isolation", "lof"] = "mad",
    threshold: float = 3.0,
    contamination: float = 0.1,
    return_scores: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Detect outliers in survival data using robust methods.
    
    Parameters
    ----------
    data : SurvivalDataset
        Input data
    method : {"iqr", "mad", "isolation", "lof"}, default="mad"
        Outlier detection method:
        - "iqr": Interquartile range method
        - "mad": Median Absolute Deviation
        - "isolation": Isolation Forest
        - "lof": Local Outlier Factor
    threshold : float, default=3.0
        Threshold for outlier detection (for iqr and mad)
    contamination : float, default=0.1
        Expected proportion of outliers (for isolation and lof)
    return_scores : bool, default=False
        Whether to return outlier scores along with masks
        
    Returns
    -------
    outlier_mask : np.ndarray
        Boolean array indicating outliers (True = outlier)
    scores : np.ndarray, optional
        Outlier scores if return_scores=True
        
    Examples
    --------
    >>> from drl_cox import simulate_cox_data
    >>> from drl_cox.preprocessing import detect_outliers
    >>> 
    >>> data = simulate_cox_data(n=200, d=10)
    >>> outliers = detect_outliers(data, method="mad", threshold=3.0)
    >>> print(f"Found {outliers.sum()} outliers")
    >>> 
    >>> # Get outlier scores
    >>> outliers, scores = detect_outliers(data, return_scores=True)
    >>> 
    >>> # Clean data
    >>> clean_data = SurvivalDataset(
    ...     X=data.X[~outliers],
    ...     y=data.y[~outliers],
    ...     zeta=data.zeta[~outliers]
    ... )
    """
    X = data.X
    n_samples, n_features = X.shape
    
    if method == "iqr":
        # IQR-based detection
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Check each feature
        outlier_mask = np.any(
            (X < lower_bound) | (X > upper_bound),
            axis=1
        )
        
        if return_scores:
            # Distance from bounds
            scores = np.maximum(
                np.max((lower_bound - X) / IQR, axis=1),
                np.max((X - upper_bound) / IQR, axis=1)
            )
            scores[scores < 0] = 0
            return outlier_mask, scores
            
    elif method == "mad":
        # Median Absolute Deviation
        median = np.median(X, axis=0)
        mad = median_abs_deviation(X, axis=0, scale="normal")
        
        # Prevent division by zero
        mad[mad == 0] = 1e-10
        
        # Modified Z-scores
        z_scores = np.abs((X - median) / mad)
        
        outlier_mask = np.any(z_scores > threshold, axis=1)
        
        if return_scores:
            scores = np.max(z_scores, axis=1)
            return outlier_mask, scores
            
    elif method == "isolation":
        # Isolation Forest
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "IsolationForest requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        predictions = iso_forest.fit_predict(X)
        outlier_mask = predictions == -1
        
        if return_scores:
            scores = -iso_forest.score_samples(X)
            return outlier_mask, scores
            
    elif method == "lof":
        # Local Outlier Factor
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            raise ImportError(
                "LocalOutlierFactor requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        lof = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=min(20, n_samples - 1),
        )
        predictions = lof.fit_predict(X)
        outlier_mask = predictions == -1
        
        if return_scores:
            scores = -lof.negative_outlier_factor_
            return outlier_mask, scores
            
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Choose from: 'iqr', 'mad', 'isolation', 'lof'"
        )
    
    return outlier_mask


def handle_missing_data(
    data: SurvivalDataset,
    strategy: Literal["drop", "mean", "median", "forward", "mice", "knn"] = "median",
    missing_indicator: bool = False,
    n_neighbors: int = 5,
    max_iter: int = 10,
) -> Tuple[SurvivalDataset, Optional[np.ndarray]]:
    """
    Handle missing data in survival datasets with appropriate imputation.
    
    Parameters
    ----------
    data : SurvivalDataset
        Input data with potential missing values
    strategy : str, default="median"
        Imputation strategy:
        - "drop": Remove samples with missing values
        - "mean": Replace with feature mean
        - "median": Replace with feature median
        - "forward": Forward fill (for time-series)
        - "mice": Multiple Imputation by Chained Equations
        - "knn": K-Nearest Neighbors imputation
    missing_indicator : bool, default=False
        Whether to add binary indicators for missing values
    n_neighbors : int, default=5
        Number of neighbors for KNN imputation
    max_iter : int, default=10
        Maximum iterations for MICE
        
    Returns
    -------
    data_imputed : SurvivalDataset
        Data with missing values handled
    missing_mask : np.ndarray, optional
        Binary indicators if missing_indicator=True
        
    Examples
    --------
    >>> import numpy as np
    >>> from drl_cox import SurvivalDataset
    >>> from drl_cox.preprocessing import handle_missing_data
    >>> 
    >>> # Create data with missing values
    >>> X = np.random.randn(100, 5)
    >>> X[X < -2] = np.nan  # Introduce missing values
    >>> data = SurvivalDataset(
    ...     X=X,
    ...     y=np.random.exponential(2, 100),
    ...     zeta=np.random.binomial(1, 0.7, 100)
    ... )
    >>> 
    >>> # Impute missing values
    >>> data_clean, mask = handle_missing_data(
    ...     data,
    ...     strategy="knn",
    ...     missing_indicator=True
    ... )
    """
    X = data.X.copy()
    y = data.y.copy()
    zeta = data.zeta.copy()
    
    # Check for missing values
    X_missing = np.isnan(X)
    y_missing = np.isnan(y)
    zeta_missing = np.isnan(zeta)
    
    n_missing_X = X_missing.sum()
    n_missing_y = y_missing.sum()
    n_missing_zeta = zeta_missing.sum()
    
    if n_missing_X == 0 and n_missing_y == 0 and n_missing_zeta == 0:
        warnings.warn("No missing values found in the data.")
        if missing_indicator:
            return data, np.zeros_like(X_missing, dtype=bool)
        return data, None
    
    # Store missing indicators before imputation
    missing_mask_original = X_missing.copy() if missing_indicator else None
    
    # Handle missing survival times and event indicators first
    if n_missing_y > 0 or n_missing_zeta > 0:
        if strategy == "drop":
            # Remove samples with missing y or zeta
            keep_mask = ~(y_missing | zeta_missing)
            X = X[keep_mask]
            y = y[keep_mask]
            zeta = zeta[keep_mask]
            X_missing = X_missing[keep_mask]
        else:
            # For survival times, use median imputation
            if n_missing_y > 0:
                y[y_missing] = np.nanmedian(y)
            # For event indicators, impute based on event rate
            if n_missing_zeta > 0:
                event_rate = np.nanmean(zeta)
                # Use random imputation based on event rate
                n_missing = zeta_missing.sum()
                zeta[zeta_missing] = np.random.binomial(1, event_rate, n_missing)
    
    # Handle missing features
    if strategy == "drop":
        # Remove samples with any missing features
        keep_mask = ~np.any(X_missing, axis=1)
        X = X[keep_mask]
        y = y[keep_mask]
        zeta = zeta[keep_mask]
        
    elif strategy == "mean":
        # Simple mean imputation
        for j in range(X.shape[1]):
            if np.any(X_missing[:, j]):
                X[X_missing[:, j], j] = np.nanmean(X[:, j])
                
    elif strategy == "median":
        # Simple median imputation
        for j in range(X.shape[1]):
            if np.any(X_missing[:, j]):
                X[X_missing[:, j], j] = np.nanmedian(X[:, j])
                
    elif strategy == "forward":
        # Forward fill (useful for time-series)
        for j in range(X.shape[1]):
            mask = X_missing[:, j]
            if np.any(mask):
                # Forward fill
                valid_idx = np.where(~mask)[0]
                if len(valid_idx) > 0:
                    for i in np.where(mask)[0]:
                        # Find previous valid value
                        prev_valid = valid_idx[valid_idx < i]
                        if len(prev_valid) > 0:
                            X[i, j] = X[prev_valid[-1], j]
                        else:
                            # No previous value, use next
                            next_valid = valid_idx[valid_idx > i]
                            if len(next_valid) > 0:
                                X[i, j] = X[next_valid[0], j]
                            else:
                                # No valid values, use median
                                X[i, j] = np.nanmedian(X[:, j])
                                
    elif strategy == "mice":
        # Multiple Imputation by Chained Equations
        X = _mice_imputation(X, max_iter=max_iter)
        
    elif strategy == "knn":
        # K-Nearest Neighbors imputation
        try:
            from sklearn.impute import KNNImputer
        except ImportError:
            warnings.warn(
                "KNN imputation requires scikit-learn. "
                "Falling back to median imputation."
            )
            strategy = "median"
            return handle_missing_data(
                data, strategy="median", missing_indicator=missing_indicator
            )
        
        imputer = KNNImputer(n_neighbors=n_neighbors)
        X = imputer.fit_transform(X)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Create imputed dataset
    data_imputed = SurvivalDataset(X=X, y=y, zeta=zeta)
    
    # Add missing indicators if requested
    if missing_indicator:
        if strategy == "drop":
            # Adjust mask for dropped samples
            missing_mask_original = missing_mask_original[
                ~np.any(X_missing, axis=1)
            ]
        return data_imputed, missing_mask_original
    
    return data_imputed, None


def train_test_split_survival(
    data: SurvivalDataset,
    test_size: Union[float, int] = 0.25,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[int] = None,
    stratify_by: Literal["event", "time", "both", None] = "event",
    shuffle: bool = True,
) -> Tuple[SurvivalDataset, SurvivalDataset]:
    """
    Split survival data into train and test sets with proper stratification.
    
    Parameters
    ----------
    data : SurvivalDataset
        Data to split
    test_size : float or int, default=0.25
        Proportion or absolute number for test set
    train_size : float or int, optional
        Proportion or absolute number for train set
    random_state : int, optional
        Random seed for reproducibility
    stratify_by : {"event", "time", "both", None}, default="event"
        Stratification strategy:
        - "event": Stratify by event status
        - "time": Stratify by time quartiles
        - "both": Stratify by both event and time
        - None: No stratification
    shuffle : bool, default=True
        Whether to shuffle before splitting
        
    Returns
    -------
    train_data : SurvivalDataset
        Training set
    test_data : SurvivalDataset
        Test set
        
    Examples
    --------
    >>> from drl_cox import simulate_cox_data
    >>> from drl_cox.preprocessing import train_test_split_survival
    >>> 
    >>> data = simulate_cox_data(n=300, d=10)
    >>> 
    >>> # Split with event stratification
    >>> train, test = train_test_split_survival(
    ...     data,
    ...     test_size=0.3,
    ...     stratify_by="event",
    ...     random_state=42
    ... )
    >>> 
    >>> # Check event rates are similar
    >>> train_event_rate = train.zeta.mean()
    >>> test_event_rate = test.zeta.mean()
    >>> print(f"Train events: {train_event_rate:.2%}")
    >>> print(f"Test events: {test_event_rate:.2%}")
    """
    n_samples = len(data.y)
    
    # Determine split sizes
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size
    
    if train_size is not None:
        if isinstance(train_size, float):
            n_train = int(n_samples * train_size)
        else:
            n_train = train_size
    else:
        n_train = n_samples - n_test
    
    if n_train + n_test > n_samples:
        raise ValueError(
            f"train_size + test_size = {n_train + n_test} "
            f"is larger than data size = {n_samples}"
        )
    
    # Create indices
    indices = np.arange(n_samples)
    
    # Set random seed
    rng = np.random.RandomState(random_state)
    
    if stratify_by is None:
        # Simple random split
        if shuffle:
            rng.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:n_train + n_test]
        
    elif stratify_by == "event":
        # Stratify by event status
        event_idx = np.where(data.zeta == 1)[0]
        censor_idx = np.where(data.zeta == 0)[0]
        
        # Proportional split
        n_event_test = int(len(event_idx) * test_size)
        n_censor_test = n_test - n_event_test
        
        if shuffle:
            rng.shuffle(event_idx)
            rng.shuffle(censor_idx)
        
        test_idx = np.concatenate([
            event_idx[:n_event_test],
            censor_idx[:n_censor_test]
        ])
        train_idx = np.concatenate([
            event_idx[n_event_test:],
            censor_idx[n_censor_test:]
        ])
        
    elif stratify_by == "time":
        # Stratify by time quartiles
        time_quartiles = np.quantile(data.y, [0.25, 0.5, 0.75])
        time_groups = np.digitize(data.y, time_quartiles)
        
        train_idx = []
        test_idx = []
        
        for group in np.unique(time_groups):
            group_idx = np.where(time_groups == group)[0]
            n_group_test = int(len(group_idx) * test_size)
            
            if shuffle:
                rng.shuffle(group_idx)
            
            test_idx.extend(group_idx[:n_group_test])
            train_idx.extend(group_idx[n_group_test:])
        
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        
    elif stratify_by == "both":
        # Stratify by both event and time
        time_quartiles = np.quantile(data.y, [0.5])
        time_groups = np.digitize(data.y, time_quartiles)
        
        # Create combined strata
        strata = time_groups * 2 + data.zeta
        
        train_idx = []
        test_idx = []
        
        for stratum in np.unique(strata):
            stratum_idx = np.where(strata == stratum)[0]
            n_stratum_test = max(1, int(len(stratum_idx) * test_size))
            
            if shuffle:
                rng.shuffle(stratum_idx)
            
            test_idx.extend(stratum_idx[:n_stratum_test])
            train_idx.extend(stratum_idx[n_stratum_test:])
        
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    
    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")
    
    # Create train and test datasets
    train_data = SurvivalDataset(
        X=data.X[train_idx],
        y=data.y[train_idx],
        zeta=data.zeta[train_idx]
    )
    
    test_data = SurvivalDataset(
        X=data.X[test_idx],
        y=data.y[test_idx],
        zeta=data.zeta[test_idx]
    )
    
    return train_data, test_data


def create_time_dependent_features(
    data: SurvivalDataset,
    time_points: Optional[np.ndarray] = None,
    feature_type: Literal["polynomial", "spline", "step"] = "polynomial",
    degree: int = 2,
) -> SurvivalDataset:
    """
    Create time-dependent features for survival analysis.
    
    Parameters
    ----------
    data : SurvivalDataset
        Input data
    time_points : np.ndarray, optional
        Time points for creating features (default: quartiles)
    feature_type : {"polynomial", "spline", "step"}, default="polynomial"
        Type of time-dependent features
    degree : int, default=2
        Degree for polynomial features
        
    Returns
    -------
    data_augmented : SurvivalDataset
        Data with time-dependent features added
        
    Examples
    --------
    >>> data = simulate_cox_data(n=200, d=5)
    >>> data_td = create_time_dependent_features(
    ...     data,
    ...     feature_type="polynomial",
    ...     degree=2
    ... )
    >>> print(f"Original features: {data.X.shape[1]}")
    >>> print(f"Augmented features: {data_td.X.shape[1]}")
    """
    X = data.X
    y = data.y
    
    if time_points is None:
        # Use quartiles as default time points
        time_points = np.quantile(y, [0.25, 0.5, 0.75])
    
    if feature_type == "polynomial":
        # Add polynomial time features
        time_features = []
        for d in range(1, degree + 1):
            time_features.append(y.reshape(-1, 1) ** d)
        
        # Normalize time features
        time_features = np.hstack(time_features)
        time_features = (time_features - time_features.mean(axis=0)) / (
            time_features.std(axis=0) + 1e-10
        )
        
        # Concatenate with original features
        X_augmented = np.hstack([X, time_features])
        
    elif feature_type == "spline":
        # Create B-spline basis
        from scipy.interpolate import BSpline
        
        # Create knots
        knots = np.linspace(y.min(), y.max(), degree + 1)
        
        # Create basis functions
        time_features = []
        for i in range(len(knots) - 1):
            # Simple linear spline basis
            basis = np.maximum(0, y - knots[i])
            basis = np.minimum(basis, knots[i + 1] - knots[i])
            time_features.append(basis.reshape(-1, 1))
        
        time_features = np.hstack(time_features)
        X_augmented = np.hstack([X, time_features])
        
    elif feature_type == "step":
        # Create step functions at time points
        time_features = []
        for t in time_points:
            time_features.append((y >= t).astype(float).reshape(-1, 1))
        
        time_features = np.hstack(time_features)
        X_augmented = np.hstack([X, time_features])
        
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    return SurvivalDataset(X=X_augmented, y=data.y, zeta=data.zeta)


def discretize_time(
    data: SurvivalDataset,
    n_intervals: int = 10,
    strategy: Literal["quantile", "uniform", "kmeans"] = "quantile",
) -> Tuple[SurvivalDataset, np.ndarray]:
    """
    Discretize survival times into intervals.
    
    Useful for discrete-time survival models or creating time bins.
    
    Parameters
    ----------
    data : SurvivalDataset
        Input data
    n_intervals : int, default=10
        Number of time intervals
    strategy : {"quantile", "uniform", "kmeans"}, default="quantile"
        Discretization strategy
        
    Returns
    -------
    data_discrete : SurvivalDataset
        Data with discretized times
    intervals : np.ndarray
        Interval boundaries
        
    Examples
    --------
    >>> data = simulate_cox_data(n=200, d=5)
    >>> data_disc, intervals = discretize_time(
    ...     data,
    ...     n_intervals=5,
    ...     strategy="quantile"
    ... )
    >>> print(f"Time intervals: {intervals}")
    """
    y = data.y
    
    if strategy == "quantile":
        # Use quantiles for equal-frequency bins
        quantiles = np.linspace(0, 100, n_intervals + 1)
        intervals = np.percentile(y, quantiles)
        intervals[0] = 0  # Start from 0
        intervals[-1] = y.max() * 1.01  # Ensure max is included
        
    elif strategy == "uniform":
        # Equal-width bins
        intervals = np.linspace(0, y.max() * 1.01, n_intervals + 1)
        
    elif strategy == "kmeans":
        # Use KMeans for adaptive binning
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            warnings.warn(
                "KMeans requires scikit-learn. "
                "Falling back to quantile strategy."
            )
            return discretize_time(data, n_intervals, strategy="quantile")
        
        kmeans = KMeans(n_clusters=n_intervals, random_state=42)
        clusters = kmeans.fit_predict(y.reshape(-1, 1))
        
        # Get interval boundaries from cluster centers
        centers = kmeans.cluster_centers_.flatten()
        centers_sorted = np.sort(centers)
        
        intervals = [0]
        for i in range(len(centers_sorted) - 1):
            intervals.append((centers_sorted[i] + centers_sorted[i + 1]) / 2)
        intervals.append(y.max() * 1.01)
        intervals = np.array(intervals)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Discretize times
    y_discrete = np.digitize(y, intervals) - 1
    y_discrete = np.clip(y_discrete, 0, n_intervals - 1)
    
    # Use interval midpoints as new times
    interval_midpoints = (intervals[:-1] + intervals[1:]) / 2
    y_new = interval_midpoints[y_discrete]
    
    data_discrete = SurvivalDataset(X=data.X, y=y_new, zeta=data.zeta)
    
    return data_discrete, intervals


def balance_censoring(
    data: SurvivalDataset,
    target_rate: float = 0.5,
    method: Literal["subsample", "synthetic", "weight"] = "subsample",
    random_state: Optional[int] = None,
) -> Union[SurvivalDataset, Tuple[SurvivalDataset, np.ndarray]]:
    """
    Balance censoring rate in survival data.
    
    Parameters
    ----------
    data : SurvivalDataset
        Input data
    target_rate : float, default=0.5
        Target event rate (1 - censoring rate)
    method : {"subsample", "synthetic", "weight"}, default="subsample"
        Balancing method:
        - "subsample": Undersample majority class
        - "synthetic": Generate synthetic samples (SMOTE-like)
        - "weight": Return sample weights
    random_state : int, optional
        Random seed
        
    Returns
    -------
    data_balanced : SurvivalDataset
        Balanced dataset (for subsample and synthetic)
    weights : np.ndarray
        Sample weights (for weight method)
        
    Examples
    --------
    >>> data = simulate_cox_data(n=200, d=5, censor_rate=0.7)
    >>> print(f"Original event rate: {data.zeta.mean():.2%}")
    >>> 
    >>> # Balance by subsampling
    >>> data_balanced = balance_censoring(
    ...     data,
    ...     target_rate=0.5,
    ...     method="subsample"
    ... )
    >>> print(f"Balanced event rate: {data_balanced.zeta.mean():.2%}")
    """
    current_rate = data.zeta.mean()
    n_samples = len(data.y)
    
    rng = np.random.RandomState(random_state)
    
    if method == "subsample":
        # Undersample the majority class
        event_idx = np.where(data.zeta == 1)[0]
        censor_idx = np.where(data.zeta == 0)[0]
        
        n_events = len(event_idx)
        n_censored = len(censor_idx)
        
        if current_rate < target_rate:
            # Need to reduce censored observations
            n_censored_keep = int(n_events * (1 - target_rate) / target_rate)
            if n_censored_keep < n_censored:
                censor_idx_keep = rng.choice(
                    censor_idx, n_censored_keep, replace=False
                )
                keep_idx = np.concatenate([event_idx, censor_idx_keep])
            else:
                keep_idx = np.arange(n_samples)
        else:
            # Need to reduce event observations
            n_events_keep = int(n_censored * target_rate / (1 - target_rate))
            if n_events_keep < n_events:
                event_idx_keep = rng.choice(
                    event_idx, n_events_keep, replace=False
                )
                keep_idx = np.concatenate([event_idx_keep, censor_idx])
            else:
                keep_idx = np.arange(n_samples)
        
        # Sort to maintain order
        keep_idx = np.sort(keep_idx)
        
        return SurvivalDataset(
            X=data.X[keep_idx],
            y=data.y[keep_idx],
            zeta=data.zeta[keep_idx]
        )
        
    elif method == "synthetic":
        # Generate synthetic samples using SMOTE-like approach
        event_idx = np.where(data.zeta == 1)[0]
        censor_idx = np.where(data.zeta == 0)[0]
        
        n_events = len(event_idx)
        n_censored = len(censor_idx)
        
        if current_rate < target_rate:
            # Generate synthetic events
            n_synthetic = int(n_censored * target_rate / (1 - target_rate)) - n_events
            
            if n_synthetic > 0:
                synthetic_X, synthetic_y = _generate_synthetic_samples(
                    data.X[event_idx],
                    data.y[event_idx],
                    n_synthetic,
                    k_neighbors=5,
                    random_state=random_state,
                )
                
                # Combine with original data
                X_new = np.vstack([data.X, synthetic_X])
                y_new = np.concatenate([data.y, synthetic_y])
                zeta_new = np.concatenate([data.zeta, np.ones(n_synthetic)])
                
                return SurvivalDataset(X=X_new, y=y_new, zeta=zeta_new)
        else:
            # Generate synthetic censored observations
            n_synthetic = int(n_events * (1 - target_rate) / target_rate) - n_censored
            
            if n_synthetic > 0:
                synthetic_X, synthetic_y = _generate_synthetic_samples(
                    data.X[censor_idx],
                    data.y[censor_idx],
                    n_synthetic,
                    k_neighbors=5,
                    random_state=random_state,
                )
                
                # Combine with original data
                X_new = np.vstack([data.X, synthetic_X])
                y_new = np.concatenate([data.y, synthetic_y])
                zeta_new = np.concatenate([data.zeta, np.zeros(n_synthetic)])
                
                return SurvivalDataset(X=X_new, y=y_new, zeta=zeta_new)
        
        return data
        
    elif method == "weight":
        # Compute sample weights for balancing
        weights = np.ones(n_samples)
        
        # Weight inversely proportional to class frequency
        event_weight = 1.0 / (2 * current_rate) if current_rate > 0 else 1.0
        censor_weight = 1.0 / (2 * (1 - current_rate)) if current_rate < 1 else 1.0
        
        weights[data.zeta == 1] = event_weight
        weights[data.zeta == 0] = censor_weight
        
        # Normalize weights
        weights = weights * n_samples / weights.sum()
        
        return data, weights
        
    else:
        raise ValueError(f"Unknown method: {method}")


def validate_survival_data(
    data: SurvivalDataset,
    check_finite: bool = True,
    check_positive_times: bool = True,
    check_event_indicator: bool = True,
    remove_zero_variance: bool = False,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate survival data and return diagnostics.
    
    Parameters
    ----------
    data : SurvivalDataset
        Data to validate
    check_finite : bool, default=True
        Check for NaN/Inf values
    check_positive_times : bool, default=True
        Check that survival times are positive
    check_event_indicator : bool, default=True
        Check that event indicators are binary
    remove_zero_variance : bool, default=False
        Whether to identify zero-variance features
    verbose : bool, default=True
        Print validation results
        
    Returns
    -------
    is_valid : bool
        Whether data passes all checks
    diagnostics : dict
        Detailed diagnostics information
        
    Examples
    --------
    >>> data = simulate_cox_data(n=200, d=10)
    >>> is_valid, diagnostics = validate_survival_data(data)
    >>> if is_valid:
    ...     print("Data is valid!")
    >>> else:
    ...     print("Issues found:", diagnostics["issues"])
    """
    diagnostics = {
        "n_samples": len(data.y),
        "n_features": data.X.shape[1],
        "event_rate": data.zeta.mean(),
        "issues": [],
        "warnings": [],
    }
    
    is_valid = True
    
    # Check for finite values
    if check_finite:
        if np.any(~np.isfinite(data.X)):
            n_nan = np.isnan(data.X).sum()
            n_inf = np.isinf(data.X).sum()
            diagnostics["issues"].append(
                f"Non-finite values in X: {n_nan} NaN, {n_inf} Inf"
            )
            is_valid = False
        
        if np.any(~np.isfinite(data.y)):
            diagnostics["issues"].append("Non-finite values in survival times")
            is_valid = False
    
    # Check positive times
    if check_positive_times:
        if np.any(data.y <= 0):
            n_negative = (data.y <= 0).sum()
            diagnostics["issues"].append(
                f"{n_negative} non-positive survival times"
            )
            is_valid = False
    
    # Check event indicator
    if check_event_indicator:
        unique_zeta = np.unique(data.zeta)
        if not set(unique_zeta).issubset({0, 1}):
            diagnostics["issues"].append(
                f"Invalid event indicators: {unique_zeta}"
            )
            is_valid = False
    
    # Check for zero-variance features
    if remove_zero_variance:
        variances = np.var(data.X, axis=0)
        zero_var_features = np.where(variances < 1e-10)[0]
        if len(zero_var_features) > 0:
            diagnostics["warnings"].append(
                f"Zero-variance features: {zero_var_features.tolist()}"
            )
            diagnostics["zero_variance_features"] = zero_var_features
    
    # Additional diagnostics
    diagnostics["time_range"] = (data.y.min(), data.y.max())
    diagnostics["feature_ranges"] = [
        (data.X[:, j].min(), data.X[:, j].max())
        for j in range(data.X.shape[1])
    ]
    
    # Check for extreme event rate
    if diagnostics["event_rate"] < 0.1:
        diagnostics["warnings"].append(
            f"Low event rate: {diagnostics['event_rate']:.1%}"
        )
    elif diagnostics["event_rate"] > 0.9:
        diagnostics["warnings"].append(
            f"High event rate: {diagnostics['event_rate']:.1%}"
        )
    
    if verbose:
        print("Survival Data Validation Report")
        print("=" * 40)
        print(f"Samples: {diagnostics['n_samples']}")
        print(f"Features: {diagnostics['n_features']}")
        print(f"Event rate: {diagnostics['event_rate']:.1%}")
        print(f"Time range: [{diagnostics['time_range'][0]:.2f}, "
              f"{diagnostics['time_range'][1]:.2f}]")
        
        if diagnostics["issues"]:
            print("\n❌ Issues found:")
            for issue in diagnostics["issues"]:
                print(f"  - {issue}")
        
        if diagnostics["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in diagnostics["warnings"]:
                print(f"  - {warning}")
        
        if is_valid and not diagnostics["warnings"]:
            print("\n✅ Data is valid!")
    
    return is_valid, diagnostics


def compute_feature_importance(
    data: SurvivalDataset,
    beta: np.ndarray,
    method: Literal["coefficient", "permutation", "univariate"] = "coefficient",
    n_permutations: int = 100,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute feature importance for survival models.
    
    Parameters
    ----------
    data : SurvivalDataset
        Input data
    beta : np.ndarray
        Model coefficients
    method : {"coefficient", "permutation", "univariate"}, default="coefficient"
        Method for computing importance
    n_permutations : int, default=100
        Number of permutations (for permutation method)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    importance_df : pd.DataFrame
        Feature importance scores
        
    Examples
    --------
    >>> from drl_cox import fit_drl_cox
    >>> result = fit_drl_cox(data, epsilon=0.1)
    >>> importance = compute_feature_importance(
    ...     data,
    ...     result.beta,
    ...     method="permutation"
    ... )
    >>> print(importance.head())
    """
    from .metrics import concordance_index
    
    n_features = data.X.shape[1]
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    if method == "coefficient":
        # Simple coefficient magnitude
        importance = np.abs(beta)
        
    elif method == "permutation":
        # Permutation importance
        rng = np.random.RandomState(random_state)
        
        # Baseline score
        baseline_risk = data.X @ beta
        baseline_score = concordance_index(baseline_risk, data.y, data.zeta)
        
        importance = np.zeros(n_features)
        
        for j in range(n_features):
            scores = []
            
            for _ in range(n_permutations):
                # Permute feature j
                X_perm = data.X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                
                # Compute new score
                risk_perm = X_perm @ beta
                score_perm = concordance_index(risk_perm, data.y, data.zeta)
                scores.append(baseline_score - score_perm)
            
            importance[j] = np.mean(scores)
            
    elif method == "univariate":
        # Univariate concordance
        importance = np.zeros(n_features)
        
        for j in range(n_features):
            # Use single feature as risk score
            risk_univariate = data.X[:, j]
            score = concordance_index(risk_univariate, data.y, data.zeta)
            importance[j] = np.abs(score - 0.5)  # Distance from random
            
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "rank": np.argsort(-importance) + 1,
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    
    # Add relative importance
    max_importance = importance_df["importance"].max()
    if max_importance > 0:
        importance_df["relative_importance"] = (
            importance_df["importance"] / max_importance
        )
    else:
        importance_df["relative_importance"] = 0
    
    return importance_df


# ============================================================================
# Helper Functions (Private)
# ============================================================================


def _mice_imputation(
    X: np.ndarray,
    max_iter: int = 10,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Simple MICE implementation for multiple imputation.
    """
    rng = np.random.RandomState(random_state)
    X_filled = X.copy()
    n_features = X.shape[1]
    
    # Initial imputation with mean
    for j in range(n_features):
        mask = np.isnan(X[:, j])
        if np.any(mask):
            X_filled[mask, j] = np.nanmean(X[:, j])
    
    # Iterative imputation
    for _ in range(max_iter):
        for j in range(n_features):
            mask = np.isnan(X[:, j])
            if not np.any(mask):
                continue
            
            # Use other features to predict missing values
            X_train = X_filled[~mask]
            y_train = X_filled[~mask, j]
            X_test = X_filled[mask]
            
            # Simple linear regression
            # Remove the target column from features
            feature_cols = [i for i in range(n_features) if i != j]
            X_train_feat = X_train[:, feature_cols]
            X_test_feat = X_test[:, feature_cols]
            
            # Fit linear model (simplified)
            try:
                # Add small regularization for stability
                XtX = X_train_feat.T @ X_train_feat + 0.01 * np.eye(len(feature_cols))
                Xty = X_train_feat.T @ y_train
                beta = np.linalg.solve(XtX, Xty)
                
                # Predict missing values
                predictions = X_test_feat @ beta
                
                # Add small noise
                noise = rng.normal(0, 0.01, len(predictions))
                X_filled[mask, j] = predictions + noise
            except:
                # If regression fails, keep mean imputation
                pass
    
    return X_filled


def _generate_synthetic_samples(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    k_neighbors: int = 5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic samples using SMOTE-like approach.
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    k_neighbors = min(k_neighbors, n_samples - 1)
    
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_synthetic):
        # Random sample
        idx = rng.randint(0, n_samples)
        sample = X[idx]
        time = y[idx]
        
        # Find k nearest neighbors
        distances = np.sum((X - sample) ** 2, axis=1)
        neighbor_idx = np.argsort(distances)[1:k_neighbors + 1]
        
        # Random neighbor
        nn_idx = rng.choice(neighbor_idx)
        nn = X[nn_idx]
        nn_time = y[nn_idx]
        
        # Interpolate
        alpha = rng.random()
        synthetic_sample = sample + alpha * (nn - sample)
        synthetic_time = time + alpha * (nn_time - time)
        
        synthetic_X.append(synthetic_sample)
        synthetic_y.append(synthetic_time)
    
    return np.array(synthetic_X), np.array(synthetic_y)
