from __future__ import annotations
from typing import Iterable
import numpy as np

__all__ = ["inject_covariate_shift", "inject_outliers"]


def inject_covariate_shift(
    X: np.ndarray,
    *,
    feature_indices: Iterable[int],
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Replace selected features with samples from N(mean, std^2) to induce shift.

    Parameters
    ----------
    X : np.ndarray (N, d)
    feature_indices : Iterable[int]
        Indices of columns to replace.
    mean, std : float
        Parameters of replacement Gaussian.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy array.")
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    idx = np.array(list(feature_indices), dtype=int)
    if idx.size == 0:
        return Xs
    N = Xs.shape[0]
    Xs[:, idx] = rng.normal(loc=mean, scale=std, size=(N, idx.size))
    return Xs


def inject_outliers(
    X: np.ndarray,
    *,
    ratio: float = 0.1,
    severity_std: float = 3.0,
    seed: int | None = None,
) -> np.ndarray:
    """Inject Gaussian perturbations to a random subset of entries.

    Parameters
    ----------
    X : np.ndarray (N, d)
    ratio : float
        Proportion of rows to corrupt.
    severity_std : float
        Standard deviation of the additive noise.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy array.")
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("ratio must be in [0,1].")

    rng = np.random.default_rng(seed)
    Xs = X.copy()
    N, d = Xs.shape
    m = int(np.floor(ratio * N))
    if m == 0:
        return Xs
    rows = rng.choice(N, size=m, replace=False)
    noise = rng.normal(loc=0.0, scale=severity_std, size=(m, d))
    Xs[rows, :] = Xs[rows, :] + noise
    return Xs
