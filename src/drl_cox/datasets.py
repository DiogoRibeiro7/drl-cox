from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .drl_cox import SurvivalDataset  # re-use the dataclass type

__all__ = ["load_whas500_like_csv", "simulate_cox_data"]


def load_whas500_like_csv(path: str, *, y_col: str = "y", zeta_col: str = "zeta") -> SurvivalDataset:
    """Load a CSV with columns y (duration), zeta (event), and covariates x*.

    Parameters
    ----------
    path : str
        CSV path.
    y_col : str
        Column for durations.
    zeta_col : str
        Column for event indicator.
    """
    df = pd.read_csv(path)
    if y_col not in df.columns or zeta_col not in df.columns:
        raise ValueError(f"CSV must include columns '{y_col}' and '{zeta_col}'.")

    # Infer covariate columns as all numerics except y and zeta
    cov_cols = [c for c in df.columns if c not in {y_col, zeta_col}]
    X = df[cov_cols].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    zeta = df[zeta_col].to_numpy(dtype=int)

    return SurvivalDataset(X=X, y=y, zeta=zeta)


def simulate_cox_data(
    n: int = 400,
    d: int = 10,
    seed: int = 0,
    baseline_hazard: float = 0.01,
    censor_rate: float = 0.4,
) -> SurvivalDataset:
    """Simulate proportional-hazards survival data for testing/demo."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta_true = rng.normal(size=d)
    lin = X @ beta_true
    u = rng.uniform(size=n)
    T = -np.log(u) / (baseline_hazard * np.exp(lin))
    C = rng.exponential(scale=np.quantile(T, 1 - censor_rate), size=n)
    y = np.minimum(T, C)
    zeta = (T <= C).astype(int)
    return SurvivalDataset(X=X, y=y, zeta=zeta)
