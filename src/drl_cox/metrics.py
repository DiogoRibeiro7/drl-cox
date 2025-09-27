from __future__ import annotations
from typing import Literal, Dict, Any, List, Tuple
import numpy as np

__all__ = [
    "concordance_index",
    "time_dependent_auc_iAUC",
]


def _assert_ndarray(name: str, arr: np.ndarray, ndim: int | None = None) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")


def concordance_index(risk_scores: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> float:
    """Harrell's C-index for right-censored data.

    Higher `risk_scores` means higher risk.
    """
    _assert_ndarray("risk_scores", risk_scores, 1)
    _assert_ndarray("y", y, 1)
    _assert_ndarray("zeta", zeta, 1)
    n = y.shape[0]
    if risk_scores.shape[0] != n or zeta.shape[0] != n:
        raise ValueError("Shapes mismatch in concordance_index.")

    order = np.argsort(y)
    y_sorted = y[order]
    z_sorted = zeta[order]
    r_sorted = risk_scores[order]

    concordant = 0.0
    permissible = 0.0

    for i in range(n):
        if z_sorted[i] != 1:
            continue
        for j in range(i + 1, n):
            if y_sorted[j] <= y_sorted[i]:
                continue
            permissible += 1
            if r_sorted[i] > r_sorted[j]:
                concordant += 1
            elif r_sorted[i] == r_sorted[j]:
                concordant += 0.5
    if permissible == 0:
        return float("nan")
    return float(concordant / permissible)


# --- iAUC (IPCW) helpers ---

def _km_gbar(times: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier of censoring survival G(t-)."""
    _assert_ndarray("times", times, 1)
    _assert_ndarray("zeta", zeta, 1)
    n = times.shape[0]
    if zeta.shape[0] != n:
        raise ValueError("Shapes mismatch in _km_gbar.")

    delta_c = 1 - zeta
    uniq = np.unique(np.sort(times))
    G = np.ones_like(uniq, dtype=float)

    prod = 1.0
    for k, t in enumerate(uniq):
        d = np.sum((times == t) & (delta_c == 1))
        at_risk = np.sum(times >= t)
        if at_risk > 0:
            prod *= (1.0 - d / at_risk)
        G[k] = prod

    G_left = np.ones_like(G)
    G_left[0] = 1.0
    G_left[1:] = G[:-1]
    return uniq, np.clip(G_left, 1e-6, 1.0)


def time_dependent_auc_iAUC(
    risk_scores: np.ndarray,
    y: np.ndarray,
    zeta: np.ndarray,
    times: np.ndarray | None = None,
    average: Literal["uniform", "event"] = "event",
) -> float:
    """Time-dependent iAUC via IPCW (Heagerty & Zheng, 2005)."""
    _assert_ndarray("risk_scores", risk_scores, 1)
    _assert_ndarray("y", y, 1)
    _assert_ndarray("zeta", zeta, 1)

    n = y.shape[0]
    if risk_scores.shape[0] != n or zeta.shape[0] != n:
        raise ValueError("Shapes mismatch in iAUC.")

    if times is None:
        times = np.unique(y[zeta == 1])
    if times.size == 0:
        return float("nan")

    grid_G, G_left = _km_gbar(y, zeta)

    def G_of_left(t: float) -> float:
        idx = np.searchsorted(grid_G, t, side="left") - 1
        return 1.0 if idx < 0 else float(G_left[idx])

    aucs: list[float] = []
    weights: list[float] = []

    for t in times:
        cases = (zeta == 1) & (y <= t)
        ctrls = (y > t)
        if not np.any(cases) or not np.any(ctrls):
            continue

        w_cases = np.zeros(n)
        w_ctrls = np.zeros(n)

        for i in np.where(cases)[0]:
            w_cases[i] = 1.0 / G_of_left(float(y[i]))
        Gt = G_of_left(float(t))
        w_ctrls[ctrls] = 1.0 / Gt

        if w_cases.sum() > 0:
            w_cases /= w_cases.sum()
        if w_ctrls.sum() > 0:
            w_ctrls /= w_ctrls.sum()

        rs = risk_scores
        auc_t = 0.0
        wtot = 0.0
        for i in np.where(cases)[0]:
            for j in np.where(ctrls)[0]:
                w = w_cases[i] * w_ctrls[j]
                if rs[i] > rs[j]:
                    auc_t += w
                elif rs[i] == rs[j]:
                    auc_t += 0.5 * w
                wtot += w
        if wtot > 0:
            aucs.append(auc_t / wtot)
            weights.append(float(np.sum(cases)) if average == "event" else 1.0)

    if len(aucs) == 0:
        return float("nan")
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    return float(np.sum(weights_arr * np.asarray(aucs)))
