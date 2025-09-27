from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

__all__ = ["CoxPartialLikelihood", "CoxRidge", "CoxLasso"]


def _assert_ndarray(name: str, arr: np.ndarray, ndim: int | None = None) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")


def _riskset_prefix_sums(y: np.ndarray, Xbeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by descending y so risk sets are prefixes; return sorted indices and exp(Xβ) prefix sums."""
    order = np.argsort(-y)
    y_sorted = y[order]
    Xb_sorted = Xbeta[order]
    exp_Xb = np.exp(np.clip(Xb_sorted, -50, 50))
    prefix = np.cumsum(exp_Xb)
    return order, prefix


@dataclass
class CoxPartialLikelihood:
    max_iter: int = 50
    tol: float = 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """Newton–Raphson on partial log-likelihood (no penalty)."""
        _assert_ndarray("X", X, 2)
        _assert_ndarray("y", y, 1)
        _assert_ndarray("zeta", zeta, 1)
        N, d = X.shape
        beta = np.zeros(d)

        for _ in range(self.max_iter):
            Xb = X @ beta
            order, prefix = _riskset_prefix_sums(y, Xb)
            # Efficient contributions using prefix sums
            grad = np.zeros(d)
            H = np.zeros((d, d))

            # Map inverse permutation for quick rank lookup
            inv = np.empty(N, dtype=int)
            inv[order] = np.arange(N)

            exp_Xb = np.exp(np.clip(Xb, -50, 50))

            for i in range(N):
                if zeta[i] != 1:
                    continue
                rk = inv[i]
                denom = prefix[rk]
                # E[X | riskset] under weights exp(Xβ)
                w = exp_Xb.copy()
                # zero those not in risk set: require y_j >= y_i
                mask = (y >= y[i])
                w[~mask] = 0.0
                w_sum = w.sum()
                if w_sum <= 0:
                    continue
                Ex = (X * w[:, None]).sum(axis=0) / w_sum
                grad += (X[i] - Ex)

                # Hessian contribution: Var[X | riskset]
                Exx = (X[:, :, None] * X[:, None, :] * w[:, None, None]).sum(axis=0) / w_sum
                H -= (Exx - np.outer(Ex, Ex))

            step = np.linalg.solve(H, grad)
            beta_new = beta - step
            if np.linalg.norm(beta_new - beta) < self.tol:
                beta = beta_new
                break
            beta = beta_new
        return beta


@dataclass
class CoxRidge:
    alpha: float = 1.0  # L2 penalty strength
    max_iter: int = 50
    tol: float = 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        base = CoxPartialLikelihood(max_iter=self.max_iter, tol=self.tol)
        beta = base.fit(X, y, zeta)
        # Single proximal step toward ridge (small adjustment)
        return beta / (1.0 + self.alpha)


@dataclass
class CoxLasso:
    alpha: float = 0.01  # L1 penalty strength
    max_iter: int = 100
    tol: float = 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """Coordinate descent on partial likelihood + L1.
        Simple implementation; for serious use prefer glmnet-like libraries.
        """
        _assert_ndarray("X", X, 2)
        _assert_ndarray("y", y, 1)
        _assert_ndarray("zeta", zeta, 1)
        N, d = X.shape
        beta = np.zeros(d)

        for _ in range(self.max_iter):
            beta_old = beta.copy()
            # Update each coordinate with a soft-thresholding step on gradient approx
            for j in range(d):
                # Numerical gradient (cheap approx):
                eps = 1e-5
                e_j = np.zeros(d); e_j[j] = 1.0
                g_plus = _partial_grad(X, y, zeta, beta + eps * e_j)
                g_minus = _partial_grad(X, y, zeta, beta - eps * e_j)
                grad_j = (g_plus - g_minus) / (2 * eps)
                # Soft threshold
                bj = beta[j] - 0.01 * grad_j  # small step
                beta[j] = np.sign(bj) * max(0.0, abs(bj) - self.alpha * 0.01)
            if np.linalg.norm(beta - beta_old) < self.tol:
                break
        return beta


def _partial_grad(X: np.ndarray, y: np.ndarray, zeta: np.ndarray, beta: np.ndarray) -> float:
    """Return sum of gradients along all dimensions (proxy for per-coordinate update)."""
    Xb = X @ beta
    order = np.argsort(-y)
    exp_Xb = np.exp(np.clip(Xb, -50, 50))
    grad = np.zeros_like(beta)
    for i in range(X.shape[0]):
        if zeta[i] != 1:
            continue
        mask = (y >= y[i])
        w = exp_Xb * mask
        w_sum = w.sum()
        if w_sum <= 0:
            continue
        Ex = (X * w[:, None]).sum(axis=0) / w_sum
        grad += (X[i] - Ex)
    return float(np.sum(grad))
