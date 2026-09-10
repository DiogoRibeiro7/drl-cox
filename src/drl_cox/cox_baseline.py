"""Classical Cox proportional-hazards baselines.

Three lightweight estimators built on the Breslow partial likelihood:

* :class:`CoxPartialLikelihood` -- unpenalised Newton-Raphson fit.
* :class:`CoxRidge` -- L2-penalised Newton-Raphson fit.
* :class:`CoxLasso` -- L1-penalised proximal-gradient (ISTA) fit.

The penalised objectives are normalised by the number of samples, so ``alpha`` is
comparable across datasets of different sizes (as in ``glmnet``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CoxPartialLikelihood", "CoxRidge", "CoxLasso"]

_EXP_CLIP = 50.0


def _assert_ndarray(name: str, arr: np.ndarray, ndim: int | None = None) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")


def _check_inputs(X: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> None:
    _assert_ndarray("X", X, 2)
    _assert_ndarray("y", y, 1)
    _assert_ndarray("zeta", zeta, 1)
    if y.shape[0] != X.shape[0] or zeta.shape[0] != X.shape[0]:
        raise ValueError("X, y and zeta must have the same number of samples.")


def _riskset_prefix_sums(y: np.ndarray, Xbeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by descending ``y`` so that risk sets are prefixes.

    Returns the sort order and the cumulative sums of ``exp(Xbeta)`` in that order.
    """
    order = np.argsort(-y)
    exp_Xb = np.exp(np.clip(Xbeta[order], -_EXP_CLIP, _EXP_CLIP))
    prefix = np.cumsum(exp_Xb)
    return order, prefix


def _partial_loglik_derivatives(
    X: np.ndarray, y: np.ndarray, zeta: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of the Cox log partial likelihood (Breslow ties)."""
    d = X.shape[1]
    Xb = X @ beta
    w = np.exp(np.clip(Xb, -_EXP_CLIP, _EXP_CLIP))
    grad = np.zeros(d)
    hess = np.zeros((d, d))
    for i in np.flatnonzero(zeta == 1):
        w_i = np.where(y >= y[i], w, 0.0)
        w_sum = w_i.sum()
        if w_sum <= 0:
            continue
        Ex = (X * w_i[:, None]).sum(axis=0) / w_sum
        Exx = (X.T * w_i) @ X / w_sum
        grad += X[i] - Ex
        hess -= Exx - np.outer(Ex, Ex)
    return grad, hess


def _partial_grad(X: np.ndarray, y: np.ndarray, zeta: np.ndarray, beta: np.ndarray) -> float:
    """Sum over coordinates of the partial log-likelihood gradient.

    Kept for backwards compatibility; prefer :func:`_partial_loglik_derivatives`.
    """
    grad, _ = _partial_loglik_derivatives(X, y, zeta, beta)
    return float(np.sum(grad))


def _newton_cox(
    X: np.ndarray,
    y: np.ndarray,
    zeta: np.ndarray,
    *,
    l2: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Maximise ``loglik(beta) / n - l2 * ||beta||^2 / 2`` with Newton-Raphson steps.

    Singular Hessians (no events, constant features, more features than samples,
    perfect separation) fall back to a least-squares Newton step so that the
    iteration always returns a finite coefficient vector.
    """
    n, d = X.shape
    beta = np.zeros(d)
    if not np.any(zeta == 1):
        return beta  # no events: the partial likelihood is flat, the penalised maximiser is 0
    eye = np.eye(d)
    for _ in range(max_iter):
        grad, hess = _partial_loglik_derivatives(X, y, zeta, beta)
        grad = grad / n - l2 * beta
        hess = hess / n - l2 * eye
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta_new = beta - step
        if not np.all(np.isfinite(beta_new)):
            break
        converged = float(np.linalg.norm(beta_new - beta)) < tol
        beta = beta_new
        if converged:
            break
    return beta


@dataclass
class CoxPartialLikelihood:
    """Unpenalised Cox model fitted by Newton-Raphson on the partial likelihood."""

    max_iter: int = 50
    tol: float = 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """Return the coefficient vector maximising the partial likelihood."""
        _check_inputs(X, y, zeta)
        return _newton_cox(X, y, zeta, l2=0.0, max_iter=self.max_iter, tol=self.tol)


@dataclass
class CoxRidge:
    """L2-penalised Cox model maximising ``loglik / n - alpha * ||beta||^2 / 2``."""

    alpha: float = 1.0
    max_iter: int = 50
    tol: float = 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """Return the ridge-penalised coefficient vector."""
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        _check_inputs(X, y, zeta)
        return _newton_cox(X, y, zeta, l2=self.alpha, max_iter=self.max_iter, tol=self.tol)


@dataclass
class CoxLasso:
    """L1-penalised Cox model minimising ``-loglik / n + alpha * ||beta||_1``.

    Fitted with proximal gradient descent (ISTA) using a fixed step size derived
    from a Lipschitz bound on the gradient.
    """

    alpha: float = 0.01
    max_iter: int = 100
    tol: float = 1e-6

    def fit(self, X: np.ndarray, y: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """Return the lasso-penalised coefficient vector."""
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        _check_inputs(X, y, zeta)
        n, d = X.shape
        beta = np.zeros(d)
        n_events = int(np.sum(zeta == 1))
        if n_events == 0:
            return beta

        # Each event contributes a weighted covariance of the rows of X to the Hessian
        # of -loglik / n, which is bounded by the largest squared row norm.
        lipschitz = (n_events / n) * float(np.max(np.sum(X**2, axis=1))) + 1e-12
        step = 1.0 / lipschitz
        threshold = step * self.alpha

        for _ in range(self.max_iter):
            grad, _ = _partial_loglik_derivatives(X, y, zeta, beta)
            z = beta + step * grad / n
            beta_new = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
            converged = float(np.linalg.norm(beta_new - beta)) < self.tol
            beta = beta_new
            if converged:
                break
        return beta
