"""Discretizations of the Caputo derivative and of the Riemann-Liouville integral.

All operators act on functions sampled on a uniform grid ``t = 0, h, ..., N*h``.
Two discretizations of the candidate Caputo derivative are provided:

``l1_operator``
    The uniform-grid L1 formula. At ``order = 1`` its coefficients reduce to the
    backward difference, so the same code path applies at the integer order.

``gauss_jacobi_derivative``
    Gauss-Jacobi quadrature applied to the derivative of a fitted smoother, which
    matches the weakly singular weight of the Caputo integral exactly.
"""

from __future__ import annotations

from math import gamma

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.special import roots_jacobi

__all__ = [
    "l1_operator",
    "apply_operator",
    "gauss_jacobi_derivative",
    "fractional_integral_operator",
    "smooth_on_grid",
    "fit_smoother",
]


def l1_operator(order: float, n_steps: int, step: float) -> np.ndarray:
    """Matrix of the uniform-grid L1 Caputo derivative of the given order.

    Returns ``W`` such that the derivative at grid index ``n`` is
    ``W[n] @ diff`` where ``diff[k] = x[k] - x[k-1]``.
    """
    scale = step ** (-order) / gamma(2.0 - order)
    k = np.arange(n_steps + 1)
    weights = (k + 1.0) ** (1.0 - order) - k ** (1.0 - order)
    W = np.zeros((n_steps + 1, n_steps + 1))
    for n in range(1, n_steps + 1):
        idx = np.arange(1, n + 1)
        W[n, 1 : n + 1] = scale * weights[n - idx]
    return W


def apply_operator(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply an L1 operator matrix to a sampled trajectory ``x`` of shape (N+1, d)."""
    diff = np.zeros_like(x)
    diff[1:] = x[1:] - x[:-1]
    return W @ diff


def fit_smoother(t: np.ndarray, y: np.ndarray, noise_scale: float,
                 strength: float = 1.0) -> list[UnivariateSpline]:
    """Fit one cubic smoothing spline per state coordinate.

    ``noise_scale`` is the standard deviation of the observation noise; a value of
    zero produces an interpolating spline. ``strength`` rescales the smoothing
    penalty and is used to test sensitivity to that choice.
    """
    penalty = 0.0 if noise_scale <= 0 else strength * len(t) * noise_scale ** 2
    return [UnivariateSpline(t, y[:, c], k=3, s=penalty) for c in range(y.shape[1])]


def smooth_on_grid(t_obs: np.ndarray, y_obs: np.ndarray, t_grid: np.ndarray,
                   noise_scale: float, strength: float = 1.0) -> np.ndarray:
    """Smooth possibly irregular observations and resample on a uniform grid."""
    splines = fit_smoother(t_obs, y_obs, noise_scale, strength)
    return np.column_stack([sp(t_grid) for sp in splines])


def gauss_jacobi_derivative(t_grid: np.ndarray, splines, order: float,
                            n_nodes: int = 24) -> np.ndarray:
    """Candidate Caputo derivative by Gauss-Jacobi quadrature on a fitted smoother.

    Evaluates ``t^{1-b} / Gamma(1-b) * int_0^1 (1-r)^{-b} x'(t r) dr`` with the
    weight absorbed into the quadrature rule.
    """
    nodes, weights = roots_jacobi(n_nodes, -order, 0.0)
    r = (nodes + 1.0) / 2.0
    w = weights * (2.0 ** (order - 1.0))
    t = np.asarray(t_grid, dtype=float)
    points = np.outer(t, r)
    out = np.zeros((len(t), len(splines)))
    for c, sp in enumerate(splines):
        out[:, c] = sp(points.ravel(), 1).reshape(points.shape) @ w
    prefactor = np.where(t > 0, t, 0.0) ** (1.0 - order) / gamma(1.0 - order)
    out *= prefactor[:, None]
    out[t <= 0] = 0.0
    return out


def fractional_integral_operator(order: float, n_steps: int, step: float) -> np.ndarray:
    """Matrix of the Riemann-Liouville integral of the given order.

    Uses the corrector weights of the predictor-corrector scheme, so that
    ``x(t_n) = x_0 + (I^order f)(t_n)`` reproduces the Volterra form of the model.
    """
    a = step ** order / (order * (order + 1.0))
    W = np.zeros((n_steps + 1, n_steps + 1))
    for m in range(1, n_steps + 1):
        n = m - 1
        W[m, 0] = a * (n ** (order + 1) - (n - order) * (n + 1) ** order)
        if n >= 1:
            j = np.arange(1, n + 1)
            W[m, 1 : n + 1] = a * (
                (n - j + 2) ** (order + 1)
                + (n - j) ** (order + 1)
                - 2.0 * (n - j + 1) ** (order + 1)
            )
        W[m, m] = a
    return W / gamma(order)
