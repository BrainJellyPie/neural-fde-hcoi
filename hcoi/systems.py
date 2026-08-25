"""Vector fields and the fractional predictor-corrector solver.

The systems studied in the paper are

``NeuralVectorField``
    A fixed-weight multilayer perceptron with tanh activations, used as a
    nonlinear right-hand side whose trajectories carry forward-solver error.

``DissipativeVectorField``
    The same network with an added linear damping term. Anisotropic damping makes
    the trajectories concentrate on a slow manifold, which separates the state
    dimension from the dimension of the region the trajectories actually visit.

``linear_system_matrix`` / ``ordinary_flow``
    The two-dimensional linear system used for the closed-form studies and, at
    integer order, for the boundary test.
"""

from __future__ import annotations

from math import gamma

import numpy as np
from scipy.linalg import expm

__all__ = [
    "NeuralVectorField",
    "DissipativeVectorField",
    "linear_system_matrix",
    "ordinary_flow",
    "collision_initial_states",
    "solve_caputo",
    "empirical_lipschitz",
]


class NeuralVectorField:
    """Fixed-weight tanh network used as the nonlinear vector field."""

    def __init__(self, dim: int = 2, width: int = 16, scale: float = 0.6,
                 seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.9, (dim, width))
        self.b1 = rng.normal(0, 0.3, width)
        self.W2 = rng.normal(0, 0.9, (width, width))
        self.b2 = rng.normal(0, 0.3, width)
        self.W3 = rng.normal(0, 0.9, (width, dim))
        self.b3 = rng.normal(0, 0.2, dim)
        self.scale = scale
        self.dim = dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return self.scale * (h @ self.W3 + self.b3)


class DissipativeVectorField:
    """Neural vector field with an added linear damping term ``-damping * x``."""

    def __init__(self, dim: int, damping, seed: int = 1, scale: float = 0.6) -> None:
        self.network = NeuralVectorField(dim=dim, seed=seed, scale=scale)
        self.damping = np.asarray(damping, dtype=float)
        self.dim = dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return -x * self.damping + self.network(x)


def linear_system_matrix() -> np.ndarray:
    """Coefficient matrix of the two-dimensional linear system used in the paper."""
    return np.array([[-0.9, -1.5], [1.1, -0.4]])


def ordinary_flow(A: np.ndarray, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Solution of ``x' = A x`` in closed form, shape (len(t), n_states, dim)."""
    return np.stack(
        [np.stack([expm(A * s) @ x0[m] for s in t]) for m in range(len(x0))], axis=1
    )


def collision_initial_states(A: np.ndarray, common_state: np.ndarray,
                             visit_times: np.ndarray) -> np.ndarray:
    """Initial states whose ordinary trajectories share a state at distinct times."""
    return np.stack([expm(-A * s) @ common_state for s in visit_times])


def solve_caputo(field, x0: np.ndarray, order: float, step: float,
                 n_steps: int) -> np.ndarray:
    """Predictor-corrector solution of the commensurate Caputo system.

    Parameters
    ----------
    field : callable mapping an array of states to an array of derivatives
    x0 : array of shape (n_trajectories, dim)
    order : fractional order in (0, 1]
    step, n_steps : uniform time step and number of steps

    Returns
    -------
    array of shape (n_steps + 1, n_trajectories, dim)
    """
    n_traj, dim = x0.shape
    x = np.zeros((n_steps + 1, n_traj, dim))
    f = np.zeros((n_steps + 1, n_traj, dim))
    x[0] = x0
    f[0] = field(x0)
    inv_gamma = 1.0 / gamma(order)
    corrector = step ** order / (order * (order + 1.0))
    predictor = step ** order / order
    for k in range(n_steps):
        j = np.arange(k + 1)
        pred_w = predictor * ((k + 1 - j) ** order - (k - j) ** order)
        prediction = x0 + inv_gamma * np.einsum("j,jbd->bd", pred_w, f[: k + 1])
        f_pred = field(prediction)
        corr_w = np.empty(k + 1)
        corr_w[0] = corrector * (
            k ** (order + 1) - (k - order) * (k + 1) ** order
        )
        if k >= 1:
            jj = np.arange(1, k + 1)
            corr_w[1:] = corrector * (
                (k - jj + 2) ** (order + 1)
                + (k - jj) ** (order + 1)
                - 2.0 * (k - jj + 1) ** (order + 1)
            )
        x[k + 1] = x0 + inv_gamma * (
            np.einsum("j,jbd->bd", corr_w, f[: k + 1]) + corrector * f_pred
        )
        f[k + 1] = field(x[k + 1])
    return x


def empirical_lipschitz(states: np.ndarray, values: np.ndarray,
                        n_probes: int = 400, seed: int = 0,
                        min_distance: float = 1e-3) -> float:
    """Largest difference quotient of ``values`` over sampled pairs of ``states``."""
    rng = np.random.default_rng(seed)
    probes = rng.choice(len(states), size=min(n_probes, len(states)), replace=False)
    best = 0.0
    for i in probes:
        d = np.linalg.norm(states - states[i], axis=1)
        far = d > min_distance
        if far.any():
            ratio = np.linalg.norm(values[far] - values[i], axis=1) / d[far]
            best = max(best, float(ratio.max()))
    return best


# --------------------------------------------------------------------------- #
# Generators outside the single-order autonomous model class
# --------------------------------------------------------------------------- #
class ForcedVectorField:
    """Neural field with an explicit time-dependent forcing term.

    Violates the autonomy assumption (A2): the right-hand side depends on time
    as well as on the state, so no single autonomous field explains the labels.
    """

    def __init__(self, dim: int = 2, amplitude: float = 0.25,
                 frequency: float = 2.0, seed: int = 1) -> None:
        self.network = NeuralVectorField(dim=dim, seed=seed)
        self.amplitude = amplitude
        self.frequency = frequency
        self.dim = dim
        self.time = 0.0

    def set_time(self, t: float) -> None:
        self.time = t

    def __call__(self, x: np.ndarray) -> np.ndarray:
        drive = self.amplitude * np.sin(self.frequency * np.pi * self.time)
        return self.network(x) + drive


def solve_caputo_variable_order(field, x0: np.ndarray, order_of_time,
                                step: float, n_steps: int) -> np.ndarray:
    """Predictor-corrector solution when the order varies slowly with time.

    At each step the memory kernel uses the order in force at that time, which
    is the piecewise-frozen scheme commonly used for variable-order models. The
    resulting trajectories have no single commensurate order.
    """
    n_traj, dim = x0.shape
    x = np.zeros((n_steps + 1, n_traj, dim))
    f = np.zeros((n_steps + 1, n_traj, dim))
    x[0] = x0
    f[0] = field(x0)
    for k in range(n_steps):
        order = float(order_of_time((k + 1) * step))
        inv_gamma = 1.0 / gamma(order)
        corrector = step ** order / (order * (order + 1.0))
        predictor = step ** order / order
        j = np.arange(k + 1)
        pred_w = predictor * ((k + 1 - j) ** order - (k - j) ** order)
        prediction = x0 + inv_gamma * np.einsum("j,jbd->bd", pred_w, f[: k + 1])
        f_pred = field(prediction)
        corr_w = np.empty(k + 1)
        corr_w[0] = corrector * (k ** (order + 1) - (k - order) * (k + 1) ** order)
        if k >= 1:
            jj = np.arange(1, k + 1)
            corr_w[1:] = corrector * (
                (k - jj + 2) ** (order + 1)
                + (k - jj) ** (order + 1)
                - 2.0 * (k - jj + 1) ** (order + 1)
            )
        x[k + 1] = x0 + inv_gamma * (
            np.einsum("j,jbd->bd", corr_w, f[: k + 1]) + corrector * f_pred
        )
        f[k + 1] = field(x[k + 1])
    return x


def solve_caputo_forced(field: "ForcedVectorField", x0: np.ndarray, order: float,
                        step: float, n_steps: int) -> np.ndarray:
    """Predictor-corrector solution for a right-hand side that depends on time."""
    n_traj, dim = x0.shape
    x = np.zeros((n_steps + 1, n_traj, dim))
    f = np.zeros((n_steps + 1, n_traj, dim))
    x[0] = x0
    field.set_time(0.0)
    f[0] = field(x0)
    inv_gamma = 1.0 / gamma(order)
    corrector = step ** order / (order * (order + 1.0))
    predictor = step ** order / order
    for k in range(n_steps):
        j = np.arange(k + 1)
        pred_w = predictor * ((k + 1 - j) ** order - (k - j) ** order)
        prediction = x0 + inv_gamma * np.einsum("j,jbd->bd", pred_w, f[: k + 1])
        field.set_time((k + 1) * step)
        f_pred = field(prediction)
        corr_w = np.empty(k + 1)
        corr_w[0] = corrector * (k ** (order + 1) - (k - order) * (k + 1) ** order)
        if k >= 1:
            jj = np.arange(1, k + 1)
            corr_w[1:] = corrector * (
                (k - jj + 2) ** (order + 1)
                + (k - jj) ** (order + 1)
                - 2.0 * (k - jj + 1) ** (order + 1)
            )
        x[k + 1] = x0 + inv_gamma * (
            np.einsum("j,jbd->bd", corr_w, f[: k + 1]) + corrector * f_pred
        )
        field.set_time((k + 1) * step)
        f[k + 1] = field(x[k + 1])
    return x


class LinearVectorField:
    """Linear autonomous field ``x -> A x``, used for the linear Caputo system."""

    def __init__(self, A: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)
        self.dim = self.A.shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.A.T


def mittag_leffler_matrix(A: np.ndarray, s: float, order: float,
                          n_terms: int = 120) -> np.ndarray:
    """Matrix Mittag-Leffler function ``E_order(A s^order)`` by its series."""
    term = np.eye(A.shape[0])
    total = term / gamma(1.0)
    scaled = A * (s ** order)
    for k in range(1, n_terms):
        term = term @ scaled
        total = total + term / gamma(order * k + 1.0)
    return total


def mittag_leffler_flow(A: np.ndarray, x0: np.ndarray, t: np.ndarray,
                        order: float) -> np.ndarray:
    """Solution of the linear Caputo system in closed form.

    Returns an array of shape (len(t), n_states, dim). Because the solution is
    analytic, forward-solver error does not enter, which isolates the
    identifiability effects from the numerical scheme.
    """
    flows = [mittag_leffler_matrix(A, s, order) for s in t]
    return np.stack([np.stack([E @ x0[m] for m in range(len(x0))]) for E in flows])


def fractional_collision_states(A: np.ndarray, common_state: np.ndarray,
                                visit_times: np.ndarray,
                                order: float) -> np.ndarray:
    """Initial states whose fractional trajectories share a state at distinct times."""
    return np.stack([
        np.linalg.solve(mittag_leffler_matrix(A, s, order), common_state)
        for s in visit_times])


def shoot_collision_states(field, common_state: np.ndarray, visit_times,
                           order: float, step: float, n_steps: int,
                           n_iterations: int = 25, tolerance: float = 1e-9):
    """Initial states whose nonlinear trajectories pass through a common state.

    For a nonlinear field the initial state cannot be obtained in closed form, so
    it is found by Newton iteration on the shooting residual, with the Jacobian
    of the time-``t`` map taken by finite differences.
    """
    common_state = np.asarray(common_state, dtype=float)
    visit_times = np.asarray(visit_times, dtype=float)
    dim = len(common_state)
    x0 = np.tile(common_state, (len(visit_times), 1))
    index = np.clip(np.round(visit_times / step).astype(int), 1, n_steps)

    def reached(states):
        trajectory = solve_caputo(field, states, order, step, n_steps)
        return np.stack([trajectory[index[m], m] for m in range(len(states))])

    delta = 1e-6
    for _ in range(n_iterations):
        base = reached(x0)
        residual = base - common_state[None, :]
        if np.max(np.linalg.norm(residual, axis=1)) < tolerance:
            break
        jacobian = np.zeros((len(x0), dim, dim))
        for c in range(dim):
            bumped = x0.copy()
            bumped[:, c] += delta
            jacobian[:, :, c] = (reached(bumped) - base) / delta
        for m in range(len(x0)):
            try:
                x0[m] -= np.linalg.solve(jacobian[m], residual[m])
            except np.linalg.LinAlgError:
                x0[m] -= residual[m]
    return x0
