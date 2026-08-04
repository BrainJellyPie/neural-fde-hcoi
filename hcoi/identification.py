"""Pair selection, the history-compatibility profile, and the order estimators.

The compatibility profile measures, for each candidate order, how far the
candidate labels are from being explainable by a single autonomous vector field
of bounded Lipschitz constant. Pairs of observations that are close in state but
distant in history carry that information, so the pair set determines what the
profile can detect.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .caputo import apply_operator, l1_operator, smooth_on_grid

__all__ = [
    "nearest_neighbor_pairs",
    "closest_cross_trajectory_pairs",
    "violation_profile",
    "rescale",
    "identification_set",
    "candidate_labels",
    "estimate_order",
    "early_time_slope",
    "affine_profile_order",
    "joint_field_fit",
]


# --------------------------------------------------------------------------- #
# Pair selection
# --------------------------------------------------------------------------- #
def nearest_neighbor_pairs(states: np.ndarray, trajectory_id: np.ndarray,
                           times: np.ndarray, n_neighbors: int = 8,
                           history_separation: float = 0.15) -> np.ndarray:
    """State-space nearest neighbors, excluding adjacent same-trajectory samples.

    Distant same-trajectory pairs are retained, since they carry self-intersection
    information.
    """
    tree = cKDTree(states)
    _, idx = tree.query(states, k=n_neighbors + 1)
    pairs = set()
    for i in range(len(states)):
        for j in idx[i, 1:]:
            different_history = trajectory_id[j] != trajectory_id[i]
            separated = abs(times[j] - times[i]) >= history_separation
            if different_history or separated:
                pairs.add((min(i, j), max(i, j)))
    return np.array(sorted(pairs)) if pairs else np.empty((0, 2), dtype=int)


def closest_cross_trajectory_pairs(states: np.ndarray, trajectory_id: np.ndarray,
                                   n_pairs: int = 30,
                                   n_neighbors: int = 8) -> np.ndarray:
    """The ``n_pairs`` closest pairs of states lying on different trajectories.

    These approximate the exact collisions of the theory without being engineered.
    """
    tree = cKDTree(states)
    dist, idx = tree.query(states, k=n_neighbors)
    candidates: dict[tuple[int, int], float] = {}
    for i in range(len(states)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if trajectory_id[j] != trajectory_id[i]:
                key = (min(i, j), max(i, j))
                candidates[key] = min(candidates.get(key, np.inf), d)
    ordered = sorted(candidates.items(), key=lambda kv: kv[1])[:n_pairs]
    return np.array([k for k, _ in ordered]) if ordered else np.empty((0, 2), dtype=int)


# --------------------------------------------------------------------------- #
# Compatibility profile
# --------------------------------------------------------------------------- #
def violation_profile(orders, labels: dict, states: np.ndarray, pairs: np.ndarray,
                      budget: float, statistic: str = "max",
                      quantile: float = 0.9):
    """Profile of the pairwise violation of the Lipschitz budget.

    For each candidate order the violation of a pair is
    ``max(|v_i - v_j| - budget * |z_i - z_j|, 0)``. ``statistic`` selects the hard
    maximum or an upper quantile, the latter being less sensitive to a single
    mis-selected pair.
    """
    if len(pairs) == 0:
        return None
    gap = np.linalg.norm(states[pairs[:, 0]] - states[pairs[:, 1]], axis=1)
    profile = np.empty(len(orders))
    for k, b in enumerate(orders):
        v = labels[b]
        jump = np.linalg.norm(v[pairs[:, 0]] - v[pairs[:, 1]], axis=1)
        violation = np.clip(jump - budget * gap, 0.0, None)
        profile[k] = violation.max() if statistic == "max" else np.quantile(
            violation, quantile
        )
    return profile


def rescale(profile: np.ndarray) -> np.ndarray:
    """Map a profile to the unit range so that profiles can be averaged."""
    span = profile.max() - profile.min()
    return (profile - profile.min()) / (span + 1e-300)


def identification_set(orders: np.ndarray, profile: np.ndarray,
                       tolerance: float = 0.15):
    """Orders at which the rescaled profile stays below ``tolerance``."""
    selected = orders[profile <= tolerance]
    if len(selected) == 0:
        selected = orders[profile <= profile.min() + 1e-12]
    return float(selected.min()), float(selected.max()), float(
        selected.max() - selected.min()
    )


def candidate_labels(trajectories, operators: dict, orders, start: int, stop: int):
    """Stack candidate Caputo labels and states over the analysis window."""
    labels = {
        b: np.concatenate([apply_operator(operators[b], x)[start:stop]
                           for x in trajectories])
        for b in orders
    }
    states = np.concatenate([x[start:stop] for x in trajectories])
    trajectory_id = np.concatenate(
        [np.full(stop - start, m) for m in range(len(trajectories))]
    )
    return labels, states, trajectory_id


# --------------------------------------------------------------------------- #
# Order estimators
# --------------------------------------------------------------------------- #
def estimate_order(observations: np.ndarray, t: np.ndarray, orders: np.ndarray,
                   operators: dict, budget: float, noise_scale: float,
                   start: int, stop: int,
                   smoothing_strengths=(0.7, 1.0, 1.4),
                   n_closest_pairs: int = 30,
                   quantile: float = 0.9,
                   set_tolerance: float = 0.15,
                   abstain_width: float = 0.30,
                   variant: str = "consensus"):
    """History-compatible order identification.

    ``variant="neighbors"`` uses one smoother and the nearest-neighbor pair set
    scored by the hard maximum. ``variant="consensus"`` evaluates the profile at
    several smoothing strengths and on two pair sets, rescales each profile to a
    common range and averages them, so that an order is favored only when the
    smoothing choices and the pair sets agree.

    Returns a dictionary with the estimate, the identification set, whether the
    procedure abstains, and the intermediate quantities at unit smoothing.
    """
    n_traj = observations.shape[1]
    strengths = smoothing_strengths if variant == "consensus" else (1.0,)
    profiles = []
    reference = None
    for strength in strengths:
        smoothed = [
            smooth_on_grid(t, observations[:, m, :], t, noise_scale, strength)
            for m in range(n_traj)
        ]
        labels, states, traj_id = candidate_labels(smoothed, operators, orders,
                                                   start, stop)
        times = np.concatenate([t[start:stop]] * n_traj)
        if strength == 1.0:
            reference = dict(labels=labels, states=states, trajectory_id=traj_id,
                             times=times)
        neighbor_pairs = nearest_neighbor_pairs(states, traj_id, times)
        p = violation_profile(orders, labels, states, neighbor_pairs, budget, "max")
        if p is not None:
            profiles.append(rescale(p))
        if variant == "consensus":
            close_pairs = closest_cross_trajectory_pairs(states, traj_id,
                                                         n_closest_pairs)
            q = violation_profile(orders, labels, states, close_pairs, budget,
                                  "quantile", quantile)
            if q is not None:
                profiles.append(rescale(q))

    if not profiles:
        return dict(order=None, profile=None, width=None, abstained=True,
                    reference=reference)
    combined = np.mean(profiles, axis=0)
    lower, upper, width = identification_set(orders, combined, set_tolerance)
    abstained = width > abstain_width
    estimate = None if abstained else float(orders[int(np.argmin(combined))])
    return dict(order=estimate,
                argmin=float(orders[int(np.argmin(combined))]),
                profile=combined, lower=lower, upper=upper, width=width,
                abstained=abstained, reference=reference)


def early_time_slope(t: np.ndarray, trajectory: np.ndarray, x0: np.ndarray,
                     lower: float = 0.012, upper: float = 0.10) -> float:
    """Order estimate from the slope of the initial displacement on log axes."""
    window = (t >= lower) & (t <= upper)
    radius = np.linalg.norm(trajectory[window] - x0, axis=1)
    positive = radius > 0
    slope, _ = np.polyfit(np.log(t[window][positive]), np.log(radius[positive]), 1)
    return float(slope)


def affine_profile_order(orders, labels: dict, states: np.ndarray) -> float:
    """Order minimizing the residual of a single affine autonomous field.

    Exact on a linear system and unable to represent a nonlinear field, so this
    serves as a structural diagnostic and not as a general-purpose comparator.
    """
    design = np.column_stack([states, np.ones(len(states))])
    residuals = []
    for b in orders:
        coef, *_ = np.linalg.lstsq(design, labels[b], rcond=None)
        residuals.append(np.sqrt(np.mean((labels[b] - design @ coef) ** 2)))
    return float(orders[int(np.argmin(residuals))])


def joint_field_fit(orders, states: np.ndarray, targets: np.ndarray,
                    integral_operators: dict, n_features: int = 64,
                    ridge: float = 1e-8, seed: int = 0):
    """Fit the order and a nonlinear field together by trajectory reconstruction.

    The field is represented by a random feature expansion with a linear output
    layer, so the reconstruction error is quadratic in the output weights and the
    optimal field at each candidate order is obtained in closed form. The reported
    error is therefore the global minimum over fields at that order.

    Parameters
    ----------
    states : list of per-trajectory state arrays on the uniform grid
    targets : list of per-trajectory displacements ``x(t) - x(0)``
    integral_operators : Riemann-Liouville integral matrices keyed by order

    Returns
    -------
    orders, relative reconstruction error per order, fitted field per order
    """
    rng = np.random.default_rng(seed)
    stacked = np.concatenate(states)
    center, spread = stacked.mean(0), stacked.std(0) + 1e-12
    W = rng.normal(0, 1.5, (stacked.shape[1], n_features))
    b = rng.normal(0, 0.5, n_features)
    features = [np.tanh(((x - center) / spread) @ W + b) for x in states]
    all_features = np.tanh(((stacked - center) / spread) @ W + b)
    scale = float(np.sqrt(np.mean(np.concatenate(targets) ** 2)))

    errors, fields = [], []
    for order in orders:
        design = np.concatenate([integral_operators[order] @ f for f in features])
        target = np.concatenate(targets)
        gram = design.T @ design + ridge * np.eye(n_features)
        coef = np.linalg.solve(gram, design.T @ target)
        residual = target - design @ coef
        errors.append(float(np.sqrt(np.mean(residual ** 2)) / scale))
        fields.append(all_features @ coef)
    return np.asarray(errors), fields, stacked
