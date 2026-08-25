"""Diagnostics computed from observed states alone.

These quantities indicate, before any order estimate is formed, whether a dataset
carries the structure that history-compatible identification requires.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "transversality_ratio",
    "overlap_availability",
    "relative_cross_trajectory_distance",
    "effective_dimension",
]


def transversality_ratio(orders: np.ndarray, labels: dict, states: np.ndarray,
                         pairs: np.ndarray, reference_order: float, budget: float,
                         state_error: float, label_error_fraction: float = 0.02):
    """Rate at which a pair separates candidate orders, relative to its error level.

    A pair with a ratio above one separates candidate orders more strongly than its
    own error level and therefore carries usable order information. The count of
    such pairs, and not their proportion, is what tracks recovery accuracy.
    """
    if len(pairs) == 0:
        return np.empty(0)
    k = int(np.argmin(np.abs(orders - reference_order)))
    lo, hi = orders[max(k - 1, 0)], orders[min(k + 1, len(orders) - 1)]
    slope = (labels[hi] - labels[lo]) / (hi - lo)
    gap = np.linalg.norm(states[pairs[:, 0]] - states[pairs[:, 1]], axis=1)
    separation = np.linalg.norm(slope[pairs[:, 0]] - slope[pairs[:, 1]], axis=1)
    label_scale = float(np.linalg.norm(labels[orders[k]], axis=1).mean())
    denominator = 2 * (label_error_fraction * label_scale + state_error) + budget * (
        gap + 2 * state_error
    )
    return separation / denominator


def overlap_availability(states: np.ndarray, trajectory_id: np.ndarray,
                         state_scale: float, stride: int = 9,
                         n_neighbors: int = 16, close_fraction: float = 0.05):
    """Distribution of distances to the nearest state on a different trajectory."""
    tree = cKDTree(states)
    nearest = []
    for i in range(0, len(states), stride):
        dist, idx = tree.query(states[i], k=n_neighbors)
        cross = [d for d, j in zip(dist[1:], idx[1:])
                 if trajectory_id[j] != trajectory_id[i]]
        if cross:
            nearest.append(min(cross))
    nearest = np.asarray(nearest) if nearest else np.array([np.inf])
    finite = np.isfinite(nearest)
    return dict(
        median=float(np.median(nearest[finite])) if finite.any() else float("nan"),
        fraction_close=float(np.mean(finite & (nearest <= close_fraction * state_scale))),
        fraction_without_neighbor=float(np.mean(~finite)),
    )


def relative_cross_trajectory_distance(trajectories: np.ndarray, start: int,
                                       stop: int, n_queries: int = 300,
                                       seed: int = 0) -> float:
    """Median nearest distance to another trajectory, divided by the state scale.

    Computed exactly by querying each trajectory separately, so no pair is missed
    when the trajectories are far apart.
    """
    rng = np.random.default_rng(seed)
    n_traj = trajectories.shape[1]
    scale = float(np.sqrt(np.mean(trajectories ** 2)))
    segments = [trajectories[start:stop, m, :] for m in range(n_traj)]
    trees = [cKDTree(seg) for seg in segments]
    values = []
    for m, n in zip(rng.integers(0, n_traj, n_queries),
                    rng.integers(0, stop - start, n_queries)):
        point = segments[m][n]
        values.append(min(trees[j].query(point)[0] for j in range(n_traj) if j != m))
    return float(np.median(values)) / scale


def effective_dimension(distances_by_count: dict) -> dict:
    """Dimension of the visited region from the scaling of overlap distances.

    For curves lying in a ``D``-dimensional region, the median distance to the
    nearest state on another trajectory decreases as ``M^{-1/(D-1)}`` in the number
    of trajectories ``M``. The slope on logarithmic axes therefore determines ``D``.
    """
    counts = np.array(sorted(distances_by_count))
    values = np.array([distances_by_count[c] for c in counts])
    slope = float(np.polyfit(np.log(counts), np.log(values), 1)[0])
    # D = 1 - 1/slope inverts the scaling law. The inversion is ill-conditioned
    # as the slope approaches zero, which happens when the overlap distance
    # barely responds to adding trajectories, so the slope is reported alongside
    # the dimension and the conversion is marked unreliable in that regime.
    reliable = slope < -0.05
    dimension = float(1.0 - 1.0 / slope) if reliable else float("nan")
    return dict(counts=counts.tolist(), distances=values.tolist(), slope=slope,
                dimension=dimension, reliable=bool(reliable))
