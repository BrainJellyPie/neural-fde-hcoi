"""Behavior of the estimator at the endpoints of the order range.

The theory assumes an order in a compact subset of the open unit interval. Two
tests probe what happens at the endpoints. The first generates ordinary dynamics,
whose order is one, in closed form and asks whether the estimator reports them as
ordinary. The second places the true order near zero and asks whether the reported
range still covers it.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from hcoi.caputo import l1_operator, smooth_on_grid
from hcoi.identification import (
    candidate_labels,
    early_time_slope,
    estimate_order,
    identification_set,
    rescale,
)
from hcoi.systems import (
    NeuralVectorField,
    collision_initial_states,
    empirical_lipschitz,
    linear_system_matrix,
    ordinary_flow,
    solve_caputo,
)


def integer_order_test(horizon=1.0, n_steps=240, n_trajectories=4,
                       n_replications=20, noise_levels=(0.0, 0.01, 0.03), seed=11):
    """Ordinary dynamics with the candidate grid extended to the integer order.

    At order one the L1 coefficients reduce to the backward difference, so the same
    pipeline applies without modification.
    """
    rng = np.random.default_rng(seed)
    A = linear_system_matrix()
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    orders = np.round(np.arange(0.20, 1.0001, 0.025), 4)
    operators = {b: l1_operator(b, n_steps, step) for b in orders}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    common_state = np.array([0.35, 0.15])
    visit_times = np.linspace(0.30, 0.90, n_trajectories)
    x0 = collision_initial_states(A, common_state, visit_times)
    reference = ordinary_flow(A, x0, t)
    state_scale = float(np.sqrt(np.mean(reference ** 2)))
    visit_index = np.clip(np.round(visit_times / step).astype(int), 1, n_steps)

    rows = []
    for noise in noise_levels:
        sigma = noise * state_scale
        errors, covered, slope_errors = [], [], []
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            profile = []
            for b in orders:
                labels = np.stack([
                    (operators[b] @ _differences(smoothed[m]))[visit_index[m]]
                    for m in range(n_trajectories)])
                profile.append(max(
                    np.linalg.norm(labels[i] - labels[j])
                    for i in range(n_trajectories)
                    for j in range(i + 1, n_trajectories)))
            profile = rescale(np.asarray(profile))
            estimate = float(orders[int(np.argmin(profile))])
            errors.append(abs(estimate - 1.0))
            _, upper, _ = identification_set(orders, profile)
            covered.append(float(upper >= 1.0 - 1e-9))
            slope_errors.append(abs(early_time_slope(t, observations[:, 0, :], x0[0])
                                    - 1.0))
        rows.append({"noise": noise,
                     "compatibility_mae": float(np.mean(errors)),
                     "identification_set_covers_one": float(np.mean(covered)),
                     "early_time_slope_mae": float(np.mean(slope_errors))})
    return rows


def near_zero_test(true_order=0.10, horizon=1.0, n_steps=200, n_trajectories=16,
                   n_replications=10, noise_levels=(0.0, 0.01, 0.03), seed=3):
    """A true order near the lower endpoint, with the grid extended downward."""
    rng = np.random.default_rng(seed)
    field = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    orders = np.round(np.arange(0.05, 0.4001, 0.025), 4)
    operators = {b: l1_operator(b, n_steps, step) for b in orders}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = solve_caputo(field, x0, true_order, step, n_steps)
    state_scale = float(np.sqrt(np.mean(reference ** 2)))
    flat = reference.reshape(-1, field.dim)
    budget = 1.2 * empirical_lipschitz(flat, field(flat), seed=seed)

    rows = []
    for noise in noise_levels:
        sigma = noise * state_scale
        errors, widths, covered, slope_errors = [], [], [], []
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            result = estimate_order(observations, t, orders, operators, budget,
                                    sigma, start, stop, variant="consensus")
            errors.append(abs(result["argmin"] - true_order))
            widths.append(result["width"])
            covered.append(float(result["lower"] - 1e-9 <= true_order
                                 <= result["upper"] + 1e-9))
            slope_errors.append(abs(early_time_slope(t, observations[:, 0, :], x0[0])
                                    - true_order))
        rows.append({"noise": noise,
                     "order_mae": float(np.mean(errors)),
                     "identification_set_covers_truth": float(np.mean(covered)),
                     "identification_set_width": float(np.mean(widths)),
                     "early_time_slope_mae": float(np.mean(slope_errors))})
    return rows


def _differences(x):
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return d


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def run(output_dir="results", quick=False):
    os.makedirs(output_dir, exist_ok=True)
    kwargs = dict(n_replications=4, noise_levels=(0.0, 0.03)) if quick else {}
    integer_rows = integer_order_test(**kwargs)
    near_zero_rows = near_zero_test(**({**kwargs, "n_trajectories": 10}
                                       if quick else {}))
    _write_csv(os.path.join(output_dir, "integer_order_boundary.csv"), integer_rows)
    _write_csv(os.path.join(output_dir, "near_zero_endpoint.csv"), near_zero_rows)
    return dict(integer_order=integer_rows, near_zero=near_zero_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run(args.output_dir, args.quick)
    for row in result["integer_order"]:
        print(f"integer order, noise {row['noise']:g}: MAE "
              f"{row['compatibility_mae']:.3f}, set covers one "
              f"{row['identification_set_covers_one']:.2f}")
    for row in result["near_zero"]:
        print(f"near zero, noise {row['noise']:g}: MAE {row['order_mae']:.3f}, "
              f"set covers truth {row['identification_set_covers_truth']:.2f}, "
              f"width {row['identification_set_width']:.3f}")


if __name__ == "__main__":
    main()
