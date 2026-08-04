"""Sensitivity of the procedure to its settings, and comparison of discretizations.

Three questions are addressed on the same trajectories, changing only the
post-processing so that the comparison is not confounded by different data:
how the estimate and the identification set respond to the regularity budget,
whether a budget selected from data alone performs as well as one computed from
the true field, and how the estimate responds to the smoothing strength, the
sampling density, and the choice of Caputo discretization.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from hcoi.caputo import (
    fit_smoother,
    gauss_jacobi_derivative,
    l1_operator,
    smooth_on_grid,
)
from hcoi.identification import (
    candidate_labels,
    closest_cross_trajectory_pairs,
    identification_set,
    nearest_neighbor_pairs,
    rescale,
    violation_profile,
)
from hcoi.systems import NeuralVectorField, empirical_lipschitz, solve_caputo


def _combined_profile(orders, labels, states, close_pairs, neighbor_pairs, budget):
    parts = []
    for pairs, statistic in ((close_pairs, "quantile"), (neighbor_pairs, "max")):
        p = violation_profile(orders, labels, states, pairs, budget, statistic)
        if p is not None:
            parts.append(rescale(p))
    return np.mean(parts, axis=0) if parts else None


def _select_budget(orders, labels, states, close_pairs, reference_budget,
                   n_grid=25, tolerance=0.05):
    """Smallest budget at which the near-overlap profile attains a small minimum."""
    for budget in np.geomspace(0.05 * reference_budget, 8 * reference_budget, n_grid):
        p = violation_profile(orders, labels, states, close_pairs, budget,
                              "quantile")
        if p is not None and p.min() <= tolerance * (np.median(p) + 1e-300):
            return float(budget)
    return float(8 * reference_budget)


def run(true_order=0.60, horizon=1.0, n_steps=200, n_trajectories=16,
        n_replications=10, noise_levels=(0.01, 0.03), seed=7,
        output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    field = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    orders = np.round(np.arange(0.30, 0.9001, 0.025), 4)
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
    operators = {b: l1_operator(b, n_steps, step) for b in orders}

    x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = solve_caputo(field, x0, true_order, step, n_steps)
    state_scale = float(np.sqrt(np.mean(reference ** 2)))
    flat = reference.reshape(-1, field.dim)
    reference_budget = 1.2 * empirical_lipschitz(flat, field(flat), seed=seed)
    exact_labels = np.stack([field(reference[:, m, :])
                             for m in range(n_trajectories)], axis=1)

    multipliers = (0.25, 0.5, 1.0, 2.0, 4.0)
    budget_rows, smoothing_rows, discretization_rows = [], [], []

    for noise in noise_levels:
        sigma = noise * state_scale
        collected = {m: dict(errors=[], widths=[]) for m in multipliers}
        collected["data"] = dict(errors=[], widths=[], selected=[])
        for _ in range(n_replications):
            observations = reference + rng.normal(0.0, sigma, reference.shape)
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            labels, states, traj_id = candidate_labels(smoothed, operators, orders,
                                                       start, stop)
            times = np.concatenate([t[start:stop]] * n_trajectories)
            close_pairs = closest_cross_trajectory_pairs(states, traj_id)
            neighbor_pairs = nearest_neighbor_pairs(states, traj_id, times)
            for m in multipliers:
                profile = _combined_profile(orders, labels, states, close_pairs,
                                            neighbor_pairs, m * reference_budget)
                estimate = float(orders[int(np.argmin(profile))])
                _, _, width = identification_set(orders, profile)
                collected[m]["errors"].append(abs(estimate - true_order))
                collected[m]["widths"].append(width)
            selected = _select_budget(orders, labels, states, close_pairs,
                                      reference_budget)
            profile = _combined_profile(orders, labels, states, close_pairs,
                                        neighbor_pairs, selected)
            estimate = float(orders[int(np.argmin(profile))])
            _, _, width = identification_set(orders, profile)
            collected["data"]["errors"].append(abs(estimate - true_order))
            collected["data"]["widths"].append(width)
            collected["data"]["selected"].append(selected / reference_budget)

        row = {"noise": noise}
        for m in multipliers:
            row[f"mae_{m}x"] = float(np.mean(collected[m]["errors"]))
            row[f"width_{m}x"] = float(np.mean(collected[m]["widths"]))
        row["mae_data_driven"] = float(np.mean(collected["data"]["errors"]))
        row["width_data_driven"] = float(np.mean(collected["data"]["widths"]))
        row["selected_budget_ratio"] = float(np.mean(collected["data"]["selected"]))
        budget_rows.append(row)

    for noise in noise_levels:
        sigma = noise * state_scale
        for strength in (0.5, 1.0, 2.0):
            for density in (n_steps // 2, n_steps):
                stride = n_steps // density
                grid = np.linspace(0.0, horizon, density + 1)
                ops = {b: l1_operator(b, density, horizon / density) for b in orders}
                lo, hi = int(0.20 * density), int(0.95 * density)
                errors = []
                for _ in range(n_replications):
                    observations = reference[::stride] + rng.normal(
                        0.0, sigma, reference[::stride].shape)
                    smoothed = [smooth_on_grid(grid, observations[:, m, :], grid,
                                               sigma, strength)
                                for m in range(n_trajectories)]
                    labels, states, traj_id = candidate_labels(smoothed, ops, orders,
                                                               lo, hi)
                    times = np.concatenate([grid[lo:hi]] * n_trajectories)
                    profile = _combined_profile(
                        orders, labels, states,
                        closest_cross_trajectory_pairs(states, traj_id),
                        nearest_neighbor_pairs(states, traj_id, times),
                        reference_budget)
                    errors.append(abs(float(orders[int(np.argmin(profile))])
                                      - true_order))
                smoothing_rows.append({"noise": noise, "smoothing_strength": strength,
                                       "n_samples": density,
                                       "mae": float(np.mean(errors))})

    window = slice(start, stop)
    denominator = float(np.sqrt(np.mean(exact_labels[window] ** 2)))
    for noise in (0.0,) + tuple(noise_levels):
        sigma = noise * state_scale
        errors = {name: [] for name in ("l1", "gauss_jacobi", "l1_unsmoothed")}
        order_errors = {name: [] for name in errors}
        for _ in range(max(3, n_replications // 2)):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            splines = [fit_smoother(t, observations[:, m, :], sigma)
                       for m in range(n_trajectories)]
            raw = [observations[:, m, :] for m in range(n_trajectories)]

            per_method = {}
            per_method["l1"] = {
                b: np.concatenate([
                    (operators[b] @ _differences(smoothed[m]))[window]
                    for m in range(n_trajectories)])
                for b in orders}
            per_method["gauss_jacobi"] = {
                b: np.concatenate([
                    gauss_jacobi_derivative(t, splines[m], b)[window]
                    for m in range(n_trajectories)])
                for b in orders}
            per_method["l1_unsmoothed"] = {
                b: np.concatenate([
                    (operators[b] @ _differences(raw[m]))[window]
                    for m in range(n_trajectories)])
                for b in orders}

            states = np.concatenate([smoothed[m][window]
                                     for m in range(n_trajectories)])
            traj_id = np.concatenate([np.full(stop - start, m)
                                      for m in range(n_trajectories)])
            times = np.concatenate([t[window]] * n_trajectories)
            close_pairs = closest_cross_trajectory_pairs(states, traj_id)
            neighbor_pairs = nearest_neighbor_pairs(states, traj_id, times)
            truth = np.concatenate([exact_labels[window, m, :]
                                    for m in range(n_trajectories)])
            for name, labels in per_method.items():
                errors[name].append(
                    float(np.sqrt(np.mean((labels[true_order] - truth) ** 2))
                          / denominator))
                profile = _combined_profile(orders, labels, states, close_pairs,
                                            neighbor_pairs, reference_budget)
                order_errors[name].append(
                    abs(float(orders[int(np.argmin(profile))]) - true_order))
        discretization_rows.append({
            "noise": noise,
            **{f"reconstruction_error_{k}": float(np.mean(v))
               for k, v in errors.items()},
            **{f"order_mae_{k}": float(np.mean(v)) for k, v in order_errors.items()},
        })

    timing_rows = _timings(field, orders, horizon, seed)

    _write_csv(os.path.join(output_dir, "budget_sensitivity.csv"), budget_rows)
    _write_csv(os.path.join(output_dir, "smoothing_sensitivity.csv"), smoothing_rows)
    _write_csv(os.path.join(output_dir, "discretization_comparison.csv"),
               discretization_rows)
    _write_csv(os.path.join(output_dir, "wall_clock.csv"), timing_rows)
    return dict(budget=budget_rows, smoothing=smoothing_rows,
                discretization=discretization_rows, timing=timing_rows)


def _differences(x):
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return d


def _timings(field, orders, horizon, seed):
    rng = np.random.default_rng(seed + 1)
    rows = []
    for n_traj, n_steps in ((6, 100), (12, 200), (24, 200), (24, 400)):
        step = horizon / n_steps
        t = np.linspace(0.0, horizon, n_steps + 1)
        x0 = rng.normal(0.0, 0.45, (n_traj, field.dim))
        reference = solve_caputo(field, x0, 0.60, step, n_steps)
        sigma = 0.01 * float(np.sqrt(np.mean(reference ** 2)))
        observations = reference + rng.normal(0.0, sigma, reference.shape)
        for n_orders in (9, 17, 25, 41):
            grid = np.round(np.linspace(0.30, 0.90, n_orders), 4)
            clock = time.perf_counter()
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_traj)]
            t_smooth = time.perf_counter() - clock

            clock = time.perf_counter()
            ops = {b: l1_operator(b, n_steps, step) for b in grid}
            start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
            labels, states, traj_id = candidate_labels(smoothed, ops, grid,
                                                       start, stop)
            t_labels = time.perf_counter() - clock

            times = np.concatenate([t[start:stop]] * n_traj)
            clock = time.perf_counter()
            close_pairs = closest_cross_trajectory_pairs(states, traj_id)
            neighbor_pairs = nearest_neighbor_pairs(states, traj_id, times)
            t_pairs = time.perf_counter() - clock

            clock = time.perf_counter()
            violation_profile(grid, labels, states, close_pairs, 5.0, "quantile")
            violation_profile(grid, labels, states, neighbor_pairs, 5.0, "max")
            t_profile = time.perf_counter() - clock

            rows.append({
                "n_observations": n_traj * (n_steps + 1), "n_orders": n_orders,
                "smoothing_s": t_smooth, "labels_s": t_labels,
                "pair_selection_s": t_pairs, "profile_s": t_profile,
                "total_s": t_smooth + t_labels + t_pairs + t_profile,
            })
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    kwargs = dict(output_dir=args.output_dir)
    if args.quick:
        kwargs.update(n_steps=100, n_trajectories=10, n_replications=2,
                      noise_levels=(0.03,))
    result = run(**kwargs)
    for row in result["budget"]:
        print(f"noise {row['noise']:g}: MAE at reference budget "
              f"{row['mae_1.0x']:.3f}, data-driven {row['mae_data_driven']:.3f} "
              f"(selected {row['selected_budget_ratio']:.1f}x)")
    for row in result["discretization"]:
        print(f"noise {row['noise']:g}: reconstruction error "
              f"L1 {row['reconstruction_error_l1']:.4f}, "
              f"Gauss-Jacobi {row['reconstruction_error_gauss_jacobi']:.4f}, "
              f"unsmoothed {row['reconstruction_error_l1_unsmoothed']:.4f}")


if __name__ == "__main__":
    main()
