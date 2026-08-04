"""Order recovery from random histories, with every pair selected automatically.

Initial states are drawn at random on the nonlinear neural system and no state
overlap is constructed, so any near-overlap that the estimator uses arises on its
own. The script reports the accuracy of two variants of the procedure, the
availability and informativeness of the overlaps that occur, and the effect of
retaining only a random subset of the observation times.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from hcoi.caputo import l1_operator, smooth_on_grid
from hcoi.diagnostics import overlap_availability, transversality_ratio
from hcoi.identification import (
    affine_profile_order,
    candidate_labels,
    closest_cross_trajectory_pairs,
    early_time_slope,
    estimate_order,
)
from hcoi.systems import NeuralVectorField, empirical_lipschitz, solve_caputo

SUCCESS_TOLERANCE = 0.05 + 1e-4


def run(true_order=0.60, horizon=1.0, n_steps=200, n_trajectories=16,
        n_replications=14, noise_levels=(0.0, 0.01, 0.03, 0.05),
        retention_levels=(0.70, 0.40), seed=0, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    field = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    orders = np.round(np.arange(0.30, 0.9001, 0.025), 4)
    operators = {b: l1_operator(b, n_steps, step) for b in orders}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = solve_caputo(field, x0, true_order, step, n_steps)
    state_scale = float(np.sqrt(np.mean(reference ** 2)))

    flat = reference.reshape(-1, field.dim)
    budget = 1.2 * empirical_lipschitz(flat, field(flat), seed=seed)

    accuracy_rows, diagnostic_rows, sampling_rows = [], [], []

    for noise in noise_levels:
        sigma = noise * state_scale
        record = {name: dict(errors=[], hits=[], abstained=0)
                  for name in ("early", "affine", "neighbors", "consensus")}
        availability, informative_counts = [], []

        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0
            )
            for name, variant in (("neighbors", "neighbors"),
                                  ("consensus", "consensus")):
                result = estimate_order(observations, t, orders, operators, budget,
                                        sigma, start, stop, variant=variant)
                if result["abstained"]:
                    record[name]["abstained"] += 1
                    continue
                error = round(abs(result["order"] - true_order), 4)
                record[name]["errors"].append(error)
                record[name]["hits"].append(error <= SUCCESS_TOLERANCE)

            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            labels, states, traj_id = candidate_labels(smoothed, operators, orders,
                                                       start, stop)
            for name, value in (
                ("early", early_time_slope(t, observations[:, 0, :], x0[0])),
                ("affine", affine_profile_order(orders, labels, states)),
            ):
                error = round(abs(value - true_order), 4)
                record[name]["errors"].append(error)
                record[name]["hits"].append(error <= SUCCESS_TOLERANCE)

            availability.append(overlap_availability(states, traj_id, state_scale))
            close_pairs = closest_cross_trajectory_pairs(states, traj_id)
            ratio = transversality_ratio(orders, labels, states, close_pairs,
                                         true_order, budget, sigma)
            informative_counts.append(int(np.sum(ratio > 1.0)) if len(ratio) else 0)

        row = {"noise": noise}
        for name in ("early", "affine", "neighbors", "consensus"):
            e = record[name]
            total = len(e["errors"]) + e["abstained"]
            row[f"{name}_mae"] = float(np.mean(e["errors"])) if e["errors"] else float("nan")
            row[f"{name}_success"] = float(np.mean(e["hits"])) if e["hits"] else float("nan")
            row[f"{name}_abstention"] = e["abstained"] / total if total else float("nan")
        accuracy_rows.append(row)

        diagnostic_rows.append({
            "noise": noise,
            "median_cross_distance": float(np.mean([a["median"] for a in availability])),
            "fraction_close": float(np.mean([a["fraction_close"] for a in availability])),
            "informative_pairs": float(np.mean(informative_counts)),
        })

    for retention in retention_levels:
        for noise in (0.01, 0.03):
            sigma = noise * state_scale
            errors, hits, abstained = [], [], 0
            for _ in range(max(4, n_replications // 2)):
                observations = reference + rng.normal(0.0, sigma, reference.shape)
                resampled = np.empty_like(observations)
                for m in range(n_trajectories):
                    keep = np.sort(rng.choice(np.arange(1, n_steps),
                                              size=int(retention * (n_steps - 1)),
                                              replace=False))
                    keep = np.concatenate([[0], keep, [n_steps]])
                    resampled[:, m, :] = smooth_on_grid(t[keep],
                                                        observations[keep, m, :],
                                                        t, sigma)
                result = estimate_order(resampled, t, orders, operators, budget,
                                        sigma, start, stop, variant="consensus")
                if result["abstained"]:
                    abstained += 1
                    continue
                error = round(abs(result["order"] - true_order), 4)
                errors.append(error)
                hits.append(error <= SUCCESS_TOLERANCE)
            total = len(errors) + abstained
            sampling_rows.append({
                "retention": retention, "noise": noise,
                "mae": float(np.mean(errors)) if errors else float("nan"),
                "success": float(np.mean(hits)) if hits else float("nan"),
                "abstention": abstained / total if total else float("nan"),
            })

    _write_csv(os.path.join(output_dir, "random_histories_accuracy.csv"), accuracy_rows)
    _write_csv(os.path.join(output_dir, "random_histories_diagnostics.csv"), diagnostic_rows)
    _write_csv(os.path.join(output_dir, "random_histories_sampling.csv"), sampling_rows)
    return dict(accuracy=accuracy_rows, diagnostics=diagnostic_rows,
                sampling=sampling_rows, budget=budget)


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
    parser.add_argument("--quick", action="store_true",
                        help="reduced settings for a fast check")
    args = parser.parse_args()
    kwargs = dict(output_dir=args.output_dir)
    if args.quick:
        kwargs.update(n_steps=100, n_trajectories=10, n_replications=3,
                      noise_levels=(0.0, 0.03), retention_levels=(0.70,))
    result = run(**kwargs)
    for row in result["accuracy"]:
        print(f"noise {row['noise']:g}: "
              f"neighbors MAE {row['neighbors_mae']:.3f} "
              f"(abstain {row['neighbors_abstention']:.2f}), "
              f"consensus MAE {row['consensus_mae']:.3f} "
              f"(abstain {row['consensus_abstention']:.2f})")


if __name__ == "__main__":
    main()
