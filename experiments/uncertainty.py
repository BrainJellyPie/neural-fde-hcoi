"""Statistical uncertainty for the comparisons reported in the earlier tables.

Two questions are addressed. The first is whether the proposed estimator and
the joint fitting of the order with a nonlinear field differ by more than the
replication noise, which requires the two to be evaluated on the same
replications and compared as a paired difference. The second is how precise the
Monte Carlo results of the earlier sections are, which requires intervals on
every reported mean rather than the mean alone.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from hcoi.caputo import (
    fit_smoother,
    fractional_integral_operator,
    gauss_jacobi_derivative,
    l1_operator,
    smooth_on_grid,
)
from hcoi.identification import (
    affine_profile_order,
    candidate_labels,
    covers,
    early_time_slope,
    estimate_order,
    identification_set,
    joint_field_fit,
    rescale,
)
from hcoi.reporting import bootstrap_interval, selective_risk
from hcoi.systems import (
    LinearVectorField,
    NeuralVectorField,
    collision_initial_states,
    fractional_collision_states,
    mittag_leffler_flow,
    empirical_lipschitz,
    linear_system_matrix,
    ordinary_flow,
    solve_caputo,
)

ORDERS = np.round(np.arange(0.30, 0.9001, 0.025), 4)
TOL = 0.05 + 1e-9


def _differences(x):
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return d


# --------------------------------------------------------------------------- #
# 1. Paired comparison against joint fitting
# --------------------------------------------------------------------------- #
def paired_joint_fitting(true_order=0.60, horizon=1.0, n_steps=200,
                         n_trajectories=16, n_replications=30,
                         noise_levels=(0.0, 0.01, 0.03), n_features=64,
                         seed=0, output_dir="results"):
    """The proposed estimator and joint fitting on identical replications.

    Both estimators see the same noisy trajectories in each replication, so the
    difference between them can be reported as a paired quantity. Without this
    the two means are not directly comparable, and a difference of a few
    thousandths cannot be distinguished from replication noise.
    """
    rng = np.random.default_rng(seed)
    field = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    operators = {b: l1_operator(b, n_steps, step) for b in ORDERS}
    integrals = {b: fractional_integral_operator(b, n_steps, step) for b in ORDERS}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = solve_caputo(field, x0, true_order, step, n_steps)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    flat = reference.reshape(-1, field.dim)
    budget = 1.2 * empirical_lipschitz(flat, field(flat), seed=seed)

    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        proposed, joint = [], []
        for rep in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            r = estimate_order(observations, t, ORDERS, operators, budget, sigma,
                               start, stop, variant="consensus")
            proposed.append(abs(r["argmin"] - true_order))
            targets = [x - x[0][None, :] for x in smoothed]
            errors, _, _ = joint_field_fit(ORDERS, smoothed, targets, integrals,
                                           n_features, seed=10 + rep)
            joint.append(abs(float(ORDERS[int(np.argmin(errors))]) - true_order))
        proposed = np.asarray(proposed)
        joint = np.asarray(joint)
        difference = joint - proposed
        row = dict(noise=noise, n_replications=n_replications)
        for name, values in (("proposed", proposed), ("joint_fitting", joint)):
            lo, hi = bootstrap_interval(values)
            row[f"{name}_mae"] = float(values.mean())
            row[f"{name}_sd"] = float(values.std(ddof=1))
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
        lo, hi = bootstrap_interval(difference)
        row["paired_difference"] = float(difference.mean())
        row["paired_ci_low"], row["paired_ci_high"] = lo, hi
        # The difference is distinguishable from replication noise only when the
        # interval excludes zero.
        row["separated"] = bool(lo > 0 or hi < 0)
        rows.append(row)
    _write_csv(os.path.join(output_dir, "paired_joint_fitting.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# 2. Intervals for the linear Monte Carlo study
# --------------------------------------------------------------------------- #
def linear_uncertainty(true_order=0.65, horizon=1.0, n_steps=240,
                       n_trajectories=4, n_replications=60,
                       noise_levels=(0.0, 0.0025, 0.01, 0.03, 0.05, 0.08),
                       seed=17, output_dir="results"):
    """Intervals for the estimators compared on the linear system."""
    rng = np.random.default_rng(seed)
    A = linear_system_matrix()
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    operators = {b: l1_operator(b, n_steps, step) for b in ORDERS}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    common = np.array([0.35, 0.15])
    visit = np.linspace(0.30, 0.90, n_trajectories)
    x0 = fractional_collision_states(A, common, visit, true_order)
    reference = mittag_leffler_flow(A, x0, t, true_order)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    budget = 1.2 * float(np.linalg.norm(A, 2))
    index = np.clip(np.round(visit / step).astype(int), 1, n_steps)

    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        acc = {k: [] for k in ("early", "affine", "designed")}
        ab, cov = [], []
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            labels, states, _ = candidate_labels(smoothed, operators, ORDERS,
                                                 start, stop)
            acc["early"].append(abs(early_time_slope(t, observations[:, 0, :], x0[0])
                                    - true_order))
            acc["affine"].append(abs(affine_profile_order(ORDERS, labels, states)
                                     - true_order))
            profile = []
            for b in ORDERS:
                lab = np.stack([(operators[b] @ _differences(smoothed[m]))[index[m]]
                                for m in range(n_trajectories)])
                profile.append(max(
                    np.linalg.norm(lab[i] - lab[j])
                    for i in range(n_trajectories)
                    for j in range(i + 1, n_trajectories)))
            profile = rescale(np.asarray(profile))
            acc["designed"].append(abs(float(ORDERS[int(np.argmin(profile))])
                                       - true_order))
            low, up, width = identification_set(ORDERS, profile)
            ab.append(width > 0.30)
            cov.append(bool(low - 1e-9 <= true_order <= up + 1e-9))
        row = dict(noise=noise, n_replications=n_replications)
        for name, values in acc.items():
            values = np.asarray(values)
            lo, hi = bootstrap_interval(values)
            row[f"{name}_mae"] = float(values.mean())
            row[f"{name}_sd"] = float(values.std(ddof=1))
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
            row[f"{name}_success"] = float((values <= TOL).mean())
        row["abstention"] = float(np.mean(ab))
        row["set_coverage"] = float(np.mean(cov))
        rows.append(row)
    _write_csv(os.path.join(output_dir, "linear_uncertainty.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# 3. Intervals for the discretization comparison
# --------------------------------------------------------------------------- #
def discretization_uncertainty(true_order=0.60, horizon=1.0, n_steps=200,
                               n_trajectories=16, n_replications=20,
                               noise_levels=(0.0, 0.01, 0.03), seed=23,
                               output_dir="results"):
    """Intervals on the reconstruction error of each Caputo discretization."""
    rng = np.random.default_rng(seed)
    field = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    operators = {b: l1_operator(b, n_steps, step) for b in ORDERS}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
    window = slice(start, stop)

    x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = solve_caputo(field, x0, true_order, step, n_steps)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    exact = np.stack([field(reference[:, m, :]) for m in range(n_trajectories)],
                     axis=1)
    denominator = float(np.sqrt(np.mean(exact[window] ** 2)))

    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        acc = {k: [] for k in ("l1", "gauss_jacobi", "l1_unsmoothed")}
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            splines = [fit_smoother(t, observations[:, m, :], sigma)
                       for m in range(n_trajectories)]
            truth = np.concatenate([exact[window, m, :]
                                    for m in range(n_trajectories)])
            per = {
                "l1": np.concatenate([
                    (operators[true_order] @ _differences(smoothed[m]))[window]
                    for m in range(n_trajectories)]),
                "gauss_jacobi": np.concatenate([
                    gauss_jacobi_derivative(t, splines[m], true_order)[window]
                    for m in range(n_trajectories)]),
                "l1_unsmoothed": np.concatenate([
                    (operators[true_order] @ _differences(observations[:, m, :]))[window]
                    for m in range(n_trajectories)]),
            }
            for name, values in per.items():
                acc[name].append(float(np.sqrt(np.mean((values - truth) ** 2))
                                       / denominator))
        row = dict(noise=noise, n_replications=n_replications)
        for name, values in acc.items():
            values = np.asarray(values)
            lo, hi = bootstrap_interval(values)
            row[f"{name}_error"] = float(values.mean())
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
        rows.append(row)
    _write_csv(os.path.join(output_dir, "discretization_uncertainty.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# 4. Intervals for the integer-order boundary test
# --------------------------------------------------------------------------- #
def boundary_uncertainty(horizon=1.0, n_steps=240, n_trajectories=4,
                         n_replications=40, noise_levels=(0.0, 0.01, 0.03),
                         seed=31, output_dir="results"):
    """Intervals for the test that places the true order at the integer endpoint."""
    rng = np.random.default_rng(seed)
    A = linear_system_matrix()
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    orders = np.round(np.arange(0.20, 1.0001, 0.025), 4)
    operators = {b: l1_operator(b, n_steps, step) for b in orders}

    common = np.array([0.35, 0.15])
    visit = np.linspace(0.30, 0.90, n_trajectories)
    x0 = collision_initial_states(A, common, visit)
    reference = ordinary_flow(A, x0, t)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    index = np.clip(np.round(visit / step).astype(int), 1, n_steps)

    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        errors, covered, slopes = [], [], []
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                        for m in range(n_trajectories)]
            profile = []
            for b in orders:
                labels = np.stack([
                    (operators[b] @ _differences(smoothed[m]))[index[m]]
                    for m in range(n_trajectories)])
                profile.append(max(
                    np.linalg.norm(labels[i] - labels[j])
                    for i in range(n_trajectories)
                    for j in range(i + 1, n_trajectories)))
            profile = rescale(np.asarray(profile))
            errors.append(abs(float(orders[int(np.argmin(profile))]) - 1.0))
            _, upper, _ = identification_set(orders, profile)
            covered.append(float(upper >= 1.0 - 1e-9))
            slopes.append(abs(early_time_slope(t, observations[:, 0, :], x0[0]) - 1.0))
        row = dict(noise=noise, n_replications=n_replications)
        for name, values in (("compatibility", errors), ("early_time", slopes)):
            values = np.asarray(values)
            lo, hi = bootstrap_interval(values)
            row[f"{name}_mae"] = float(values.mean())
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
        row["contains_integer_order"] = float(np.mean(covered))
        lo, hi = bootstrap_interval(covered)
        row["contains_ci_low"], row["contains_ci_high"] = lo, hi
        rows.append(row)
    _write_csv(os.path.join(output_dir, "boundary_uncertainty.csv"), rows)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{row[k]:.4f}" if isinstance(row.get(k), float)
                                 else row.get(k, "")) for k in keys})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    q = args.quick
    started = time.time()

    print("[1] paired comparison against joint fitting", flush=True)
    paired_joint_fitting(n_replications=3 if q else 30,
                         noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03),
                         output_dir=args.output_dir)
    print(f"    done {time.time() - started:.0f}s", flush=True)

    print("[2] linear system, with intervals", flush=True)
    linear_uncertainty(n_replications=4 if q else 60,
                       noise_levels=(0.0, 0.03) if q else
                       (0.0, 0.0025, 0.01, 0.03, 0.05, 0.08),
                       output_dir=args.output_dir)
    print(f"    done {time.time() - started:.0f}s", flush=True)

    print("[3] discretization, with intervals", flush=True)
    discretization_uncertainty(n_replications=3 if q else 20,
                               noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03),
                               output_dir=args.output_dir)
    print(f"    done {time.time() - started:.0f}s", flush=True)

    print("[4] integer-order boundary, with intervals", flush=True)
    boundary_uncertainty(n_replications=4 if q else 40,
                         noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03),
                         output_dir=args.output_dir)
    print(f"    done {time.time() - started:.0f}s", flush=True)

    print(f"\nResults written to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
