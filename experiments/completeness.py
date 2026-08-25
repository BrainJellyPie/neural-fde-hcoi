"""Settings that the procedure fixes in advance, and the cost of each stage.

The studies here close the remaining gaps between what the procedure is defined
to do and what is measured. The error threshold of the compatibility score is
swept rather than assumed, the designed-collision study on the nonlinear system
is repeated with intervals, the cost is resolved by stage rather than in total,
the settings held fixed throughout the paper are varied one at a time, and the
effective-dimension estimate is recomputed over different ranges of the number
of trajectories.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from hcoi.caputo import l1_operator, smooth_on_grid
from hcoi.diagnostics import effective_dimension, relative_cross_trajectory_distance
from hcoi.identification import (
    affine_profile_order,
    candidate_labels,
    closest_cross_trajectory_pairs,
    covers,
    early_time_slope,
    estimate_order,
    nearest_neighbor_pairs,
    violation_profile,
)
from hcoi.reporting import bootstrap_interval, selective_risk
from hcoi.systems import (
    DissipativeVectorField,
    NeuralVectorField,
    empirical_lipschitz,
    shoot_collision_states,
    solve_caputo,
)

ORDERS = np.round(np.arange(0.30, 0.9001, 0.025), 4)
TOL = 0.05 + 1e-9


def _neural_setup(true_order=0.60, horizon=1.0, n_steps=200, n_trajectories=16,
                  seed=0, designed=False, n_designed=8):
    rng = np.random.default_rng(seed)
    field = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    operators = {b: l1_operator(b, n_steps, step) for b in ORDERS}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
    if designed:
        visit = np.linspace(0.30, 0.90, n_designed)
        x0 = shoot_collision_states(field, np.array([0.35, 0.15]), visit,
                                    true_order, step, n_steps)
        n_trajectories = n_designed
    else:
        x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = solve_caputo(field, x0, true_order, step, n_steps)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    flat = reference.reshape(-1, field.dim)
    budget = 1.2 * empirical_lipschitz(flat, field(flat), seed=seed)
    return dict(rng=rng, field=field, t=t, step=step, operators=operators,
                start=start, stop=stop, x0=x0, reference=reference, scale=scale,
                budget=budget, n_trajectories=n_trajectories, n_steps=n_steps,
                horizon=horizon)


# --------------------------------------------------------------------------- #
# 1. Error threshold of the compatibility score
# --------------------------------------------------------------------------- #
def threshold_study(true_order=0.60, n_replications=25,
                    noise_levels=(0.01, 0.03), multipliers=(0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0),
                    seed=0, output_dir="results"):
    """Sensitivity to the error threshold subtracted from the pairwise violation.

    The score of Section VII subtracts a threshold that bounds the pairwise error
    from state and derivative estimation. A reference value is formed from the
    observation noise and the smoothing spread, and the sweep reports what the
    estimate and the reported set do as the threshold is scaled around it. A
    threshold larger than the true error level can only push the profile toward
    zero, so it widens the reported set without moving the minimum.
    """
    cfg = _neural_setup(true_order=true_order, seed=seed)
    rng, reference, scale = cfg["rng"], cfg["reference"], cfg["scale"]
    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        reference_threshold = _reference_threshold(cfg, sigma)
        acc = {m: dict(err=[], ab=[], cov=[], w=[]) for m in multipliers}
        for _ in range(n_replications):
            observations = reference + rng.normal(0.0, sigma, reference.shape)
            for m in multipliers:
                r = estimate_order(observations, cfg["t"], ORDERS, cfg["operators"],
                                   cfg["budget"], sigma, cfg["start"], cfg["stop"],
                                   variant="consensus",
                                   threshold=m * reference_threshold)
                acc[m]["err"].append(abs(r["argmin"] - true_order))
                acc[m]["ab"].append(r["abstained"])
                acc[m]["cov"].append(covers(r, true_order))
                acc[m]["w"].append(r["width"])
        for m in multipliers:
            e = acc[m]
            metrics = selective_risk(e["err"], e["ab"], covered=e["cov"], widths=e["w"])
            rows.append(dict(threshold=f"{m}x", noise=noise,
                             reference_threshold=reference_threshold, **metrics))
    _write_csv(os.path.join(output_dir, "threshold_study.csv"), rows)
    return rows


def _reference_threshold(cfg, sigma, strengths=(0.7, 1.0, 1.4)):
    """Threshold formed from the observation noise and the smoothing spread.

    The state error is taken as the observation noise level and the derivative
    error as the spread of the candidate labels across smoothing strengths, both
    of which are available without knowing the true field.
    """
    rng = np.random.default_rng(0)
    observations = cfg["reference"] + (
        rng.normal(0.0, sigma, cfg["reference"].shape) if sigma > 0 else 0.0)
    labels_by_strength = []
    for strength in strengths:
        smoothed = [smooth_on_grid(cfg["t"], observations[:, m, :], cfg["t"],
                                   sigma, strength)
                    for m in range(cfg["n_trajectories"])]
        labels, _, _ = candidate_labels(smoothed, cfg["operators"], ORDERS,
                                        cfg["start"], cfg["stop"])
        labels_by_strength.append(np.stack([labels[b] for b in ORDERS]))
    stacked = np.stack(labels_by_strength)
    derivative_error = float(np.mean(np.linalg.norm(
        stacked.max(axis=0) - stacked.min(axis=0), axis=-1)))
    return float(2.0 * derivative_error + 2.0 * cfg["budget"] * sigma)


# --------------------------------------------------------------------------- #
# 2. Designed collisions on the nonlinear system, with intervals
# --------------------------------------------------------------------------- #
def designed_collision_uncertainty(true_order=0.60, n_replications=30,
                                   noise_levels=(0.0, 0.01, 0.03, 0.05, 0.08),
                                   seed=13, output_dir="results"):
    """Intervals for the designed-collision study on the nonlinear neural field.

    The initial states are obtained by Newton shooting, so that eight histories
    pass through one common state at distinct times to machine precision.
    """
    cfg = _neural_setup(true_order=true_order, seed=seed, designed=True)
    rng, reference, scale = cfg["rng"], cfg["reference"], cfg["scale"]
    n_traj = cfg["n_trajectories"]
    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        acc = {k: [] for k in ("early", "affine", "consensus")}
        ab, cov = [], []
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(cfg["t"], observations[:, m, :], cfg["t"], sigma)
                        for m in range(n_traj)]
            labels, states, _ = candidate_labels(smoothed, cfg["operators"], ORDERS,
                                                 cfg["start"], cfg["stop"])
            acc["early"].append(abs(early_time_slope(cfg["t"], observations[:, 0, :],
                                                     cfg["x0"][0]) - true_order))
            acc["affine"].append(abs(affine_profile_order(ORDERS, labels, states)
                                     - true_order))
            r = estimate_order(observations, cfg["t"], ORDERS, cfg["operators"],
                               cfg["budget"], sigma, cfg["start"], cfg["stop"],
                               variant="consensus")
            acc["consensus"].append(abs(r["argmin"] - true_order))
            ab.append(r["abstained"]); cov.append(covers(r, true_order))
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
    _write_csv(os.path.join(output_dir, "designed_collision_uncertainty.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# 3. Cost resolved by stage
# --------------------------------------------------------------------------- #
def stage_timing(horizon=1.0, n_repeats=5, seed=9, output_dir="results"):
    """Cost of each stage separately, including the data-driven budget search."""
    field = NeuralVectorField(seed=1)
    rng = np.random.default_rng(seed)
    rows = []
    for n_traj, n_steps in ((6, 100), (12, 200), (24, 200), (24, 400)):
        step = horizon / n_steps
        t = np.linspace(0.0, horizon, n_steps + 1)
        x0 = rng.normal(0.0, 0.45, (n_traj, field.dim))
        reference = solve_caputo(field, x0, 0.60, step, n_steps)
        sigma = 0.01 * float(np.sqrt(np.mean(reference ** 2)))
        observations = reference + rng.normal(0.0, sigma, reference.shape)
        for n_orders in (9, 25, 41):
            grid = np.round(np.linspace(0.30, 0.90, n_orders), 4)
            start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
            stages = {k: [] for k in ("smoothing", "operators", "labels",
                                      "pairs", "profile", "budget_search")}
            for _ in range(n_repeats):
                clock = time.perf_counter()
                smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                            for m in range(n_traj)]
                stages["smoothing"].append(time.perf_counter() - clock)

                clock = time.perf_counter()
                ops = {b: l1_operator(b, n_steps, step) for b in grid}
                stages["operators"].append(time.perf_counter() - clock)

                clock = time.perf_counter()
                labels, states, traj_id = candidate_labels(smoothed, ops, grid,
                                                           start, stop)
                stages["labels"].append(time.perf_counter() - clock)

                times = np.concatenate([t[start:stop]] * n_traj)
                clock = time.perf_counter()
                close_pairs = closest_cross_trajectory_pairs(states, traj_id)
                neighbor_pairs = nearest_neighbor_pairs(states, traj_id, times)
                stages["pairs"].append(time.perf_counter() - clock)

                clock = time.perf_counter()
                violation_profile(grid, labels, states, close_pairs, 5.0, "quantile")
                violation_profile(grid, labels, states, neighbor_pairs, 5.0, "max")
                stages["profile"].append(time.perf_counter() - clock)

                clock = time.perf_counter()
                for budget in np.geomspace(0.25, 40.0, 25):
                    p = violation_profile(grid, labels, states, close_pairs, budget,
                                          "quantile")
                    if p is not None and p.min() <= 0.05 * (np.median(p) + 1e-300):
                        break
                stages["budget_search"].append(time.perf_counter() - clock)

            row = dict(n_observations=n_traj * (n_steps + 1), n_orders=n_orders)
            total = 0.0
            for name, values in stages.items():
                row[f"{name}_s"] = float(np.mean(values))
                row[f"{name}_sd"] = float(np.std(values, ddof=1))
                if name != "budget_search":
                    total += float(np.mean(values))
            row["total_s"] = total
            row["budget_search_share"] = row["budget_search_s"] / total
            rows.append(row)
    _write_csv(os.path.join(output_dir, "stage_timing.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# 4. Settings held fixed throughout the paper
# --------------------------------------------------------------------------- #
def settings_sensitivity(true_order=0.60, n_replications=20, noise=0.03,
                         seed=0, output_dir="results"):
    """One-at-a-time variation of the settings the procedure fixes in advance."""
    cfg = _neural_setup(true_order=true_order, seed=seed)
    rng, reference, scale = cfg["rng"], cfg["reference"], cfg["scale"]
    sigma = noise * scale
    defaults = dict(n_neighbors=8, history_separation=0.15, n_closest_pairs=30,
                    quantile=0.9, set_tolerance=0.15, abstain_width=0.30)
    sweeps = {
        "n_neighbors": (4, 6, 8, 12, 16),
        "history_separation": (0.05, 0.10, 0.15, 0.25),
        "n_closest_pairs": (10, 20, 30, 50, 80),
        "quantile": (0.75, 0.8, 0.9, 0.95, 1.0),
        "set_tolerance": (0.05, 0.10, 0.15, 0.25),
        "abstain_width": (0.20, 0.30, 0.40, 0.60),
    }
    observations = [reference + rng.normal(0.0, sigma, reference.shape)
                    for _ in range(n_replications)]
    rows = []
    for name, values in sweeps.items():
        for value in values:
            kwargs = dict(defaults)
            kwargs[name] = value
            err, ab, cov, wid = [], [], [], []
            for obs in observations:
                r = estimate_order(obs, cfg["t"], ORDERS, cfg["operators"],
                                   cfg["budget"], sigma, cfg["start"], cfg["stop"],
                                   variant="consensus", **kwargs)
                err.append(abs(r["argmin"] - true_order))
                ab.append(r["abstained"]); cov.append(covers(r, true_order))
                wid.append(r["width"])
            metrics = selective_risk(err, ab, covered=cov, widths=wid)
            rows.append(dict(setting=name, value=value,
                             is_default=(value == defaults[name]), **metrics))
    _write_csv(os.path.join(output_dir, "settings_sensitivity.csv"), rows)
    return rows


# --------------------------------------------------------------------------- #
# 5. Effective dimension over different ranges of the trajectory count
# --------------------------------------------------------------------------- #
CONFIGURATIONS = [
    ("d=2, non-dissipative", 2, [0.0, 0.0]),
    ("d=4, non-dissipative", 4, [0.0] * 4),
    ("d=4, slow manifold", 4, [0.2, 0.2, 4.0, 4.0]),
    ("d=6, strong slow manifold", 6, [0.2, 0.2, 12.0, 12.0, 12.0, 12.0]),
]


def dimension_range_sensitivity(true_order=0.60, horizon=1.0, n_steps=150,
                                seed=21, output_dir="results"):
    """Effective dimension estimated over different ranges of the trajectory count.

    The scaling argument is asymptotic in the number of trajectories, so the
    estimate is recomputed on a low range, a high range, and the full range to
    show how much of the reported value depends on that choice.
    """
    step = horizon / n_steps
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
    ranges = {"8-32": (8, 16, 32), "16-64": (16, 32, 64), "8-64": (8, 16, 32, 64)}
    rows = []
    for label, dim, damping in CONFIGURATIONS:
        field = DissipativeVectorField(dim, damping, seed=1)
        distances = {}
        for count in (8, 16, 32, 64):
            rng = np.random.default_rng(100 + count)
            x0 = rng.normal(0.0, 0.45, (count, dim))
            trajectory = solve_caputo(field, x0, true_order, step, n_steps)
            distances[count] = relative_cross_trajectory_distance(trajectory,
                                                                  start, stop)
        row = dict(system=label, state_dimension=dim)
        for name, counts in ranges.items():
            subset = {c: distances[c] for c in counts}
            row[f"dimension_{name}"] = effective_dimension(subset)["dimension"]
        values = [row[f"dimension_{n}"] for n in ranges]
        row["spread"] = float(max(values) - min(values))
        rows.append(row)
    _write_csv(os.path.join(output_dir, "dimension_range_sensitivity.csv"), rows)
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

    def stamp(name):
        print(f"    {name} done at {time.time() - started:.0f}s", flush=True)

    print("[1] error threshold of the score", flush=True)
    threshold_study(n_replications=3 if q else 25,
                    noise_levels=(0.03,) if q else (0.01, 0.03),
                    output_dir=args.output_dir)
    stamp("threshold")

    print("[2] designed collisions on the nonlinear system", flush=True)
    designed_collision_uncertainty(
        n_replications=3 if q else 30,
        noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03, 0.05, 0.08),
        output_dir=args.output_dir)
    stamp("designed collisions")

    print("[3] cost by stage", flush=True)
    stage_timing(n_repeats=2 if q else 5, output_dir=args.output_dir)
    stamp("stage timing")

    print("[4] settings held fixed", flush=True)
    settings_sensitivity(n_replications=3 if q else 20, output_dir=args.output_dir)
    stamp("settings")

    print("[5] effective dimension over ranges of M", flush=True)
    dimension_range_sensitivity(output_dir=args.output_dir)
    stamp("dimension ranges")

    print(f"\nResults written to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
