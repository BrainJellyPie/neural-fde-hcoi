"""Reliability of the reported output, and behavior outside the model class.

Four questions are addressed. The first is how the procedure should be scored
when it may decline to answer, which requires separating how often it reports
from how accurate it is when it does. The second is how the reported order and
the reported set respond to the regularity budget when the same estimator is
used throughout. The third is what the procedure does when the data are
generated outside the single-order autonomous class it assumes. The fourth is
how precisely the effective dimension of the visited region can be measured.

Every quantity is reported with a bootstrap interval over replications.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time

import numpy as np

from hcoi.caputo import fractional_integral_operator, l1_operator, smooth_on_grid
from hcoi.diagnostics import (
    effective_dimension,
    relative_cross_trajectory_distance,
    transversality_ratio,
)
from hcoi.identification import (
    candidate_labels,
    closest_cross_trajectory_pairs,
    covers,
    early_time_slope,
    estimate_order,
    grid_search_order,
    nearest_neighbor_pairs,
    rescale,
    violation_profile,
    identification_set,
)
from hcoi.plotting import color, plt, save
from hcoi.reporting import bootstrap_interval, selective_risk
from hcoi.systems import (
    DissipativeVectorField,
    linear_system_matrix,
    ForcedVectorField,
    NeuralVectorField,
    empirical_lipschitz,
    solve_caputo,
    solve_caputo_forced,
    solve_caputo_variable_order,
)

ORDERS = np.round(np.arange(0.30, 0.9001, 0.025), 4)

# Every study below uses the same trajectory configuration, obtained from this
# seed, so that the tables are directly comparable and differ only in the
# quantity under study.
CONFIGURATION_SEED = 0

# Abstention threshold on the feasibility residual, fixed once on the in-class
# system and used unchanged in every study below.
RESIDUAL_THRESHOLD = 0.10


def _setup(true_order=0.60, horizon=1.0, n_steps=200, n_trajectories=16, seed=0,
           field=None, solver=None):
    """Shared configuration: operators, reference trajectories, budget."""
    rng = np.random.default_rng(seed)
    field = field or NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    operators = {b: l1_operator(b, n_steps, step) for b in ORDERS}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
    x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = (solver or solve_caputo)(field, x0, true_order, step, n_steps)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    flat = reference.reshape(-1, field.dim)
    budget = 1.2 * empirical_lipschitz(flat, field(flat), seed=seed)
    return dict(rng=rng, field=field, step=step, t=t, operators=operators,
                start=start, stop=stop, x0=x0, reference=reference,
                scale=scale, budget=budget, n_trajectories=n_trajectories,
                n_steps=n_steps, horizon=horizon)


# --------------------------------------------------------------------------- #
# 1. Reporting behavior of the estimator
# --------------------------------------------------------------------------- #
def reporting_study(true_order=0.60, n_replications=40,
                    noise_levels=(0.0, 0.01, 0.03, 0.05, 0.08), seed=CONFIGURATION_SEED,
                    output_dir="results"):
    """How often the procedure answers, and how accurate it is when it does."""
    cfg = _setup(true_order=true_order, seed=seed)
    rng, reference, scale = cfg["rng"], cfg["reference"], cfg["scale"]
    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        settings = [("neighbors, width rule", "neighbors", None),
                    ("consensus, width rule", "consensus", None),
                    ("consensus, residual rule", "consensus", RESIDUAL_THRESHOLD)]
        record = {name: dict(err=[], ab=[], cov=[], w=[], res=[])
                  for name, _, _ in settings}
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            for name, variant, residual_threshold in settings:
                r = estimate_order(observations, cfg["t"], ORDERS, cfg["operators"],
                                   cfg["budget"], sigma, cfg["start"], cfg["stop"],
                                   variant=variant,
                                   abstain_residual=residual_threshold)
                record[name]["err"].append(abs(r["argmin"] - true_order))
                record[name]["ab"].append(r["abstained"])
                record[name]["cov"].append(covers(r, true_order))
                record[name]["w"].append(r["width"])
                record[name]["res"].append(r["residual"])
        for name, _, _ in settings:
            e = record[name]
            m = selective_risk(e["err"], e["ab"], covered=e["cov"], widths=e["w"])
            rows.append(dict(variant=name, noise=noise,
                             mean_residual=float(np.nanmean(e["res"])), **m))
    _write_csv(os.path.join(output_dir, "reporting_behavior.csv"), rows)
    _plot_reporting(rows, os.path.join(output_dir, "figure_reporting"))
    return rows


def _plot_reporting(rows, path):
    names = []
    for r in rows:
        if r["variant"] not in names:
            names.append(r["variant"])
    handles = []
    noises = sorted({r["noise"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.3))
    for i, name in enumerate(names):
        rs = [next(r for r in rows if r["variant"] == name and r["noise"] == n)
              for n in noises]
        ax[0].plot(noises, [r["false_confidence_rate"] for r in rs], "-o", ms=4,
                   color=color(i), label=name)
        ax[1].plot(noises, [r["reporting_rate"] for r in rs], "-o", ms=4,
                   color=color(i), label=name)
        ax[2].plot([r["reporting_rate"] for r in rs],
                   [r["false_confidence_rate"] for r in rs], "-o", ms=4,
                   color=color(i), label=name)
    ax[0].set_xlabel("relative observation noise")
    ax[0].set_ylabel("false-confidence rate")
    ax[0].set_title("(A) Reporting an estimate\noutside the tolerance")
    ax[1].set_xlabel("relative observation noise")
    ax[1].set_ylabel("reporting rate")
    ax[1].set_title("(B) How often an estimate\nis reported")
    ax[2].set_xlabel("reporting rate")
    ax[2].set_ylabel("false-confidence rate")
    ax[2].set_title("(C) Risk against coverage")
    for a in ax:
        a.legend(fontsize=6)
    fig.tight_layout()
    save(fig, path)


def threshold_study(true_order=0.60, n_replications=16,
                    noise_levels=(0.0, 0.01, 0.03, 0.05, 0.08),
                    thresholds=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50), seed=0,
                    output_dir="results"):
    """Operating points of the abstention rule as its threshold is varied.

    Each run is evaluated once and then scored under every threshold, so the
    resulting curve traces the attainable pairs of reporting rate and risk on
    one set of data.
    """
    cfg = _setup(true_order=true_order, seed=seed)
    rng, reference, scale = cfg["rng"], cfg["reference"], cfg["scale"]
    errors, residuals, widths = [], [], []
    for noise in noise_levels:
        sigma = noise * scale
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            r = estimate_order(observations, cfg["t"], ORDERS, cfg["operators"],
                               cfg["budget"], sigma, cfg["start"], cfg["stop"],
                               variant="consensus")
            errors.append(abs(r["argmin"] - true_order))
            residuals.append(r["residual"])
            widths.append(r["width"])
    errors = np.asarray(errors)
    residuals = np.asarray(residuals)
    widths = np.asarray(widths)
    rows = []
    for threshold in thresholds:
        reported = residuals <= threshold
        rows.append(_operating_point(f"residual <= {threshold:g}", errors, reported))
    rows.append(_operating_point("width <= 0.30", errors, widths <= 0.30))
    rows.append(_operating_point("always report", errors,
                                 np.ones(len(errors), dtype=bool)))
    _write_csv(os.path.join(output_dir, "abstention_thresholds.csv"), rows)
    _plot_thresholds(rows, os.path.join(output_dir, "figure_abstention"))
    return rows


def _operating_point(name, errors, reported):
    n = len(errors)
    wrong = (errors > 0.05 + 1e-9) & reported
    conditional = errors[reported]
    return dict(rule=name,
                reporting_rate=float(reported.mean()),
                mae_given_reported=float(conditional.mean()) if reported.any()
                else float("nan"),
                false_confidence_rate=float(wrong.sum()) / n,
                missed_opportunity=float(((errors <= 0.05) & ~reported).sum()) / n)


def _plot_thresholds(rows, path):
    residual_rows = [r for r in rows if r["rule"].startswith("residual")]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3))
    ax[0].plot([r["reporting_rate"] for r in residual_rows],
               [r["false_confidence_rate"] for r in residual_rows], "-o", ms=5,
               color=color(0), label="feasibility residual")
    for r in rows:
        if r["rule"].startswith("width"):
            ax[0].scatter([r["reporting_rate"]], [r["false_confidence_rate"]],
                          marker="s", s=55, color=color(1),
                          label="identification-set width", zorder=3)
        if r["rule"].startswith("always"):
            ax[0].scatter([r["reporting_rate"]], [r["false_confidence_rate"]],
                          marker="^", s=55, color=color(2), label="no abstention",
                          zorder=3)
    ax[0].set_xlabel("reporting rate")
    ax[0].set_ylabel("false-confidence rate")
    ax[0].set_title("(A) Risk against coverage")
    ax[0].legend(fontsize=6.5)

    ax[1].plot([r["reporting_rate"] for r in residual_rows],
               [r["mae_given_reported"] for r in residual_rows], "-o", ms=5,
               color=color(0))
    ax[1].set_xlabel("reporting rate")
    ax[1].set_ylabel("order MAE given an estimate is reported")
    ax[1].set_title("(B) Accuracy of the reported estimates")
    fig.tight_layout()
    save(fig, path)


# --------------------------------------------------------------------------- #
# 2. Regularity budget, with one estimator throughout
# --------------------------------------------------------------------------- #
def budget_study(true_order=0.60, n_replications=30, noise_levels=(0.01, 0.03),
                 multipliers=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0), seed=0,
                 output_dir="results"):
    """Budget sweep and data-driven selection, using the consensus estimator.

    The same estimator, replication count and trajectories are used here as in
    the reporting study, so the two tables are directly comparable.
    """
    cfg = _setup(true_order=true_order, seed=seed)
    rng, reference, scale = cfg["rng"], cfg["reference"], cfg["scale"]
    reference_budget = cfg["budget"]
    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        acc = {m: dict(err=[], ab=[], cov=[], w=[]) for m in multipliers}
        acc["data"] = dict(err=[], ab=[], cov=[], w=[], sel=[])
        for _ in range(n_replications):
            observations = reference + rng.normal(0.0, sigma, reference.shape)
            for m in multipliers:
                r = estimate_order(observations, cfg["t"], ORDERS, cfg["operators"],
                                   m * reference_budget, sigma, cfg["start"],
                                   cfg["stop"], variant="consensus")
                acc[m]["err"].append(abs(r["argmin"] - true_order))
                acc[m]["ab"].append(r["abstained"])
                acc[m]["cov"].append(covers(r, true_order))
                acc[m]["w"].append(r["width"])
            selected = _select_budget(observations, cfg, sigma, reference_budget)
            r = estimate_order(observations, cfg["t"], ORDERS, cfg["operators"],
                               selected, sigma, cfg["start"], cfg["stop"],
                               variant="consensus")
            acc["data"]["err"].append(abs(r["argmin"] - true_order))
            acc["data"]["ab"].append(r["abstained"])
            acc["data"]["cov"].append(covers(r, true_order))
            acc["data"]["w"].append(r["width"])
            acc["data"]["sel"].append(selected / reference_budget)
        for key in list(multipliers) + ["data"]:
            e = acc[key]
            m = selective_risk(e["err"], e["ab"], covered=e["cov"], widths=e["w"])
            label = "data-driven" if key == "data" else f"{key}x"
            row = dict(budget=label, noise=noise, **m)
            if key == "data":
                lo, hi = bootstrap_interval(e["sel"])
                row["selected_ratio"] = float(np.mean(e["sel"]))
                row["selected_ci_low"], row["selected_ci_high"] = lo, hi
            rows.append(row)
    _write_csv(os.path.join(output_dir, "budget_study.csv"), rows)
    return rows


def _select_budget(observations, cfg, sigma, reference_budget, n_grid=25,
                   tolerance=0.05):
    """Smallest budget at which the near-overlap profile attains a small minimum."""
    smoothed = [smooth_on_grid(cfg["t"], observations[:, m, :], cfg["t"], sigma)
                for m in range(cfg["n_trajectories"])]
    labels, states, traj_id = candidate_labels(smoothed, cfg["operators"], ORDERS,
                                               cfg["start"], cfg["stop"])
    pairs = closest_cross_trajectory_pairs(states, traj_id)
    for budget in np.geomspace(0.05 * reference_budget, 8 * reference_budget, n_grid):
        p = violation_profile(ORDERS, labels, states, pairs, budget, "quantile")
        if p is not None and p.min() <= tolerance * (np.median(p) + 1e-300):
            return float(budget)
    return float(8 * reference_budget)


# --------------------------------------------------------------------------- #
# 3. Generators outside the assumed model class
# --------------------------------------------------------------------------- #
def misspecification_study(n_replications=30, noise_levels=(0.0, 0.01, 0.03),
                           seed=5, output_dir="results"):
    """What the procedure reports when no single autonomous order exists.

    Three generators are compared against the matched in-class case: an order
    that drifts slowly in time, an order that drifts more strongly, and a field
    with explicit time-dependent forcing. In each case there is no true order to
    recover, so the quantity of interest is whether the procedure declines to
    report rather than returning a confident value.
    """
    horizon, n_steps, n_trajectories = 1.0, 200, 16
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    operators = {b: l1_operator(b, n_steps, step) for b in ORDERS}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    configurations = [
        ("in-class, order 0.60", "matched", None, 0.60),
        ("order drift 0.55 to 0.65", "variable", (0.55, 0.65), 0.60),
        ("order drift 0.45 to 0.75", "variable", (0.45, 0.75), 0.60),
        ("time-dependent forcing", "forced", 0.25, 0.60),
    ]
    rows, profiles = [], {}
    for label, kind, parameter, nominal in configurations:
        rng = np.random.default_rng(seed + len(label))
        if kind == "forced":
            field = ForcedVectorField(seed=1, amplitude=parameter)
            x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
            reference = solve_caputo_forced(field, x0, nominal, step, n_steps)
            probe = NeuralVectorField(seed=1)
        elif kind == "variable":
            lo, hi = parameter
            field = NeuralVectorField(seed=1)
            x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
            reference = solve_caputo_variable_order(
                field, x0, lambda s: lo + (hi - lo) * s / horizon, step, n_steps)
            probe = field
        else:
            field = NeuralVectorField(seed=1)
            x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
            reference = solve_caputo(field, x0, nominal, step, n_steps)
            probe = field

        scale = float(np.sqrt(np.mean(reference ** 2)))
        flat = reference.reshape(-1, probe.dim)
        budget = 1.2 * empirical_lipschitz(flat, probe(flat), seed=seed)

        for noise in noise_levels:
            sigma = noise * scale
            err, ab, cov, wid, mins, res = [], [], [], [], [], []
            saved = None
            for _ in range(n_replications):
                observations = reference + (
                    rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
                r = estimate_order(observations, t, ORDERS, operators, budget,
                                   sigma, start, stop, variant="consensus",
                                   abstain_residual=RESIDUAL_THRESHOLD)
                err.append(abs(r["argmin"] - nominal))
                ab.append(r["abstained"])
                cov.append(covers(r, nominal))
                wid.append(r["width"])
                mins.append(r["argmin"])
                res.append(r["residual"])
                if saved is None:
                    saved = r["profile"]
            metrics = selective_risk(err, ab, covered=cov, widths=wid)
            lo, hi = bootstrap_interval(wid)
            rows.append(dict(system=label, noise=noise, **metrics,
                             mean_residual=float(np.mean(res)),
                             width_ci_low=lo, width_ci_high=hi,
                             argmin_mean=float(np.mean(mins)),
                             argmin_sd=float(np.std(mins, ddof=1)) if len(mins) > 1 else 0.0))
            if noise == noise_levels[min(1, len(noise_levels) - 1)]:
                profiles[label] = saved.tolist()
    _write_csv(os.path.join(output_dir, "misspecification.csv"), rows)
    _plot_misspecification(rows, profiles,
                           os.path.join(output_dir, "figure_misspecification"))
    return rows


def _plot_misspecification(rows, profiles, path):
    labels = list(profiles)
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for i, lab in enumerate(labels):
        ax[0].plot(ORDERS, profiles[lab], "-", lw=1.4, color=color(i), label=lab)
    ax[0].axhline(0.15, ls=":", c="0.45", lw=1)
    ax[0].set_xlabel(r"candidate order $\beta$")
    ax[0].set_ylabel("aggregate profile")
    ax[0].set_title("(A) The profile flattens when no\nsingle order explains the data")


    handles = []
    noises = sorted({r["noise"] for r in rows})
    x = np.arange(len(noises))
    w = 0.8 / len(labels)
    off = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * w
    for i, lab in enumerate(labels):
        rs = [next(r for r in rows if r["system"] == lab and r["noise"] == n)
              for n in noises]
        handles.append(ax[1].bar(x + off[i], [r["mean_residual"] for r in rs], w,
                                 color=color(i), label=lab))
        ax[2].bar(x + off[i], [r["abstention_rate"] for r in rs], w, color=color(i))
    ax[1].axhline(RESIDUAL_THRESHOLD, ls="--", c="0.4", lw=1)
    for a, title, ylabel in ((ax[1], "(B) The feasibility residual rises\nwhen autonomy is violated",
                              "mean feasibility residual"),
                             (ax[2], "(C) Abstention follows\nthe residual", "abstention rate")):
        a.set_xticks(x)
        a.set_xticklabels([f"{v:g}" for v in noises])
        a.set_xlabel("relative observation noise")
        a.set_ylabel(ylabel)
        a.set_title(title)
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    save(fig, path)


# --------------------------------------------------------------------------- #
# 4. Precision of the effective-dimension estimate
# --------------------------------------------------------------------------- #
CONFIGURATIONS = [
    ("d=2, non-dissipative", 2, [0.0, 0.0]),
    ("d=4, non-dissipative", 4, [0.0] * 4),
    ("d=4, slow manifold", 4, [0.2, 0.2, 4.0, 4.0]),
    ("d=4, strong slow manifold", 4, [0.2, 0.2, 8.0, 8.0]),
    ("d=6, slow manifold", 6, [0.2, 0.2, 6.0, 6.0, 6.0, 6.0]),
    ("d=6, strong slow manifold", 6, [0.2, 0.2, 12.0, 12.0, 12.0, 12.0]),
]


def dimension_precision_study(true_order=0.60, horizon=1.0, n_steps=150,
                              scan_counts=(8, 16, 32, 64), n_seeds=6,
                              sample_counts=(100, 150, 225), seed=21,
                              output_dir="results"):
    """Uncertainty of the effective dimension, and its sensitivity to the design.

    The scaling is re-estimated over independent draws of the initial states, so
    the spread of the slope reflects the sampling variability of the estimate
    rather than one realization. The dependence on the number of samples per
    trajectory is measured separately.
    """
    step = horizon / n_steps
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
    rows, samples_rows = [], []
    for label, dim, damping in CONFIGURATIONS:
        field = DissipativeVectorField(dim, damping, seed=1)
        estimates, slopes = [], []
        for s in range(n_seeds):
            distances = {}
            for count in scan_counts:
                rng = np.random.default_rng(100 * s + count)
                x0 = rng.normal(0.0, 0.45, (count, dim))
                trajectory = solve_caputo(field, x0, true_order, step, n_steps)
                distances[count] = relative_cross_trajectory_distance(
                    trajectory, start, stop, seed=s)
            scan = effective_dimension(distances)
            slopes.append(scan["slope"])
            estimates.append(scan["dimension"])
        finite = [v for v in estimates if v == v]
        lo, hi = bootstrap_interval(finite) if len(finite) > 1 else (float("nan"),) * 2
        slope_lo, slope_hi = bootstrap_interval(slopes)
        rows.append(dict(system=label, state_dimension=dim,
                         slope=float(np.mean(slopes)),
                         slope_ci_low=slope_lo, slope_ci_high=slope_hi,
                         effective_dimension=float(np.mean(finite)) if finite
                         else float("nan"),
                         sd=float(np.std(finite, ddof=1)) if len(finite) > 1
                         else float("nan"),
                         ci_low=lo, ci_high=hi,
                         n_reliable=len(finite), n_seeds=n_seeds))
        for n_samples in sample_counts:
            st = horizon / n_samples
            lo_i, hi_i = int(0.20 * n_samples), int(0.95 * n_samples)
            distances = {}
            for count in scan_counts:
                rng = np.random.default_rng(500 + count)
                x0 = rng.normal(0.0, 0.45, (count, dim))
                trajectory = solve_caputo(field, x0, true_order, st, n_samples)
                distances[count] = relative_cross_trajectory_distance(
                    trajectory, lo_i, hi_i, seed=0)
            samples_rows.append(dict(system=label, n_samples=n_samples,
                                     effective_dimension=effective_dimension(
                                         distances)["dimension"]))
    _write_csv(os.path.join(output_dir, "dimension_precision.csv"), rows)
    _write_csv(os.path.join(output_dir, "dimension_sample_sensitivity.csv"),
               samples_rows)
    return rows, samples_rows


# --------------------------------------------------------------------------- #
# 5. Comparison against an optimization-based estimator
# --------------------------------------------------------------------------- #
def baseline_study(true_order=0.60, n_replications=30,
                   noise_levels=(0.0, 0.01, 0.03), seed=0, output_dir="results"):
    """The proposed estimator against a direct search over a parametrized model.

    The comparator fits a linear fractional model at each candidate order and
    selects the order with the smallest trajectory reconstruction error, which
    is the route taken by classical fractional system identification.
    """
    systems = [("neural field (nonlinear)", None),
               ("linear system", "linear")]
    rows = []
    for system_label, kind in systems:
        rows.extend(_baseline_on(system_label, kind, true_order, n_replications,
                                 noise_levels, seed))
    _write_csv(os.path.join(output_dir, "baseline_comparison.csv"), rows)
    return rows


def _baseline_on(system_label, kind, true_order, n_replications, noise_levels,
                 seed):
    """Run the comparison on one system.

    The comparator fits a linear fractional model, so it is correctly specified
    on the linear system and misspecified on the neural one. Reporting both
    separates the merit of the optimization route from the cost of assuming a
    model form.
    """
    if kind == "linear":
        matrix = linear_system_matrix()

        class _Linear:
            dim = 2

            def __call__(self, x):
                return x @ matrix.T

        cfg = _setup(true_order=true_order, seed=seed, field=_Linear())
    else:
        cfg = _setup(true_order=true_order, seed=seed)
    rng, reference, scale = cfg["rng"], cfg["reference"], cfg["scale"]
    integrals = {b: fractional_integral_operator(b, cfg["n_steps"], cfg["step"])
                 for b in ORDERS}
    rows = []
    for noise in noise_levels:
        sigma = noise * scale
        proposed, classical, early = [], [], []
        ab, cov = [], []
        for _ in range(n_replications):
            observations = reference + (
                rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
            smoothed = [smooth_on_grid(cfg["t"], observations[:, m, :], cfg["t"], sigma)
                        for m in range(cfg["n_trajectories"])]
            r = estimate_order(observations, cfg["t"], ORDERS, cfg["operators"],
                               cfg["budget"], sigma, cfg["start"], cfg["stop"],
                               variant="consensus")
            proposed.append(abs(r["argmin"] - true_order))
            ab.append(r["abstained"]); cov.append(covers(r, true_order))
            order_hat, _ = grid_search_order(ORDERS, observations, cfg["t"],
                                             integrals, smoothed)
            classical.append(abs(order_hat - true_order))
            early.append(abs(early_time_slope(cfg["t"], observations[:, 0, :],
                                              cfg["x0"][0]) - true_order))
        row = dict(system=system_label, noise=noise)
        for name, values in (("proposed", proposed), ("classical", classical),
                             ("early_time", early)):
            lo, hi = bootstrap_interval(values)
            row[f"{name}_mae"] = float(np.mean(values))
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
        paired = np.asarray(classical) - np.asarray(proposed)
        lo, hi = bootstrap_interval(paired)
        row["difference_mae"] = float(paired.mean())
        row["difference_ci_low"], row["difference_ci_high"] = lo, hi
        row["proposed_abstention"] = float(np.mean(ab))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 6. Measured cost with repetitions
# --------------------------------------------------------------------------- #
def timing_study(horizon=1.0, n_repeats=5, seed=9, output_dir="results"):
    """Per-stage cost, repeated, with the aggregation overhead separated."""
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
        for n_orders in (9, 17, 25, 41):
            grid = np.round(np.linspace(0.30, 0.90, n_orders), 4)
            ops = {b: l1_operator(b, n_steps, step) for b in grid}
            start, stop = int(0.20 * n_steps), int(0.95 * n_steps)
            single, consensus = [], []
            for _ in range(n_repeats):
                clock = time.perf_counter()
                smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                            for m in range(n_traj)]
                labels, states, traj_id = candidate_labels(smoothed, ops, grid,
                                                           start, stop)
                times = np.concatenate([t[start:stop]] * n_traj)
                pk = nearest_neighbor_pairs(states, traj_id, times)
                violation_profile(grid, labels, states, pk, 5.0, "max")
                single.append(time.perf_counter() - clock)

                clock = time.perf_counter()
                for strength in (0.7, 1.0, 1.4):
                    sm = [smooth_on_grid(t, observations[:, m, :], t, sigma, strength)
                          for m in range(n_traj)]
                    lb, st_, tid = candidate_labels(sm, ops, grid, start, stop)
                    tm = np.concatenate([t[start:stop]] * n_traj)
                    violation_profile(grid, lb, st_,
                                      closest_cross_trajectory_pairs(st_, tid),
                                      5.0, "quantile")
                    violation_profile(grid, lb, st_,
                                      nearest_neighbor_pairs(st_, tid, tm), 5.0, "max")
                consensus.append(time.perf_counter() - clock)
            rows.append(dict(n_observations=n_traj * (n_steps + 1), n_orders=n_orders,
                             single_profile_s=float(np.mean(single)),
                             single_sd=float(np.std(single, ddof=1)),
                             consensus_s=float(np.mean(consensus)),
                             consensus_sd=float(np.std(consensus, ddof=1)),
                             overhead_factor=float(np.mean(consensus) / np.mean(single))))
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }
    _write_csv(os.path.join(output_dir, "timing.csv"), rows)
    with open(os.path.join(output_dir, "environment.json"), "w") as handle:
        json.dump(environment, handle, indent=1)
    return rows, environment


# --------------------------------------------------------------------------- #
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
    parser.add_argument("--only", default=None,
                        help="run one study: reporting, budget, misspecification, "
                             "dimension, baseline, timing")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    q = args.quick
    started = time.time()

    def stamp(name):
        print(f"    {name} done at {time.time() - started:.0f}s", flush=True)

    todo = args.only.split(",") if args.only else [
        "reporting", "threshold", "budget", "misspecification", "dimension",
        "baseline", "timing"]

    if "reporting" in todo:
        print("[1] reporting behavior", flush=True)
        reporting_study(n_replications=4 if q else 40,
                        noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03, 0.05, 0.08),
                        output_dir=args.output_dir)
        stamp("reporting")
    if "threshold" in todo:
        print("[1b] abstention operating points", flush=True)
        threshold_study(n_replications=3 if q else 16,
                        noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03, 0.05, 0.08),
                        output_dir=args.output_dir)
        stamp("threshold")
    if "budget" in todo:
        print("[2] regularity budget", flush=True)
        budget_study(n_replications=3 if q else 30,
                     noise_levels=(0.03,) if q else (0.01, 0.03),
                     output_dir=args.output_dir)
        stamp("budget")
    if "misspecification" in todo:
        print("[3] outside the model class", flush=True)
        misspecification_study(n_replications=3 if q else 30,
                               noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03),
                               output_dir=args.output_dir)
        stamp("misspecification")
    if "dimension" in todo:
        print("[4] effective-dimension precision", flush=True)
        dimension_precision_study(n_seeds=2 if q else 6,
                                  sample_counts=(100,) if q else (100, 150, 225),
                                  output_dir=args.output_dir)
        stamp("dimension")
    if "baseline" in todo:
        print("[5] comparison against an optimization-based estimator", flush=True)
        baseline_study(n_replications=3 if q else 30,
                       noise_levels=(0.0, 0.03) if q else (0.0, 0.01, 0.03),
                       output_dir=args.output_dir)
        stamp("baseline")
    if "timing" in todo:
        print("[6] measured cost", flush=True)
        timing_study(n_repeats=2 if q else 5, output_dir=args.output_dir)
        stamp("timing")

    print(f"\nResults written to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
