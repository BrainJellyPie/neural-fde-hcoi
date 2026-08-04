"""Two studies of what determines whether the order can be recovered.

The first fits the order and a nonlinear vector field together by minimizing the
trajectory reconstruction error. Because the field enters linearly, the optimal
field at each candidate order is obtained in closed form, so the reported error is
the global minimum over fields at that order.

The second varies the state dimension and the dissipation of the vector field. The
dimension of the region the trajectories actually visit is estimated from the rate
at which the distance to the nearest state on another trajectory decreases as
trajectories are added, and recovery accuracy is compared against it.
"""

from __future__ import annotations

import argparse
import csv
import os

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
    estimate_order,
    joint_field_fit,
)
from hcoi.plotting import color, plt, save
from hcoi.systems import (
    DissipativeVectorField,
    NeuralVectorField,
    empirical_lipschitz,
    solve_caputo,
)

SUCCESS_TOLERANCE = 0.05 + 1e-4


# --------------------------------------------------------------------------- #
# Joint fitting of the order and a nonlinear field
# --------------------------------------------------------------------------- #
def joint_fitting_study(true_order=0.60, horizon=1.0, n_steps=120,
                        n_trajectories=8, n_features=64, n_seeds=5,
                        noise_levels=(0.0, 0.01, 0.03), seed=0,
                        output_dir="results"):
    rng = np.random.default_rng(seed)
    field = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    orders = np.round(np.arange(0.30, 0.9001, 0.025), 4)
    integrals = {b: fractional_integral_operator(b, n_steps, step) for b in orders}

    x0 = rng.normal(0.0, 0.45, (n_trajectories, field.dim))
    reference = solve_caputo(field, x0, true_order, step, n_steps)
    state_scale = float(np.sqrt(np.mean(reference ** 2)))

    rows, curves = [], {}
    for noise in noise_levels:
        sigma = noise * state_scale
        observations = reference + (
            rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
        smoothed = [smooth_on_grid(t, observations[:, m, :], t, sigma)
                    for m in range(n_trajectories)]
        targets = [x - x[0][None, :] for x in smoothed]

        error_curves, complexity_curves, estimates = [], [], []
        for s in range(n_seeds):
            errors, fields, stacked = joint_field_fit(orders, smoothed, targets,
                                                      integrals, n_features,
                                                      seed=10 + s)
            complexity = [empirical_lipschitz(stacked, f, n_probes=300, seed=7)
                          for f in fields]
            error_curves.append(errors)
            complexity_curves.append(complexity)
            estimates.append(float(orders[int(np.argmin(errors))]))
        mean_error = np.mean(error_curves, axis=0)
        mean_complexity = np.mean(complexity_curves, axis=0)
        curves[noise] = dict(orders=orders.tolist(),
                             reconstruction_error=mean_error.tolist(),
                             field_complexity=mean_complexity.tolist(),
                             estimates=estimates)

        band = orders[mean_error <= 2.0 * mean_error.min()]
        true_index = int(np.argmin(np.abs(orders - true_order)))
        rows.append({
            "noise": noise,
            "band_share": float(len(band) / len(orders)),
            "band_width": float(band.max() - band.min()),
            "complexity_ratio": float(mean_complexity.max()
                                      / (mean_complexity[true_index] + 1e-12)),
            "order_mae": float(np.mean([abs(a - true_order) for a in estimates])),
            "estimate_spread": float(np.max(estimates) - np.min(estimates)),
        })

    _write_csv(os.path.join(output_dir, "joint_fitting.csv"), rows)
    _plot_joint_fitting(curves, true_order,
                        os.path.join(output_dir, "figure_joint_fitting"))
    return rows


def _plot_joint_fitting(curves, true_order, path):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for k, (noise, c) in enumerate(sorted(curves.items())):
        orders = np.asarray(c["orders"])
        ax[0].plot(orders, c["reconstruction_error"], "-o", ms=2.5, color=color(k),
                   label=f"noise {noise:g}")
        ax[1].plot(orders, c["field_complexity"], "-o", ms=2.5, color=color(k),
                   label=f"noise {noise:g}")
        ax[2].scatter([noise] * len(c["estimates"]), c["estimates"], s=22,
                      color=color(3), zorder=3)
    for a, title, ylabel in (
        (ax[0], "(A) A wide band of orders reproduces\nthe trajectory comparably well",
         "relative trajectory error"),
        (ax[1], "(B) Wrong orders require\nhigher field complexity",
         "empirical Lipschitz constant"),
    ):
        a.axvline(true_order, ls="--", c="0.4", lw=1)
        a.set_yscale("log")
        a.set_xlabel(r"candidate order $\beta$")
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.legend(fontsize=7)
    ax[2].axhline(true_order, ls="--", c="0.4", lw=1, label="true order")
    ax[2].set_xlabel("relative observation noise")
    ax[2].set_ylabel(r"recovered order $\widehat\alpha$")
    ax[2].set_title("(C) Recovered order drifts upward\nas noise grows")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, path)


# --------------------------------------------------------------------------- #
# Effective dimension of the visited region
# --------------------------------------------------------------------------- #
CONFIGURATIONS = [
    ("d=2, non-dissipative", 2, [0.0, 0.0]),
    ("d=4, non-dissipative", 4, [0.0] * 4),
    ("d=4, slow manifold", 4, [0.2, 0.2, 4.0, 4.0]),
    ("d=4, strong slow manifold", 4, [0.2, 0.2, 8.0, 8.0]),
    ("d=6, slow manifold", 6, [0.2, 0.2, 6.0, 6.0, 6.0, 6.0]),
    ("d=6, strong slow manifold", 6, [0.2, 0.2, 12.0, 12.0, 12.0, 12.0]),
]


def dimension_study(true_order=0.60, horizon=1.0, n_steps=150, n_trajectories=64,
                    n_replications=8, noise_levels=(0.0, 0.01, 0.03),
                    scan_counts=(8, 16, 32, 64), seed=31, output_dir="results",
                    configurations=None):
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    orders = np.round(np.arange(0.30, 0.9001, 0.025), 4)
    operators = {b: l1_operator(b, n_steps, step) for b in orders}
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    rows, scans = [], {}
    for label, dim, damping in (configurations or CONFIGURATIONS):
        field = DissipativeVectorField(dim, damping, seed=1)

        distances = {}
        for count in scan_counts:
            rng = np.random.default_rng(21)
            x0 = rng.normal(0.0, 0.45, (count, dim))
            trajectory = solve_caputo(field, x0, true_order, step, n_steps)
            distances[count] = relative_cross_trajectory_distance(trajectory,
                                                                  start, stop)
        scans[label] = effective_dimension(distances)

        rng = np.random.default_rng(seed + dim)
        x0 = rng.normal(0.0, 0.45, (n_trajectories, dim))
        reference = solve_caputo(field, x0, true_order, step, n_steps)
        state_scale = float(np.sqrt(np.mean(reference ** 2)))
        displacement = float(
            np.mean(np.linalg.norm(reference[stop - 1] - reference[start], axis=1))
            / state_scale)
        flat = reference.reshape(-1, dim)
        budget = 1.2 * empirical_lipschitz(flat, field(flat), n_probes=400, seed=seed)
        relative_distance = relative_cross_trajectory_distance(reference, start, stop)

        for noise in noise_levels:
            sigma = noise * state_scale
            errors, hits, abstained, informative = [], [], 0, []
            for _ in range(n_replications):
                observations = reference + (
                    rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
                result = estimate_order(observations, t, orders, operators, budget,
                                        sigma, start, stop, variant="consensus")
                if result["abstained"]:
                    abstained += 1
                else:
                    error = round(abs(result["order"] - true_order), 4)
                    errors.append(error)
                    hits.append(error <= SUCCESS_TOLERANCE)
                ref = result["reference"]
                if ref is not None:
                    pairs = closest_cross_trajectory_pairs(ref["states"],
                                                           ref["trajectory_id"])
                    ratio = transversality_ratio(orders, ref["labels"], ref["states"],
                                                 pairs, true_order, budget, sigma)
                    informative.append(int(np.sum(ratio > 1.0)) if len(ratio) else 0)
            total = len(errors) + abstained
            rows.append({
                "configuration": label, "state_dimension": dim,
                "n_trajectories": n_trajectories, "noise": noise,
                "effective_dimension": scans[label]["dimension"],
                "relative_distance": relative_distance,
                "displacement_ratio": displacement,
                "informative_pairs": float(np.mean(informative)) if informative else 0.0,
                "order_mae": float(np.mean(errors)) if errors else float("nan"),
                "success": float(np.mean(hits)) if hits else float("nan"),
                "abstention": abstained / total if total else float("nan"),
            })

    _write_csv(os.path.join(output_dir, "effective_dimension.csv"), rows)
    _plot_dimension(scans, rows, os.path.join(output_dir,
                                              "figure_effective_dimension"))
    return rows


def _short(label):
    return (label.replace(", non-dissipative", " plain")
            .replace(", strong slow manifold", " strong SM")
            .replace(", slow manifold", " SM"))


def _plot_dimension(scans, rows, path):
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    labels = list(scans)
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 4.1))

    handles = []
    for i, label in enumerate(labels):
        s = scans[label]
        line, = ax[0].plot(s["counts"], s["distances"], "-o", ms=3.5, color=color(i),
                           label=f"{_short(label)} "
                                 rf"($D_{{\mathrm{{eff}}}}={s['dimension']:.2f}$)")
        handles.append(line)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(scans[labels[0]]["counts"])
    ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[0].xaxis.set_minor_locator(ticker.NullLocator())
    ax[0].set_xlabel("number of trajectories $M$")
    ax[0].set_ylabel("median cross-trajectory\nnearest-neighbor distance (relative)")
    ax[0].set_title("(A) Overlap availability scales with\nthe effective dimension")
    ax[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                 ncol=2, fontsize=6, frameon=False, handlelength=1.6,
                 columnspacing=1.0, borderaxespad=0.0)

    dims = [next(r["state_dimension"] for r in rows if r["configuration"] == l)
            for l in labels]
    effective = [scans[l]["dimension"] for l in labels]
    x = np.arange(len(labels))
    bar1 = ax[1].bar(x - 0.2, dims, 0.4, color="0.75", label="state dimension $d$")
    bar2 = ax[1].bar(x + 0.2, effective, 0.4, color=color(3),
                     label=r"effective dimension $D_{\mathrm{eff}}$")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([_short(l).replace(" ", "\n", 1) for l in labels],
                          fontsize=6)
    ax[1].set_ylabel("dimension")
    ax[1].set_title("(B) Dissipation lowers the\neffective dimension")
    ax[1].legend(handles=[bar1, bar2], loc="upper center",
                 bbox_to_anchor=(0.5, -0.20), ncol=1, fontsize=7, frameon=False,
                 borderaxespad=0.0)

    for i, label in enumerate(labels):
        for r in [r for r in rows if r["configuration"] == label]:
            y = 1.0 if r["order_mae"] != r["order_mae"] else max(r["order_mae"], 0.005)
            ax[2].scatter(r["relative_distance"], y,
                          marker=("o" if r["abstention"] < 1.0 else "x"),
                          s=42, color=color(i), zorder=3)
    ax[2].axhline(0.05, ls="--", c="0.4", lw=1)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("relative nearest-neighbor distance")
    ax[2].set_ylabel("order MAE")
    ax[2].set_title("(C) Accuracy tracks overlap availability,\nnot the state dimension")
    legend = [
        Line2D([], [], ls="none", marker="o", ms=6, color="0.35",
               label="order reported (colors as in (A))"),
        Line2D([], [], ls="none", marker="x", ms=7, color="0.35",
               label="all runs abstained, plotted at $1$"),
        Line2D([], [], ls="--", color="0.4", lw=1, label="success threshold $0.05$"),
    ]
    ax[2].legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                 ncol=1, fontsize=6.5, frameon=False, borderaxespad=0.0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    save(fig, path)


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
    os.makedirs(args.output_dir, exist_ok=True)

    joint_kwargs = dict(output_dir=args.output_dir)
    dimension_kwargs = dict(output_dir=args.output_dir)
    if args.quick:
        joint_kwargs.update(n_steps=80, n_trajectories=6, n_seeds=2,
                            noise_levels=(0.0, 0.03))
        dimension_kwargs.update(n_steps=100, n_trajectories=24, n_replications=2,
                                noise_levels=(0.03,), scan_counts=(8, 16, 32),
                                configurations=CONFIGURATIONS[:3])

    joint_rows = joint_fitting_study(**joint_kwargs)
    for row in joint_rows:
        print(f"joint fit, noise {row['noise']:g}: band share "
              f"{row['band_share']:.2f}, width {row['band_width']:.2f}, "
              f"complexity ratio {row['complexity_ratio']:.1f}, "
              f"MAE {row['order_mae']:.3f}")

    dimension_rows = dimension_study(**dimension_kwargs)
    seen = set()
    for row in dimension_rows:
        if row["configuration"] not in seen:
            seen.add(row["configuration"])
            print(f"{row['configuration']}: effective dimension "
                  f"{row['effective_dimension']:.2f}, relative distance "
                  f"{row['relative_distance']:.3f}")


if __name__ == "__main__":
    main()
