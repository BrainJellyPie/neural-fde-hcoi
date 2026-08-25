"""Render the second-round result files as LaTeX tables for the manuscript."""

from __future__ import annotations

import csv
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "results2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results2/tables"
os.makedirs(OUT, exist_ok=True)


def rows(name):
    with open(os.path.join(R, name)) as h:
        return list(csv.DictReader(h))


def f(r, k, d=3):
    try:
        v = float(r[k])
    except (KeyError, ValueError, TypeError):
        return "--"
    if v != v:
        return "--"
    return f"${v:.{d}f}$"


def ci(r, k, lo, hi, d=3):
    try:
        m = float(r[k])
    except (KeyError, ValueError, TypeError):
        return "--"
    if m != m:
        return "--"
    try:
        a, b = float(r[lo]), float(r[hi])
        if a == a and b == b:
            return f"${m:.{d}f}$ $[{a:.{d}f},{b:.{d}f}]$"
    except (KeyError, ValueError, TypeError):
        pass
    return f"${m:.{d}f}$"


def write(name, body):
    with open(os.path.join(OUT, name), "w") as h:
        h.write(body)
    print("wrote", name)


# --------------------------------------------------------------- reporting
lines = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Reporting Behavior of the Estimator Under Two Abstention Rules "
         r"($\alpha^\star=0.60$, Sixteen Random Histories, $N=200$, Forty Replications). "
         r"The Reporting Rate Is the Fraction of Runs That Return a Point Estimate, the "
         r"Mean Absolute Error Is Conditional on Reporting, and the False-Confidence Rate "
         r"Is the Fraction of All Runs That Report an Estimate Outside the Tolerance "
         r"$0.05$. Brackets Give Percentile Bootstrap Intervals.}",
         r"\label{tab:reporting}", r"\begin{tabular}{llccccc}", r"\toprule",
         r"Configuration & Noise & Reporting & MAE given reported & False confidence "
         r"& Set coverage & Feasibility residual \\", r"\midrule"]
prev = None
for r in rows("reporting_behavior.csv"):
    name = r["variant"]
    if prev is not None and name != prev:
        lines.append(r"\midrule")
    lines.append(" & ".join([
        name if name != prev else "", f(r, "noise", 2),
        f(r, "reporting_rate", 2),
        ci(r, "mae_given_reported", "mae_ci_low", "mae_ci_high"),
        f(r, "false_confidence_rate", 3), f(r, "set_coverage", 2),
        f(r, "mean_residual", 3)]) + r" \\")
    prev = name
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
write("table_reporting.tex", "\n".join(lines))

# ------------------------------------------------------------- thresholds
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Operating Points of the Two Abstention Rules, Pooled Over Relative "
         r"Noise $0$ to $0.08$. The Missed-Opportunity Rate Is the Fraction of Runs That "
         r"Abstain Although the Estimate Would Have Been Within Tolerance.}",
         r"\label{tab:abstention}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Rule & Report & MAE$\mid$rep. & False conf. & Missed \\",
         r"\midrule"]
for r in rows("abstention_thresholds.csv"):
    lines.append(" & ".join([
        r["rule"].replace("<=", r"$\le$"), f(r, "reporting_rate", 2),
        f(r, "mae_given_reported"), f(r, "false_confidence_rate", 3),
        f(r, "missed_opportunity", 2)]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
write("table_abstention.tex", "\n".join(lines))

# ------------------------------------------------------------------ budget
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Regularity Budget, Evaluated With the Same Estimator, Trajectories "
         r"and Replication Count as Table~\ref{tab:reporting}. The Sweep Extends to "
         r"$8L_{\mathrm{ref}}$ so That It Contains the Data-Driven Value.}",
         r"\label{tab:budget}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Budget & Noise & MAE$\mid$rep. & Width & Abst. \\", r"\midrule"]
prev = None
for r in rows("budget_study.csv"):
    if prev is not None and r["noise"] != prev:
        lines.append(r"\midrule")
    label = r["budget"]
    if label == "data-driven":
        label = f"data ($\\widehat L={float(r['selected_ratio']):.1f}L_{{\\mathrm{{ref}}}}$)"
    lines.append(" & ".join([
        label, f(r, "noise", 2),
        ci(r, "mae_given_reported", "mae_ci_low", "mae_ci_high"),
        f(r, "mean_width"), f(r, "abstention_rate", 2)]) + r" \\")
    prev = r["noise"]
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
write("table_budget.tex", "\n".join(lines))

# --------------------------------------------------------- misspecification
lines = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Behavior on Generators Outside the Single-Order Autonomous Class "
         r"(Thirty Replications). The Order Drift Cases Vary the Order Linearly Over the "
         r"Observation Window and the Forcing Case Adds an Explicit Time Dependence, so "
         r"in Neither Case Does a True Single Order Exist; the Reported Argmin Is Given "
         r"Relative to the Nominal Value $0.60$.}",
         r"\label{tab:misspec}", r"\begin{tabular}{llccccc}", r"\toprule",
         r"Generator & Noise & Recovered order & Feasibility residual & Abstention "
         r"& False confidence & Set width \\", r"\midrule"]
prev = None
for r in rows("misspecification.csv"):
    name = r["system"]
    if prev is not None and name != prev:
        lines.append(r"\midrule")
    lines.append(" & ".join([
        name if name != prev else "", f(r, "noise", 2),
        f"${float(r['argmin_mean']):.3f}\\pm{float(r['argmin_sd']):.3f}$",
        f(r, "mean_residual"), f(r, "abstention_rate", 2),
        f(r, "false_confidence_rate", 2), f(r, "mean_width")]) + r" \\")
    prev = name
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
write("table_misspecification.tex", "\n".join(lines))

# --------------------------------------------------------------- dimension
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Effective Dimension of the Visited Region With Its Sampling "
         r"Uncertainty, Re-Estimated Over Six Independent Draws of the Initial States. "
         r"Brackets Give Percentile Bootstrap Intervals for the Mean.}",
         r"\label{tab:effdim_ci}", r"\begin{tabular}{lccc}", r"\toprule",
         r"System & $d$ & $D_{\mathrm{eff}}$ & SD \\", r"\midrule"]
for r in rows("dimension_precision.csv"):
    lines.append(" & ".join([
        r["system"].replace("d=", "$d=$"), f"${r['state_dimension']}$",
        ci(r, "effective_dimension", "ci_low", "ci_high", 2), f(r, "sd", 2)]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
write("table_dimension_ci.tex", "\n".join(lines))

# ---------------------------------------------------------------- baseline
lines = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Comparison Against an Optimization-Based Fractional-Order Estimator "
         r"That Fits a Parametrized Linear Model at Each Candidate Order (Thirty "
         r"Replications). The Difference Column Is the Paired Difference, Positive When "
         r"the Proposed Estimator Is More Accurate.}",
         r"\label{tab:baseline}", r"\begin{tabular}{llcccc}", r"\toprule",
         r"System & Noise & Proposed & Optimization-based & Early-time slope "
         r"& Paired difference \\", r"\midrule"]
prev = None
for r in rows("baseline_comparison.csv"):
    name = r["system"]
    if prev is not None and name != prev:
        lines.append(r"\midrule")
    lines.append(" & ".join([
        name if name != prev else "", f(r, "noise", 2),
        ci(r, "proposed_mae", "proposed_ci_low", "proposed_ci_high"),
        ci(r, "classical_mae", "classical_ci_low", "classical_ci_high"),
        ci(r, "early_time_mae", "early_time_ci_low", "early_time_ci_high"),
        ci(r, "difference_mae", "difference_ci_low", "difference_ci_high")]) + r" \\")
    prev = name
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
write("table_baseline.tex", "\n".join(lines))

# ------------------------------------------------------------------ timing
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Measured Cost of One Profile and of the Six-Profile Aggregate, "
         r"Averaged Over Five Repetitions on a Single Processor Core. The Overhead "
         r"Factor Is the Ratio of the Two.}",
         r"\label{tab:timing}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{ccccc}", r"\toprule",
         r"Obs. & Orders & Single (s) & Aggregate (s) & Overhead \\",
         r"\midrule"]
for r in rows("timing.csv"):
    lines.append(" & ".join([
        f"${r['n_observations']}$", f"${r['n_orders']}$",
        f"${float(r['single_profile_s']):.3f}\\pm{float(r['single_sd']):.3f}$",
        f"${float(r['consensus_s']):.3f}\\pm{float(r['consensus_sd']):.3f}$",
        f"${float(r['overhead_factor']):.1f}\\times$"]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
write("table_timing.tex", "\n".join(lines))

print(f"\nAll tables written to {os.path.abspath(OUT)}")

# ------------------------------------------------------- paired joint fitting
if os.path.exists(os.path.join(R, "paired_joint_fitting.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Paired Comparison Against Joint Fitting on Identical "
             r"Replications (Thirty Replications). A Positive Difference Means the "
             r"Proposed Estimator Is More Accurate, and the Interval Excludes Zero "
             r"in Every Row.}",
             r"\label{tab:paired}", r"\setlength{\tabcolsep}{2pt}", r"\small",
             r"\begin{tabular}{cccc}", r"\toprule",
             r"Noise & Proposed & Joint fitting & Paired difference \\", r"\midrule"]
    for r in rows("paired_joint_fitting.csv"):
        lines.append(" & ".join([
            f(r, "noise", 2),
            ci(r, "proposed_mae", "proposed_ci_low", "proposed_ci_high", 2),
            ci(r, "joint_fitting_mae", "joint_fitting_ci_low", "joint_fitting_ci_high", 2),
            ci(r, "paired_difference", "paired_ci_low", "paired_ci_high", 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_paired.tex", "\n".join(lines))

# -------------------------------------------------------- linear uncertainty
if os.path.exists(os.path.join(R, "linear_uncertainty.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Order Recovery on the Linear System With Designed Collisions, "
             r"With Percentile Bootstrap Intervals Over Sixty Replications.}",
             r"\label{tab:linear_ci}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{ccccc}", r"\toprule",
             r"Noise & Early slope & Affine & Compatibility & Coverage \\", r"\midrule"]
    for r in rows("linear_uncertainty.csv"):
        lines.append(" & ".join([
            f(r, "noise", 4), f(r, "early_mae", 4), f(r, "affine_mae", 4),
            ci(r, "designed_mae", "designed_ci_low", "designed_ci_high", 4),
            f(r, "set_coverage", 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_linear_ci.tex", "\n".join(lines))

# ------------------------------------------------- discretization uncertainty
if os.path.exists(os.path.join(R, "discretization_uncertainty.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Reconstruction Error of Each Caputo Discretization at the "
             r"True Order, With Intervals Over Twenty Replications.}",
             r"\label{tab:discretization_ci}", r"\setlength{\tabcolsep}{2pt}", r"\small",
             r"\begin{tabular}{cccc}", r"\toprule",
             r"Noise & $L1$ & Gauss--Jacobi & $L1$, unsmoothed \\", r"\midrule"]
    for r in rows("discretization_uncertainty.csv"):
        lines.append(" & ".join([
            f(r, "noise", 2),
            ci(r, "l1_error", "l1_ci_low", "l1_ci_high", 3),
            ci(r, "gauss_jacobi_error", "gauss_jacobi_ci_low", "gauss_jacobi_ci_high", 3),
            f(r, "l1_unsmoothed_error", 4)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_discretization_ci.tex", "\n".join(lines))

# ----------------------------------------------------- boundary uncertainty
if os.path.exists(os.path.join(R, "boundary_uncertainty.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Integer-Order Boundary Test With Intervals Over Forty "
             r"Replications.}",
             r"\label{tab:boundary_ci}", r"\setlength{\tabcolsep}{2pt}", r"\small",
             r"\begin{tabular}{cccc}", r"\toprule",
             r"Noise & Compatibility & Contains $\beta=1$ & Early slope \\", r"\midrule"]
    for r in rows("boundary_uncertainty.csv"):
        lines.append(" & ".join([
            f(r, "noise", 2),
            f(r, "compatibility_mae", 4),
            ci(r, "contains_integer_order", "contains_ci_low", "contains_ci_high", 2),
            ci(r, "early_time_mae", "early_time_ci_low", "early_time_ci_high", 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_boundary_ci.tex", "\n".join(lines))

# ------------------------------------------------------------ threshold
if os.path.exists(os.path.join(R, "threshold_study.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Sensitivity to the Error Threshold of the Compatibility Score, "
             r"Expressed as a Fraction of the Worst-Case Bound $2\varepsilon_v+2L"
             r"\varepsilon_x$ Formed From the Observation Noise and the Spread of the "
             r"Candidate Labels Across Smoothing Strengths (Twenty-Five Replications).}",
             r"\label{tab:threshold}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"$\tau$ & Noise & MAE$\mid$rep. & Width & Abstention \\", r"\midrule"]
    prev = None
    for r in rows("threshold_study.csv"):
        if prev is not None and r["noise"] != prev:
            lines.append(r"\midrule")
        lines.append(" & ".join([
            "$" + r["threshold"].replace("x", r"\,\tau_{\max}") + "$",
            f(r, "noise", 2), f(r, "mae_given_reported"),
            f(r, "mean_width"), f(r, "abstention_rate", 2)]) + r" \\")
        prev = r["noise"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_threshold.tex", "\n".join(lines))

# ------------------------------------------------- designed collision (CI)
if os.path.exists(os.path.join(R, "designed_collision_uncertainty.csv")):
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Order Recovery on the Nonlinear Neural Field With Designed "
             r"Collisions Obtained by Newton Shooting, With Percentile Bootstrap "
             r"Intervals Over Thirty Replications.}",
             r"\label{tab:designed_ci}", r"\begin{tabular}{cccccc}", r"\toprule",
             r"Noise & Early slope & Affine diagnostic & Compatibility & Success "
             r"& Set coverage \\", r"\midrule"]
    for r in rows("designed_collision_uncertainty.csv"):
        lines.append(" & ".join([
            f(r, "noise", 2),
            ci(r, "early_mae", "early_ci_low", "early_ci_high", 4),
            ci(r, "affine_mae", "affine_ci_low", "affine_ci_high", 4),
            ci(r, "consensus_mae", "consensus_ci_low", "consensus_ci_high", 4),
            f(r, "consensus_success", 2), f(r, "set_coverage", 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write("table_designed_ci.tex", "\n".join(lines))

# --------------------------------------------------------- stage timing
if os.path.exists(os.path.join(R, "stage_timing.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Cost of Each Stage in Seconds, Averaged Over Five Repetitions "
             r"With Forty-One Candidate Orders. The Budget Search Is Reported "
             r"Separately Because It Is Optional.}",
             r"\label{tab:stage_timing}", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{ccccccc}", r"\toprule",
             r"Obs. & Smooth & Operators & Labels & Pairs & Profile & Budget search \\",
             r"\midrule"]
    for r in rows("stage_timing.csv"):
        if r["n_orders"] != "41":
            continue
        lines.append(" & ".join([
            f"${r['n_observations']}$", f(r, "smoothing_s"), f(r, "operators_s"),
            f(r, "labels_s"), f(r, "pairs_s"), f(r, "profile_s"),
            f(r, "budget_search_s")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_stage_timing.tex", "\n".join(lines))

# --------------------------------------------------- settings sensitivity
if os.path.exists(os.path.join(R, "settings_sensitivity.csv")):
    names = {"n_neighbors": r"neighbor count $k$",
             "history_separation": r"history separation $\Delta_{\mathrm{hist}}$",
             "n_closest_pairs": r"closest cross-trajectory pairs",
             "quantile": r"trimming quantile",
             "set_tolerance": r"identification tolerance $\zeta_{\mathrm{id}}$",
             "abstain_width": r"abstention width $w_{\max}$"}
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{One-at-a-Time Variation of the Settings Fixed in Advance, at "
             r"Relative Noise $0.03$ Over Twenty Replications. The Value Used "
             r"Throughout Is Marked With an Asterisk.}",
             r"\label{tab:settings}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Setting & Value & MAE$\mid$rep. & Abstention & Coverage \\", r"\midrule"]
    prev = None
    for r in rows("settings_sensitivity.csv"):
        if prev is not None and r["setting"] != prev:
            lines.append(r"\midrule")
        star = r"$^{*}$" if r["is_default"] == "True" else ""
        lines.append(" & ".join([
            names.get(r["setting"], r["setting"]) if r["setting"] != prev else "",
            f"${float(r['value']):g}$" + star,
            f(r, "mae_given_reported"), f(r, "abstention_rate", 2),
            f(r, "set_coverage", 2)]) + r" \\")
        prev = r["setting"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_settings.tex", "\n".join(lines))

# ------------------------------------------------ dimension range sensitivity
if os.path.exists(os.path.join(R, "dimension_range_sensitivity.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Effective Dimension Estimated Over Different Ranges of the "
             r"Trajectory Count. The Estimate Depends on the Range Used, So the "
             r"Reported Values Are Read as an Ordering Among Systems and Not as "
             r"Absolute Dimensions.}",
             r"\label{tab:dimension_range}", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"System, $M$ range & $[8,32]$ & $[16,64]$ & $[8,64]$ & Spread \\",
             r"\midrule"]
    for r in rows("dimension_range_sensitivity.csv"):
        lines.append(" & ".join([
            r["system"].replace("d=", "$d=$"), f(r, "dimension_8-32", 2),
            f(r, "dimension_16-64", 2), f(r, "dimension_8-64", 2),
            f(r, "spread", 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_dimension_range.tex", "\n".join(lines))

# ------------------------------------------------------ full parameter table
PARAMS = [
    ("Candidate orders", r"$\mathcal B$",
     r"$0.30$ to $0.90$ in steps of $0.025$; $0.20$ to $1.00$ in Section~\ref{subsec:alpha1} and $0.05$ to $0.40$ near the lower endpoint"),
    ("Analysis window", r"$[\delta,T]$", r"$0.20T$ to $0.95T$"),
    ("Neighbor count", r"$k$", r"$8$"),
    ("History separation", r"$\Delta_{\mathrm{hist}}$", r"$0.15\,T$"),
    ("Closest cross-trajectory pairs", "--", r"$30$"),
    ("Trimming quantile", "--", r"$0.9$"),
    ("Smoothing strengths", "--", r"$0.7$, $1$ and $1.4$ times the reference"),
    ("Reference smoothing penalty", "--", r"$N\sigma^2$ for a cubic smoothing spline, with $\sigma$ the noise level"),
    ("Profile normalization", r"$\varepsilon$", r"$10^{-300}$, which leaves a constant profile at zero"),
    ("Identification tolerance", r"$\zeta_{\mathrm{id}}$", r"$0.15$ on the aggregate profile"),
    ("Abstention width", r"$w_{\max}$", r"$0.30$"),
    ("Abstention residual", r"$\rho_{\max}$", r"$0.15$ when the feasibility rule is used"),
    ("Error threshold", r"$\tau$", r"$0$; the sweep against the worst-case bound is Table~\ref{tab:threshold}"),
    ("Budget grid", r"$L$", r"$25$ points geometric in $[0.05,8]\times L_{\mathrm{ref}}$"),
    ("Budget selection level", "--", r"$0.05$ times the median of the near-overlap profile"),
    ("Quadrature nodes", r"$Q$", r"$24$ Gauss--Jacobi nodes"),
    ("Random features", "--", r"$64$ tanh features, ridge $10^{-8}$"),
    ("Replications", "--", r"$30$ to $60$; stated in each caption"),
]
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Settings of the Procedure. Values Are Fixed in Advance and Used "
         r"Unchanged Unless a Table States Otherwise.}",
         r"\label{tab:parameters}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
         r"\begin{tabular}{p{0.30\columnwidth}p{0.10\columnwidth}p{0.48\columnwidth}}",
         r"\toprule", r"Setting & Symbol & Value \\", r"\midrule"]
for a, b, c in PARAMS:
    lines.append(f"{a} & {b} & {c} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
write("table_parameters.tex", "\n".join(lines))

# --------------------------------------------------- joint gradient training
if os.path.exists(os.path.join(R, "joint_training.csv")):
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{\revb{Joint Gradient Training of the Order and a Neural Vector "
             r"Field, Against the Proposed Estimator on the Same Trajectories "
             r"($\alpha^\star=0.60$, Sixteen Histories, $N=200$, Six Random "
             r"Initializations, Adam for $800$ Epochs). The Relative Root-Mean-Square "
             r"Reconstruction Error of the Trained Model Is Reported Alongside the "
             r"Recovered Order.}}",
             r"\label{tab:joint_training}", r"\setlength{\tabcolsep}{3pt}",
             r"\footnotesize",
             r"\begin{tabular}{cccccc}", r"\toprule",
             r"Noise & Order (joint) & Range & Rel.\ RMSE & MAE (joint) & Error (proposed) \\",
             r"\midrule"]
    for r in rows("joint_training.csv"):
        lines.append(" & ".join([
            f(r, "noise", 2),
            f"${float(r['joint_order_mean']):.3f}\\pm{float(r['joint_order_sd']):.3f}$",
            f"$[{float(r['joint_order_min']):.3f},{float(r['joint_order_max']):.3f}]$",
            f(r, "joint_relative_rmse", 3),
            ci(r, "joint_mae", "joint_ci_low", "joint_ci_high"),
            f(r, "proposed_error")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_joint_training.tex", "\n".join(lines))
