"""Reporting metrics for a procedure that may decline to answer.

An estimator that abstains cannot be summarized by a mean error alone, because
the mean is then conditional on the runs that reported. The quantities here
separate how often the procedure answers, how accurate it is when it answers,
how often it answers confidently but wrongly, and whether the reported set
covers the true order.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "selective_risk",
    "bootstrap_interval",
    "format_with_interval",
]

SUCCESS_TOLERANCE = 0.05 + 1e-9


def selective_risk(errors, abstained, covered=None, widths=None,
                   tolerance: float = SUCCESS_TOLERANCE, n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Summarize a set of runs in which the procedure may abstain.

    Parameters
    ----------
    errors : per-run absolute order error; entries for abstained runs are ignored
        by the conditional quantities but still counted in the reporting rate
    abstained : per-run boolean, True when no point estimate was reported
    covered : optional per-run boolean, True when the reported identification
        set contains the true order
    widths : optional per-run identification-set width

    Returns
    -------
    dict with the reporting rate, the error conditional on reporting, the
    false-confidence rate (reported but outside the tolerance), the selective
    risk, and the coverage of the identification set.
    """
    errors = np.asarray(errors, dtype=float)
    abstained = np.asarray(abstained, dtype=bool)
    n = len(abstained)
    reported = ~abstained
    n_reported = int(reported.sum())

    conditional = errors[reported]
    wrong_and_reported = conditional > tolerance if n_reported else np.array([])

    out = {
        "n_runs": n,
        "reporting_rate": n_reported / n if n else float("nan"),
        "abstention_rate": 1.0 - n_reported / n if n else float("nan"),
        "mae_given_reported": float(conditional.mean()) if n_reported else float("nan"),
        "success_given_reported": float((~wrong_and_reported).mean())
        if n_reported else float("nan"),
        # Fraction of all runs that report an estimate outside the tolerance.
        # This is the quantity an abstention rule is meant to keep small.
        "false_confidence_rate": float(wrong_and_reported.sum()) / n if n else float("nan"),
        # Error over all runs, charging an abstained run the tolerance itself,
        # so that abstaining is neither free nor as costly as a wrong answer.
        "selective_risk": float(
            (conditional.sum() + tolerance * (n - n_reported)) / n
        ) if n else float("nan"),
    }
    if n_reported:
        lo, hi = bootstrap_interval(conditional, n_boot=n_boot, seed=seed)
        out["mae_ci_low"], out["mae_ci_high"] = lo, hi
        out["mae_sd"] = float(conditional.std(ddof=1)) if n_reported > 1 else 0.0
    else:
        out["mae_ci_low"] = out["mae_ci_high"] = out["mae_sd"] = float("nan")
    if covered is not None:
        covered = np.asarray(covered, dtype=bool)
        out["set_coverage"] = float(covered.mean()) if len(covered) else float("nan")
    if widths is not None:
        widths = np.asarray(widths, dtype=float)
        finite = np.isfinite(widths)
        out["mean_width"] = float(widths[finite].mean()) if finite.any() else float("nan")
    return out


def bootstrap_interval(values, level: float = 0.95, n_boot: int = 2000,
                       seed: int = 0):
    """Percentile bootstrap interval for the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    half = (1.0 - level) / 2.0
    return (float(np.quantile(draws, half)), float(np.quantile(draws, 1.0 - half)))


def format_with_interval(mean: float, low: float, high: float,
                         digits: int = 3) -> str:
    """Render a mean with its interval for a table cell."""
    if mean != mean:
        return "--"
    if low != low or high != high:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
