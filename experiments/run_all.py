"""Run every experiment reported in the paper and write the results to disk.

Usage
-----
    python -m experiments.run_all                 # full settings
    python -m experiments.run_all --quick         # reduced settings, fast check
    python -m experiments.run_all --output-dir out
"""

from __future__ import annotations

import argparse
import os
import time

from experiments import (
    order_range_endpoints,
    random_histories,
    recovery_determinants,
    settings_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true",
                        help="reduced settings for a fast check")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()

    def elapsed() -> str:
        return f"{time.time() - started:.0f}s"

    print("[1/4] random histories with automatic pair selection")
    if args.quick:
        random_histories.run(output_dir=args.output_dir, n_steps=100,
                             n_trajectories=10, n_replications=3,
                             noise_levels=(0.0, 0.03), retention_levels=(0.70,))
    else:
        random_histories.run(output_dir=args.output_dir)
    print("      done", elapsed())

    print("[2/4] regularity budget, smoothing, and discretization")
    if args.quick:
        settings_sensitivity.run(output_dir=args.output_dir, n_steps=100,
                                 n_trajectories=10, n_replications=2,
                                 noise_levels=(0.03,))
    else:
        settings_sensitivity.run(output_dir=args.output_dir)
    print("      done", elapsed())

    print("[3/4] endpoints of the order range")
    order_range_endpoints.run(output_dir=args.output_dir, quick=args.quick)
    print("      done", elapsed())

    print("[4/4] determinants of recovery")
    if args.quick:
        recovery_determinants.joint_fitting_study(
            output_dir=args.output_dir, n_steps=80, n_trajectories=6, n_seeds=2,
            noise_levels=(0.0, 0.03))
        recovery_determinants.dimension_study(
            output_dir=args.output_dir, n_steps=100, n_trajectories=24,
            n_replications=2, noise_levels=(0.03,), scan_counts=(8, 16, 32),
            configurations=recovery_determinants.CONFIGURATIONS[:3])
    else:
        recovery_determinants.joint_fitting_study(output_dir=args.output_dir)
        recovery_determinants.dimension_study(output_dir=args.output_dir)
    print("      done", elapsed())

    print(f"\nAll results written to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
