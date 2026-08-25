# History-Compatible Identification of the Fractional Order in Neural FDEs

Reference implementation for the paper *Identifiability and Deterministic Stable
Recovery of the Fractional Order in Neural Fractional Differential Equations*
(IEEE Access).

A neural fractional differential equation fits a single order together with a
neural vector field. Different orders can reproduce the same trajectory almost
equally well, so a small reconstruction error is not evidence for the order. This
repository implements an order estimate that is formed before any field is
trained, together with the diagnostics that indicate in advance whether a dataset
supports it.

## Installation

```bash
git clone https://github.com/---/neural-fde-hcoi
cd neural-fde-hcoi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only NumPy, SciPy, and Matplotlib are required. No GPU is used.

## Reproducing the reported results

```bash
python -m experiments.run_all                 # writes CSV files and figures to results/
python -m experiments.run_all --quick         # reduced settings, finishes in seconds
```

The full run takes a few minutes on a single processor core. Each experiment can
also be run on its own:

```bash
python -m experiments.random_histories
python -m experiments.settings_sensitivity
python -m experiments.order_range_endpoints
python -m experiments.recovery_determinants
```

The studies added for the second revision write to `results2/` and are run with

```bash
python -m experiments.reporting_and_scope --output-dir results2
python -m experiments.uncertainty          --output-dir results2
python -m experiments.completeness         --output-dir results2
python -m experiments.joint_training       --output-dir results2
python  experiments/make_tables.py results2 results2/tables
```

Each accepts `--quick` for a reduced run. The four studies take about twenty
minutes in total on a single processor core, of which the joint training takes
the larger part.

All random seeds are fixed in the experiment scripts, so repeated runs on the same
machine give identical numbers.

## What each experiment reports

| Script | Question | Output |
| --- | --- | --- |
| `random_histories` | Do informative state overlaps occur when no collision is engineered, and can they be found automatically? | `random_histories_accuracy.csv`, `random_histories_diagnostics.csv`, `random_histories_sampling.csv` |
| `settings_sensitivity` | How do the regularity budget, the smoothing strength, the sampling density, and the choice of Caputo discretization affect the estimate? | `budget_sensitivity.csv`, `smoothing_sensitivity.csv`, `discretization_comparison.csv`, `wall_clock.csv` |
| `order_range_endpoints` | Is ordinary dynamics reported as ordinary, and does the reported range still cover a true order near zero? | `integer_order_boundary.csv`, `near_zero_endpoint.csv` |
| `recovery_determinants` | Does fitting the order and the field together determine the order, and what governs recovery as the state dimension grows? | `joint_fitting.csv`, `effective_dimension.csv`, `figure_joint_fitting.pdf`, `figure_effective_dimension.pdf` |
| `reporting_and_scope` | How often does the procedure answer, how accurate is it when it does, and what happens outside the assumed model class? | `reporting_behavior.csv`, `abstention_thresholds.csv`, `misspecification.csv`, `budget_study.csv` |
| `uncertainty` | Do the estimator comparisons survive the replication spread? | `paired_joint_fitting.csv`, `linear_uncertainty.csv`, `discretization_uncertainty.csv`, `boundary_uncertainty.csv` |
| `completeness` | What do the error threshold, the fixed settings, and the fitting range do to the result? | `threshold_study.csv`, `settings_sensitivity.csv`, `stage_timing.csv`, `dimension_range_sensitivity.csv` |
| `joint_training` | Does an end-to-end trained Neural FDE recover the order? | `joint_training.csv`, `joint_training_runs.csv` |

## Library layout

```
hcoi/
  caputo.py          L1 and Gauss-Jacobi discretizations, fractional integral,
                     smoothing splines
  systems.py         neural and dissipative vector fields, linear system,
                     predictor-corrector solver
  identification.py  pair selection, compatibility profile, identification set,
                     order estimators
  diagnostics.py     transversality ratio, overlap availability, effective dimension
```

### Minimal example

```python
import numpy as np
from hcoi.caputo import l1_operator
from hcoi.identification import estimate_order
from hcoi.systems import NeuralVectorField, empirical_lipschitz, solve_caputo

field = NeuralVectorField(seed=1)
n_steps, horizon = 200, 1.0
step = horizon / n_steps
t = np.linspace(0.0, horizon, n_steps + 1)

rng = np.random.default_rng(0)
x0 = rng.normal(0.0, 0.45, (16, 2))
trajectories = solve_caputo(field, x0, 0.60, step, n_steps)

scale = float(np.sqrt(np.mean(trajectories ** 2)))
noise = 0.01 * scale
observations = trajectories + rng.normal(0.0, noise, trajectories.shape)

orders = np.round(np.arange(0.30, 0.9001, 0.025), 4)
operators = {b: l1_operator(b, n_steps, step) for b in orders}
flat = trajectories.reshape(-1, 2)
budget = 1.2 * empirical_lipschitz(flat, field(flat))

result = estimate_order(observations, t, orders, operators, budget, noise,
                        start=int(0.20 * n_steps), stop=int(0.95 * n_steps))
print(result["order"], result["lower"], result["upper"], result["abstained"])
```

## Method in brief

1. **Candidate labels.** For each candidate order, the Caputo derivative of every
   observation is reconstructed from a fitted smoother. Two discretizations are
   provided; at order one the L1 coefficients reduce to the backward difference,
   so the integer order needs no special case.
2. **Pair selection.** Observations that are close in state but separated in
   history carry order information. Two pair sets are used, the state-space
   nearest neighbors and the closest pairs lying on different trajectories.
3. **Compatibility profile.** For each candidate order, the profile measures how
   far the candidate labels are from being explainable by one autonomous field of
   bounded Lipschitz constant. The true order makes this violation vanish.
4. **Identification set and abstention.** The reported output is the set of orders
   at which the profile stays small. When that set is wide the data do not
   determine the order, and the procedure abstains instead of returning a point
   estimate.

Two settings can be chosen from the data alone. The smoothing strength is handled
by evaluating the profile at several strengths and combining them. The regularity
budget is selected as the smallest value at which the profile of the closest pairs
attains a small minimum.

## Diagnostics

Two quantities are computable before any estimate is formed and indicate whether a
dataset carries the required structure.

- **Transversality ratio.** How strongly a pair separates candidate orders relative
  to its own error level. The number of pairs with a ratio above one, and not their
  proportion, is what tracks accuracy.
- **Effective dimension.** The dimension of the region the trajectories visit,
  estimated from the rate at which the distance to the nearest state on another
  trajectory decreases as trajectories are added. Recovery tracks this quantity and
  not the dimension of the state space.

## Citation

```bibtex
@article{ryu2026identifiability,
  title   = {Identifiability and Deterministic Stable Recovery of the Fractional
             Order in Neural Fractional Differential Equations},
  author  = {Ryu, Donghun and Lee, Minhyeok},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
