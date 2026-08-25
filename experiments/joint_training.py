"""Joint gradient training of the fractional order and a neural vector field.

This is the procedure a Neural FDE is trained with in practice: the order and
the network weights are optimized together against a trajectory reconstruction
loss, by gradient descent, from several random initializations. The forward map
is the predictor-corrector solver, so the loss is not quadratic in the weights
and the order enters through the memory kernel as well as through the solution.

The experiment records what such training recovers, and how the recovered order
varies across initializations, on the same system used elsewhere in the paper.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from math import gamma

import numpy as np

from hcoi.caputo import l1_operator
from hcoi.identification import covers, estimate_order
from hcoi.reporting import bootstrap_interval
from hcoi.systems import NeuralVectorField, empirical_lipschitz, solve_caputo

ORDERS = np.round(np.arange(0.30, 0.9001, 0.025), 4)


# --------------------------------------------------------------------------- #
# A small trainable network and the differentiable solver
# --------------------------------------------------------------------------- #
class TrainableField:
    """One hidden layer with tanh activation, trained by gradient descent."""

    def __init__(self, dim: int, width: int = 32, seed: int = 0,
                 scale: float = 2.0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, scale / np.sqrt(dim), (dim, width))
        self.b1 = np.zeros(width)
        self.W2 = rng.normal(0, scale / np.sqrt(width), (width, dim))
        self.b2 = np.zeros(dim)
        self.dim = dim

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward_cache(self, x):
        pre = x @ self.W1 + self.b1
        h = np.tanh(pre)
        return h @ self.W2 + self.b2, (x, h)

    def backward(self, cache, grad_out):
        """Gradients of the parameters and of the input, given dL/doutput."""
        x, h = cache
        gW2 = h.reshape(-1, h.shape[-1]).T @ grad_out.reshape(-1, grad_out.shape[-1])
        gb2 = grad_out.reshape(-1, grad_out.shape[-1]).sum(axis=0)
        gh = grad_out @ self.W2.T
        gpre = gh * (1.0 - h ** 2)
        gW1 = x.reshape(-1, x.shape[-1]).T @ gpre.reshape(-1, gpre.shape[-1])
        gb1 = gpre.reshape(-1, gpre.shape[-1]).sum(axis=0)
        gx = gpre @ self.W1.T
        return [gW1, gb1, gW2, gb2], gx


def _corrector_weights(order: float, step: float, k: int):
    a = step ** order / (order * (order + 1.0))
    w = np.empty(k + 1)
    w[0] = a * (k ** (order + 1) - (k - order) * (k + 1) ** order)
    if k >= 1:
        j = np.arange(1, k + 1)
        w[1:] = a * ((k - j + 2) ** (order + 1) + (k - j) ** (order + 1)
                     - 2.0 * (k - j + 1) ** (order + 1))
    return w


def solve_forward(field: TrainableField, x0, order, step, n_steps):
    """Explicit fractional Adams step, retaining the caches for backpropagation."""
    n_traj, dim = x0.shape
    x = np.zeros((n_steps + 1, n_traj, dim))
    f = np.zeros((n_steps + 1, n_traj, dim))
    caches = [None] * (n_steps + 1)
    x[0] = x0
    f[0], caches[0] = field.forward_cache(x0)
    inv_gamma = 1.0 / gamma(order)
    for k in range(n_steps):
        w = _corrector_weights(order, step, k)
        a = step ** order / (order * (order + 1.0))
        x[k + 1] = x0 + inv_gamma * (np.einsum("j,jbd->bd", w, f[: k + 1])
                                     + a * f[k])
        f[k + 1], caches[k + 1] = field.forward_cache(x[k + 1])
    return x, f, caches


def loss_and_grads(field, x0, order, step, n_steps, target, delta=1e-4):
    """Reconstruction loss, parameter gradients, and a finite-difference d/dorder.

    The parameter gradients are obtained by differentiating the solver, and the
    derivative with respect to the order is taken by central differences, since
    the order enters through the Gamma function and the memory weights.
    """
    x, f, caches = solve_forward(field, x0, order, step, n_steps)
    residual = x - target
    loss = float(np.mean(residual ** 2))

    scale = 2.0 / residual.size
    inv_gamma = 1.0 / gamma(order)
    grads = [np.zeros_like(p) for p in field.parameters()]
    # Adjoint of the solve. x[k+1] depends on f[0..k], and f[j] depends on x[j],
    # so the sensitivity of the loss to f[j] accumulates the weight that j
    # carries in every later step.
    a = step ** order / (order * (order + 1.0))
    grad_x = scale * residual.copy()
    grad_f = np.zeros_like(x)
    for k in range(n_steps, 0, -1):
        w = _corrector_weights(order, step, k - 1)
        gk = grad_x[k]
        for j in range(k):
            grad_f[j] += inv_gamma * w[j] * gk
        grad_f[k - 1] += inv_gamma * a * gk
    for j in range(n_steps, -1, -1):
        if not grad_f[j].any():
            continue
        g, gx = field.backward(caches[j], grad_f[j])
        for i in range(len(grads)):
            grads[i] += g[i]
        if j >= 1:
            # x[j] also feeds the loss directly and feeds f[j]; propagate the
            # latter back into the steps that produced x[j].
            w = _corrector_weights(order, step, j - 1)
            for m in range(j):
                grad_f[m] += inv_gamma * w[m] * gx
            grad_f[j - 1] += inv_gamma * a * gx

    def loss_at(o):
        xo, _, _ = solve_forward(field, x0, o, step, n_steps)
        return float(np.mean((xo - target) ** 2))

    lo = max(0.05, order - delta)
    hi = min(0.99, order + delta)
    d_order = (loss_at(hi) - loss_at(lo)) / (hi - lo)
    return loss, grads, d_order


def train_joint(x0, target, step, n_steps, seed, n_epochs=1500,
                learning_rate=1e-2, order_rate=3e-3, order_init=0.5,
                width=32, weight_decay=0.0):
    """Adam on the network weights and on the order, from one initialization."""
    field = TrainableField(x0.shape[1], width=width, seed=seed)
    params = field.parameters()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    m_o = v_o = 0.0
    order = float(order_init)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for t in range(1, n_epochs + 1):
        loss, grads, d_order = loss_and_grads(field, x0, order, step, n_steps, target)
        for i, p in enumerate(params):
            g = grads[i] + weight_decay * p
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            p -= learning_rate * mh / (np.sqrt(vh) + eps)
        m_o = b1 * m_o + (1 - b1) * d_order
        v_o = b2 * v_o + (1 - b2) * d_order * d_order
        order -= order_rate * (m_o / (1 - b1 ** t)) / (np.sqrt(v_o / (1 - b2 ** t)) + eps)
        order = float(np.clip(order, 0.30, 0.90))
        history.append((loss, order))
    final_loss, _, _ = loss_and_grads(field, x0, order, step, n_steps, target)
    return order, final_loss, history


# --------------------------------------------------------------------------- #
def run(true_order=0.60, horizon=1.0, n_steps=200, n_trajectories=16,
        n_inits=6, n_epochs=800, noise_levels=(0.0, 0.01, 0.03),
        seed=0, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    generator = NeuralVectorField(seed=1)
    step = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)
    x0 = rng.normal(0.0, 0.45, (n_trajectories, generator.dim))
    reference = solve_caputo(generator, x0, true_order, step, n_steps)
    scale = float(np.sqrt(np.mean(reference ** 2)))

    operators = {b: l1_operator(b, n_steps, step) for b in ORDERS}
    flat = reference.reshape(-1, generator.dim)
    budget = 1.2 * empirical_lipschitz(flat, generator(flat), seed=seed)
    start, stop = int(0.20 * n_steps), int(0.95 * n_steps)

    rows, per_init = [], []
    for noise in noise_levels:
        sigma = noise * scale
        observations = reference + (
            rng.normal(0.0, sigma, reference.shape) if sigma > 0 else 0.0)
        orders, losses = [], []
        for s in range(n_inits):
            o, l, _ = train_joint(x0, observations, step, n_steps, seed=100 + s,
                                  n_epochs=n_epochs,
                                  order_init=0.40 + 0.05 * (s % 5))
            orders.append(o); losses.append(l)
            per_init.append(dict(noise=noise, initialization=s,
                                 recovered_order=o, order_error=abs(o - true_order),
                                 final_loss=l,
                                 relative_rmse=float(np.sqrt(l) / scale)))
        orders = np.asarray(orders)
        errors = np.abs(orders - true_order)
        lo, hi = bootstrap_interval(errors)

        r = estimate_order(observations, t, ORDERS, operators, budget, sigma,
                           start, stop, variant="consensus")
        rows.append(dict(
            noise=noise, n_initializations=n_inits,
            joint_order_mean=float(orders.mean()),
            joint_order_sd=float(orders.std(ddof=1)),
            joint_order_min=float(orders.min()),
            joint_order_max=float(orders.max()),
            joint_mae=float(errors.mean()),
            joint_ci_low=lo, joint_ci_high=hi,
            joint_loss_mean=float(np.mean(losses)),
            joint_loss_sd=float(np.std(losses, ddof=1)),
            joint_relative_rmse=float(np.mean(np.sqrt(losses)) / scale),
            proposed_order=float(r["argmin"]),
            proposed_error=abs(r["argmin"] - true_order),
            proposed_covers=bool(covers(r, true_order)),
        ))
    _write_csv(os.path.join(output_dir, "joint_training.csv"), rows)
    _write_csv(os.path.join(output_dir, "joint_training_runs.csv"), per_init)
    return rows, per_init


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{row[k]:.4f}" if isinstance(row.get(k), float)
                            else row.get(k, "")) for k in keys})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    started = time.time()
    kw = dict(output_dir=a.output_dir)
    if a.quick:
        kw.update(n_steps=30, n_trajectories=4, n_inits=2, n_epochs=200,
                  noise_levels=(0.0, 0.03))
    rows, _ = run(**kw)
    for r in rows:
        print(f"noise {r['noise']:g}: joint order "
              f"{r['joint_order_mean']:.3f} +/- {r['joint_order_sd']:.3f} "
              f"(range {r['joint_order_min']:.3f} to {r['joint_order_max']:.3f}), "
              f"loss {r['joint_loss_mean']:.2e}, "
              f"proposed {r['proposed_order']:.3f}")
    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
