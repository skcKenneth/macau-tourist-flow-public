"""Gradient-correctness analysis for the MFG calibration (Goal B).

We calibrate with a **one-step consistency gradient**: run the fixed point to
convergence under ``torch.no_grad`` for a stable density ``rho_fp``, then take ONE
HJB+FP step with autograd and backprop through it. This module asks: how close is
that to the *true* gradient of the equilibrium loss with respect to the parameters?

It implements three gradients of the same scalar loss ``L(rho*(theta))`` and lets
the caller compare them:

- ``one_step_grad``  — our method (cheap; treats ``rho_fp`` as constant in theta).
- ``unrolled_grad``  — backprop through the damped iteration unrolled to (near)
  convergence; the practical ground truth when ``K`` is large.
- ``ift_grad``       — the implicit-function-theorem (adjoint) gradient
  ``dL/dtheta = (dL/drho)(I - dS/drho)^{-1}(dS/dtheta)`` at the fixed point
  ``rho* = S(rho*)`` with ``S = FP o HJB``; the exact equilibrium gradient.

Because ``S`` is a contraction in the convergent regime, the adjoint linear solve
``w = gL + (dS/drho)^T w`` converges as a Neumann series and is computed with
autograd vector-Jacobian products. ``unrolled`` and ``ift`` should agree; the
*bias* of ``one_step`` relative to them is what we quantify.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

def _cumulative_distribution(rho: torch.Tensor) -> torch.Tensor:
    """Normalised cumulative (day-summed) density across all nodes."""
    c = rho.sum(dim=0)
    return c / (c.sum() + 1e-12)


def _theta_leaves(alpha: torch.Tensor, beta: float, gamma: float) -> dict[str, torch.Tensor]:
    """Build leaf parameter tensors (requires_grad) for differentiation."""
    return {
        "alpha": alpha.clone().detach().requires_grad_(True),
        "beta": torch.tensor(float(beta), requires_grad=True),
        "gamma": torch.tensor(float(gamma), requires_grad=True),
    }


def _loss(rho: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between the model's normalised cumulative node distribution and a target."""
    return torch.nn.functional.mse_loss(_cumulative_distribution(rho), target)


def _flat(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def one_step_grad(solver, g, target, theta, damping: float) -> torch.Tensor:
    """Our one-step consistency gradient (rho_fp detached from theta)."""
    with torch.no_grad():
        solver.params = {k: v.detach() for k, v in theta.items()}
        rho_fp, _, _ = solver.fixed_point_iteration(g, damping=damping)
    solver.params = theta
    u = solver.solve_hjb_backward(rho_fp)
    rho = solver.solve_fp_forward(u, g)
    L = _loss(rho, target)
    grads = torch.autograd.grad(L, list(theta.values()))
    return _flat(list(grads))


def unrolled_grad(solver, g, target, theta, damping: float, K: int = 80) -> torch.Tensor:
    """Backprop through the damped iteration unrolled K times (ground truth)."""
    solver.params = theta
    N = solver.N_nodes
    rho = torch.zeros(solver.T_steps, N, dtype=torch.float32)
    for _ in range(K):
        u = solver.solve_hjb_backward(rho)
        rho_new = solver.solve_fp_forward(u, g)
        rho = (1.0 - damping) * rho + damping * rho_new
    L = _loss(rho, target)
    grads = torch.autograd.grad(L, list(theta.values()))
    return _flat(list(grads))


def ift_grad(solver, g, target, theta, damping: float, n_adjoint: int = 200) -> torch.Tensor:
    """Implicit-function-theorem (adjoint) gradient at the fixed point."""
    # 1) Equilibrium (no grad).
    with torch.no_grad():
        solver.params = {k: v.detach() for k, v in theta.items()}
        rho_star, _, _ = solver.fixed_point_iteration(g, damping=damping)
    rho_star = rho_star.detach().requires_grad_(True)

    # 2) One application of the DAMPED map T_lambda = (1-lambda) rho + lambda*FP(HJB)
    #    at rho_star (grad on). The fixed point is identical to that of FP o HJB, but
    #    T_lambda is a contraction in the convergent regime, so the adjoint Neumann
    #    series below converges (the undamped map need not be contractive).
    solver.params = theta
    u = solver.solve_hjb_backward(rho_star)
    S_undamped = solver.solve_fp_forward(u, g)
    S = (1.0 - damping) * rho_star + damping * S_undamped

    # 3) dL/drho at the fixed point.
    L = _loss(rho_star, target)
    gL = torch.autograd.grad(L, rho_star, retain_graph=True)[0]

    # 4) Adjoint solve  w = gL + (dS/drho)^T w   (Neumann iteration; S is a contraction).
    w = gL.clone()
    for _ in range(n_adjoint):
        JT_w = torch.autograd.grad(S, rho_star, grad_outputs=w, retain_graph=True)[0]
        w_new = gL + JT_w
        if float((w_new - w).abs().max().item()) < 1e-9:
            w = w_new
            break
        w = w_new

    # 5) dL/dtheta = (dS/dtheta)^T w.
    grads = torch.autograd.grad(S, list(theta.values()), grad_outputs=w)
    return _flat(list(grads))


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Cosine similarity and relative L2 error of gradient ``a`` vs reference ``b``."""
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    rel = float((a - b).norm().item() / (b.norm().item() + 1e-12))
    return {"cosine": cos, "rel_l2_error": rel}
