"""Tests for the gradient-correctness analysis (Goal B)."""

from __future__ import annotations

import networkx as nx
import torch

from src.calibration.gradient_check import (
    _theta_leaves,
    compare,
    ift_grad,
    one_step_grad,
    unrolled_grad,
)
from src.models.mfg_solver import MFGSolver


def _setup():
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    for i in range(4):
        for j in range(4):
            if i != j:
                G.add_edge(i, j, length=200.0)
    solver = MFGSolver(G=G, params={"alpha": [0.0, 2.0, 1.0, 0.5], "beta": 0.05, "gamma": 0.0},
                       dt=0.08333, T=4.0, epsilon=0.1, tol=1e-5, max_iter=400, node_order=list(range(4)))
    T_steps = solver.T_steps
    t = torch.linspace(0.0, (T_steps - 1) * 0.08333, T_steps)
    g0 = torch.exp(-0.5 * ((t - 1.5) / 1.0) ** 2)
    g0 = g0 / (g0.sum() * 0.08333 + 1e-12)
    g = torch.zeros(T_steps, 4)
    g[:, 0] = 100.0 * g0
    target = torch.tensor([0.10, 0.45, 0.30, 0.15])  # over all 4 nodes (hub + 3 attractions)
    return solver, g, target


def test_unrolled_matches_ift():
    """The unrolled-to-convergence gradient and the IFT gradient must agree."""
    solver, g, target = _setup()
    theta = _theta_leaves(torch.tensor([0.0, 2.0, 1.0, 0.5]), 0.05, 0.0)
    gu = unrolled_grad(solver, g, target, theta, damping=0.5, K=120)
    theta2 = _theta_leaves(torch.tensor([0.0, 2.0, 1.0, 0.5]), 0.05, 0.0)
    gi = ift_grad(solver, g, target, theta2, damping=0.5, n_adjoint=300)
    m = compare(gu, gi)
    assert m["cosine"] > 0.99, m
    assert m["rel_l2_error"] < 0.1, m


def test_one_step_directionally_aligned():
    """The one-step gradient should be positively aligned with the true gradient."""
    solver, g, target = _setup()
    theta = _theta_leaves(torch.tensor([0.0, 2.0, 1.0, 0.5]), 0.05, 0.0)
    g1 = one_step_grad(solver, g, target, theta, damping=0.5)
    theta2 = _theta_leaves(torch.tensor([0.0, 2.0, 1.0, 0.5]), 0.05, 0.0)
    gi = ift_grad(solver, g, target, theta2, damping=0.5, n_adjoint=300)
    m = compare(g1, gi)
    assert m["cosine"] > 0.5, m  # aligned (descent direction), though biased in magnitude
