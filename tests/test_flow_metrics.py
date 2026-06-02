"""Unit tests for the flow-based cross-cutting metrics (metrics 6 & 7).

Exercises ``MFGSolver.transition_flows`` and the ``mean_attractions_visited`` /
``mean_walking_distance`` metrics on a small synthetic 4-node star graph (no OSM
required).
"""

from __future__ import annotations

import networkx as nx
import torch

from src.evaluation.metrics import mean_attractions_visited, mean_walking_distance
from src.models.mfg_solver import MFGSolver


def _make_star_graph(edge_length_m: float = 200.0) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    for i in range(4):
        for j in range(4):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


def _make_solver() -> MFGSolver:
    return MFGSolver(
        G=_make_star_graph(),
        params={
            "alpha": torch.tensor([0.0, 2.0, 1.0, 0.5], dtype=torch.float32),
            "beta": torch.tensor(0.0133, dtype=torch.float32),
            "gamma": torch.tensor(0.0, dtype=torch.float32),
        },
        dt=0.08333,
        T=14.0,
        epsilon=0.1,
        tol=5e-4,
        max_iter=200,
        node_order=list(range(4)),
    )


def _make_arrivals(T_steps: int, dt: float, n_tourists: float) -> torch.Tensor:
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - 2.0) / 1.5) ** 2)
    g_norm = g_raw / (g_raw.sum() * dt + 1e-12)
    g = torch.zeros(T_steps, 4, dtype=torch.float32)
    g[:, 0] = (n_tourists * g_norm).float()  # all arrivals at transit node 0
    return g


def _solve_and_flows():
    solver = _make_solver()
    n_tourists = 100.0
    g = _make_arrivals(solver.T_steps, solver.dt, n_tourists)
    with torch.no_grad():
        rho, u, _ = solver.fixed_point_iteration(g, damping=0.5)
    flows = solver.transition_flows(rho, u)
    return solver, flows, g, n_tourists


def test_transition_flows_shape_and_nonneg():
    solver, flows, _, _ = _solve_and_flows()
    assert flows.shape == (solver.T_steps - 1, 4, 4)
    assert torch.all(flows >= 0.0)
    assert torch.isfinite(flows).all()


def test_flow_mass_conserved_per_step():
    """Each step's flow rows must sum to <= the source density (rest exits)."""
    solver, flows, _, _ = _solve_and_flows()
    with torch.no_grad():
        rho, u, _ = solver.fixed_point_iteration(
            _make_arrivals(solver.T_steps, solver.dt, 100.0), damping=0.5
        )
    # Row sum of flow[t] is the moving mass; cannot exceed rho[t] (exit absorbs rest).
    moving = flows.sum(dim=2)  # (T-1, N)
    assert torch.all(moving <= rho[:-1] + 1e-4)


def test_mean_attractions_visited_positive_and_normalised():
    solver, flows, _, n_tourists = _solve_and_flows()
    mask = torch.tensor([False, True, True, True])  # nodes 1-3 are attractions
    visited = mean_attractions_visited(flows, mask, n_tourists)
    assert visited > 0.0
    # Entries per tourist cannot exceed the number of movement steps (one per step).
    assert visited < solver.T_steps


def test_mean_walking_distance_matches_manual():
    _, flows, _, n_tourists = _solve_and_flows()
    solver = _make_solver()
    dist = mean_walking_distance(flows, solver.D, n_tourists)
    assert dist >= 0.0
    # Manual recompute with inf masked.
    d = torch.where(torch.isinf(solver.D), torch.zeros_like(solver.D), solver.D)
    manual = float((flows * d).sum().item()) / n_tourists
    assert abs(dist - manual) < 1e-6


def test_zero_gamma_still_walks():
    """gamma=0 means no walk penalty, so tourists do move between nodes."""
    _, flows, _, n_tourists = _solve_and_flows()
    solver = _make_solver()
    dist = mean_walking_distance(flows, solver.D, n_tourists)
    assert dist > 0.0
