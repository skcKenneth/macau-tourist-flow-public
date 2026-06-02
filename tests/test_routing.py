"""Unit tests for EXP-08 routing recommendations.

Covers:
- MFGSolver.routing_bonus: backward-compatibility (zero/None is a no-op), the
  directional effect of a positive bonus, and the inf-mask (no bonus on
  non-edges).
- RoutingOptimizer: zero init, peak-loss reduction, sparsity under heavy L1,
  differentiability of the bonus, and the documented return keys.

Tests use small synthetic graphs (no OSM) and short horizons for speed.
"""

from __future__ import annotations

import networkx as nx
import torch

from src.models.mfg_solver import MFGSolver
from src.optimization.interventions import RoutingOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _star_graph(n: int = 4, edge_length_m: float = 200.0) -> nx.DiGraph:
    """Complete directed graph on n nodes (node 0 = transit hub)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


def _partial_graph() -> nx.DiGraph:
    """3-node graph with edges 0->1 and 0->2 only (no 1<->2 edge => D inf)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(3))
    for j in (1, 2):
        G.add_edge(0, j, length=200.0)
        G.add_edge(j, 0, length=200.0)
    return G


def _arrivals(T_steps: int, dt: float, n_tourists: float, n_nodes: int) -> torch.Tensor:
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - 2.0) / 1.5) ** 2)
    g_norm = g_raw / (g_raw.sum() * dt + 1e-12)
    g = torch.zeros(T_steps, n_nodes, dtype=torch.float32)
    g[:, 0] = (n_tourists * g_norm).float()
    return g


def _make_solver(G, params, T_hours=6.0, routing_bonus=None) -> MFGSolver:
    n = G.number_of_nodes()
    return MFGSolver(
        G=G,
        params=params,
        dt=0.08333,
        T=T_hours,
        epsilon=0.1,
        tol=5e-4,
        max_iter=200,
        node_order=list(range(n)),
        routing_bonus=routing_bonus,
    )


def _solve(solver, g):
    with torch.no_grad():
        rho, _, _ = solver.fixed_point_iteration(g, damping=0.5)
    return rho


# ---------------------------------------------------------------------------
# Solver: backward compatibility
# ---------------------------------------------------------------------------


class TestRoutingBonusBackwardCompat:

    def test_default_routing_bonus_is_zero(self):
        """A solver built without routing_bonus initialises it to an (N,N) zero matrix."""
        G = _star_graph(4)
        params = {"alpha": [0.0, 1.0, 1.0, 1.0], "beta": 0.01, "gamma": 0.0}
        solver = _make_solver(G, params)
        assert solver.routing_bonus.shape == (4, 4)
        assert torch.count_nonzero(solver.routing_bonus).item() == 0

    def test_none_matches_explicit_zeros(self):
        """routing_bonus=None must give the identical equilibrium to explicit zeros."""
        G = _star_graph(4)
        params = {"alpha": [0.0, 2.0, 1.0, 0.5], "beta": 0.01, "gamma": 0.0}
        dt = 0.08333
        T_steps = int(round(6.0 / dt))
        g = _arrivals(T_steps, dt, 100.0, 4)

        rho_none = _solve(_make_solver(G, params, routing_bonus=None), g)
        rho_zero = _solve(
            _make_solver(G, params, routing_bonus=torch.zeros(4, 4)), g
        )
        assert torch.allclose(rho_none, rho_zero, atol=1e-7)

    def test_uniform_alpha_still_symmetric(self):
        """Backward-compat: uniform alpha + zero bonus => ~equal attraction shares.

        Guards the EXP-03 symmetry property after the routing_bonus change.
        """
        G = _star_graph(4)
        params = {"alpha": [0.0, 1.0, 1.0, 1.0], "beta": 0.01, "gamma": 0.0}
        dt = 0.08333
        T_steps = int(round(6.0 / dt))
        g = _arrivals(T_steps, dt, 100.0, 4)
        rho = _solve(_make_solver(G, params), g)
        cum = rho[:, 1:4].sum(dim=0)
        shares = cum / cum.sum()
        assert torch.allclose(shares, torch.full((3,), 1 / 3), atol=1e-3)


# ---------------------------------------------------------------------------
# Solver: routing bonus effect
# ---------------------------------------------------------------------------


class TestRoutingBonusEffect:

    def test_positive_bonus_diverts_mass(self):
        """A positive bonus on edge 0->1 raises node 1's peak above symmetric node 2."""
        G = _star_graph(4)
        params = {"alpha": [0.0, 1.0, 1.0, 1.0], "beta": 0.01, "gamma": 0.0}
        dt = 0.08333
        T_steps = int(round(6.0 / dt))
        g = _arrivals(T_steps, dt, 100.0, 4)

        eta = torch.zeros(4, 4)
        eta[0, 1] = 0.5  # strongly recommend hub -> node 1
        solver = _make_solver(G, params, routing_bonus=eta)
        rho = _solve(solver, g)

        peak1 = float(rho[:, 1].max().item())
        peak2 = float(rho[:, 2].max().item())
        assert peak1 > peak2, (
            f"Bonus on 0->1 should favour node 1: peak1={peak1:.4f}, peak2={peak2:.4f}"
        )

    def test_bonus_on_nonedge_has_no_effect(self):
        """A bonus on a non-existent edge (D=inf) must be masked out (no effect)."""
        G = _partial_graph()  # no edge between 1 and 2
        params = {"alpha": [0.0, 1.0, 1.0], "beta": 0.01, "gamma": 0.0}
        dt = 0.08333
        T_steps = int(round(6.0 / dt))
        g = _arrivals(T_steps, dt, 100.0, 3)

        assert not torch.isfinite(_make_solver(G, params).D[1, 2])

        rho_zero = _solve(_make_solver(G, params, routing_bonus=torch.zeros(3, 3)), g)
        eta = torch.zeros(3, 3)
        eta[1, 2] = 5.0  # huge bonus on a non-edge
        rho_bonus = _solve(_make_solver(G, params, routing_bonus=eta), g)

        assert torch.allclose(rho_zero, rho_bonus, atol=1e-7), (
            "Bonus on a non-edge changed the equilibrium (mask failed)"
        )


# ---------------------------------------------------------------------------
# RoutingOptimizer
# ---------------------------------------------------------------------------


def _opt_setup():
    G = _star_graph(4)
    params = {"alpha": [0.0, 3.0, 0.3, 0.3], "beta": 0.05, "gamma": 0.0}
    dt = 0.08333
    T_steps = int(round(6.0 / dt))
    g = _arrivals(T_steps, dt, 100.0, 4)
    solver = _make_solver(G, params)
    return solver, g


class TestRoutingOptimizer:

    def test_eta_matrix_zero_at_origin(self):
        """_eta_matrix(0) is all zeros (the no-intervention baseline)."""
        solver, g = _opt_setup()
        opt = RoutingOptimizer(solver, g, bottleneck_idx=1, attraction_count=4)
        raw = torch.zeros(len(opt.candidate_edges))
        eta = opt._eta_matrix(raw)
        assert torch.count_nonzero(eta).item() == 0

    def test_optimise_returns_expected_keys(self):
        solver, g = _opt_setup()
        opt = RoutingOptimizer(solver, g, bottleneck_idx=1, attraction_count=4,
                               peak_temp=10.0)
        res = opt.optimise(n_steps=15, lr=5e-2, log_every=100)
        for key in ("eta", "peak_density_reduction_pct", "visit_reduction_pct",
                    "peak_baseline", "peak_optimized", "top_edges", "loss_history"):
            assert key in res
        assert res["eta"].shape == (4, 4)
        assert len(res["loss_history"]) == 15
        assert len(res["top_edges"]) == 5

    def test_optimise_reduces_loss(self):
        """Gradient descent should not increase the objective."""
        solver, g = _opt_setup()
        opt = RoutingOptimizer(solver, g, bottleneck_idx=1, attraction_count=4,
                               lambda_l1=1e-3, peak_temp=10.0)
        res = opt.optimise(n_steps=40, lr=5e-2, log_every=100)
        assert res["loss_history"][-1] <= res["loss_history"][0] + 1e-6

    def test_optimise_reduces_peak(self):
        """The optimised bonus should reduce (not increase) bottleneck peak density."""
        solver, g = _opt_setup()
        opt = RoutingOptimizer(solver, g, bottleneck_idx=1, attraction_count=4,
                               lambda_l1=1e-3, peak_temp=10.0)
        res = opt.optimise(n_steps=40, lr=5e-2, log_every=100)
        assert res["peak_density_reduction_pct"] >= -1e-6

    def test_heavy_l1_keeps_eta_small(self):
        """A very large L1 weight should keep the optimised bonus near zero (sparse)."""
        solver, g = _opt_setup()
        opt = RoutingOptimizer(solver, g, bottleneck_idx=1, attraction_count=4,
                               lambda_l1=1e3, peak_temp=10.0)
        res = opt.optimise(n_steps=40, lr=5e-2, log_every=100)
        assert float(res["eta"].abs().sum().item()) < 0.1

    def test_eta_is_differentiable(self):
        """Gradient must flow from a peak loss back to the raw routing parameter."""
        solver, g = _opt_setup()
        opt = RoutingOptimizer(solver, g, bottleneck_idx=1, attraction_count=4)
        raw = torch.zeros(len(opt.candidate_edges), requires_grad=True)
        eta = opt._eta_matrix(raw)
        solver.routing_bonus = eta
        with torch.no_grad():
            rho_fp, _, _ = solver.fixed_point_iteration(g, damping=0.5)
        u = solver.solve_hjb_backward(rho_fp)
        rho_pred = solver.solve_fp_forward(u, g)
        loss = opt._smooth_peak(rho_pred[:, 1])
        loss.backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()
