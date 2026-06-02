"""Tests for src.models.mfg_solver.MFGSolver.

EXP-03 benchmark: solver validation on synthetic data (analytical verification
on a 3-node and 4-node graph where equilibrium can be computed by hand).
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch

from src.models.mfg_solver import MFGSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def triangle_graph() -> nx.DiGraph:
    """Fully connected 3-node DiGraph (symmetric distances)."""
    G = nx.DiGraph()
    for i in range(3):
        for j in range(3):
            if i != j:
                G.add_edge(i, j, length=100.0)
    return G


@pytest.fixture()
def star_graph_4() -> nx.DiGraph:
    """4-node graph: node 0 is transit hub, nodes 1-3 are attractions.

    All-pairs edges (K_4 directed), equal distance 200 m.
    alpha = [0, 2, 1, 0.5], beta=0, gamma=0 → no congestion, no walk cost.
    """
    G = nx.DiGraph()
    for i in range(4):
        for j in range(4):
            if i != j:
                G.add_edge(i, j, length=200.0)
    return G


def _make_solver(
    G: nx.DiGraph,
    alpha: list[float],
    beta: float = 0.0,
    gamma: float = 0.0,
    epsilon: float = 0.1,
    dt: float = 5 / 60,
    T: float = 1.0,   # 1-hour horizon → 12 steps
    tol: float = 1e-5,
    max_iter: int = 200,
) -> MFGSolver:
    N = len(alpha)
    params = {
        "alpha": torch.tensor(alpha, dtype=torch.float32),
        "beta": beta,
        "gamma": gamma,
    }
    return MFGSolver(G, params, dt=dt, T=T, epsilon=epsilon, tol=tol, max_iter=max_iter)


def _zero_arrivals(T_steps: int, N: int) -> torch.Tensor:
    return torch.zeros(T_steps, N)


def _arrivals_at(node: int, T_steps: int, N: int, rate: float = 1.0) -> torch.Tensor:
    g = torch.zeros(T_steps, N)
    g[:, node] = rate
    return g


# ---------------------------------------------------------------------------
# Distance matrix tests
# ---------------------------------------------------------------------------


class TestDistanceMatrix:
    def test_shape(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        assert solver.D.shape == (3, 3)

    def test_diagonal_is_zero(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        assert solver.D.diagonal().sum().item() == pytest.approx(0.0)

    def test_edges_populated(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        # All off-diagonal entries should be 100.0
        off_diag = solver.D[~torch.eye(3, dtype=torch.bool)]
        assert (off_diag == 100.0).all()

    def test_missing_edge_is_inf(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, length=100.0)  # only one direction, node 2 isolated
        G.add_node(2)
        solver = _make_solver(G, alpha=[0, 1, 0.5], T=0.5)
        # D[1, 0] has no edge → inf
        assert solver.D[1, 0].item() == float("inf")
        # D[0, 2] has no edge → inf
        assert solver.D[0, 2].item() == float("inf")


# ---------------------------------------------------------------------------
# HJB backward tests
# ---------------------------------------------------------------------------


class TestHJBBackward:
    def test_output_shape(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        rho = torch.zeros(solver.T_steps, 3)
        u = solver.solve_hjb_backward(rho)
        assert u.shape == (solver.T_steps, 3)

    def test_terminal_condition(self, triangle_graph):
        """u[T-1, :] should equal alpha * dt (immediate reward, no continuation)."""
        alpha = [2.0, 1.0, 0.5]
        dt = 5 / 60
        solver = _make_solver(triangle_graph, alpha=alpha, dt=dt, T=0.5)
        rho = torch.zeros(solver.T_steps, 3)
        u = solver.solve_hjb_backward(rho)
        for v, a in enumerate(alpha):
            assert u[-1, v].item() == pytest.approx(a * dt, rel=1e-4)

    def test_dtype_float32(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        rho = torch.zeros(solver.T_steps, 3)
        u = solver.solve_hjb_backward(rho)
        assert u.dtype == torch.float32

    def test_higher_alpha_gives_higher_u(self, triangle_graph):
        """With no congestion, node with highest alpha should have highest u at t=0."""
        alpha = [0.5, 2.0, 1.0]
        solver = _make_solver(triangle_graph, alpha=alpha, beta=0.0, gamma=0.0)
        rho = torch.zeros(solver.T_steps, 3)
        u = solver.solve_hjb_backward(rho)
        # Node 1 has highest alpha → highest u
        assert u[0, 1].item() > u[0, 2].item() > u[0, 0].item()

    def test_congestion_reduces_u(self, triangle_graph):
        """Adding congestion at a node should reduce its cost-to-go."""
        alpha = [1.0, 1.0, 1.0]
        solver_no_cong = _make_solver(triangle_graph, alpha=alpha, beta=0.0)
        solver_cong = _make_solver(triangle_graph, alpha=alpha, beta=1.0)

        rho_high = torch.zeros(solver_no_cong.T_steps, 3)
        rho_high[:, 0] = 0.5  # dense at node 0

        u_no = solver_no_cong.solve_hjb_backward(torch.zeros_like(rho_high))
        u_yes = solver_cong.solve_hjb_backward(rho_high)

        # Node 0's u should be lower with high congestion
        assert u_yes[0, 0].item() < u_no[0, 0].item()

    def test_u_nonnegative_when_alpha_positive(self, triangle_graph):
        """With positive alpha and small beta/gamma, u should be non-negative."""
        solver = _make_solver(triangle_graph, alpha=[2.0, 1.0, 0.5], beta=0.01)
        rho = torch.zeros(solver.T_steps, 3)
        rho[:, :] = 0.01  # small density
        u = solver.solve_hjb_backward(rho)
        assert u.min().item() >= -1e-6  # allow tiny negatives from float arithmetic


# ---------------------------------------------------------------------------
# FP forward tests
# ---------------------------------------------------------------------------


class TestFPForward:
    def test_output_shape(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        u = torch.zeros(solver.T_steps, 3)
        g = _zero_arrivals(solver.T_steps, 3)
        rho = solver.solve_fp_forward(u, g)
        assert rho.shape == (solver.T_steps, 3)

    def test_nonnegative(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 2.0, 0.5], beta=0.5)
        u = torch.zeros(solver.T_steps, 3)
        g = _arrivals_at(0, solver.T_steps, 3, rate=1.0)
        rho = solver.solve_fp_forward(u, g)
        assert rho.min().item() >= -1e-6

    def test_zero_arrivals_gives_zero_density(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        u = torch.zeros(solver.T_steps, 3)
        g = _zero_arrivals(solver.T_steps, 3)
        rho = solver.solve_fp_forward(u, g)
        assert rho.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_first_step_is_zero(self, triangle_graph):
        """rho[0] is the initial state (no tourists at t=0 with zero rho_init)."""
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        u = torch.zeros(solver.T_steps, 3)
        g = _arrivals_at(0, solver.T_steps, 3, rate=100.0)
        rho = solver.solve_fp_forward(u, g)
        assert rho[0].abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_arrivals_accumulate(self, triangle_graph):
        """With constant arrivals, total density should increase over time."""
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        u = torch.zeros(solver.T_steps, 3)
        g = _arrivals_at(0, solver.T_steps, 3, rate=10.0)
        rho = solver.solve_fp_forward(u, g)
        # With exit option, mass doesn't strictly increase, but should be > 0 after step 1
        assert rho[1].sum().item() > 0.0

    def test_dtype_float32(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        u = torch.zeros(solver.T_steps, 3)
        g = _zero_arrivals(solver.T_steps, 3)
        rho = solver.solve_fp_forward(u, g)
        assert rho.dtype == torch.float32


# ---------------------------------------------------------------------------
# Fixed-point iteration tests
# ---------------------------------------------------------------------------


class TestFixedPointIteration:
    def test_output_shapes(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        g = _arrivals_at(0, solver.T_steps, 3, rate=1.0)
        rho_eq, u_eq, info = solver.fixed_point_iteration(g)
        assert rho_eq.shape == (solver.T_steps, 3)
        assert u_eq.shape == (solver.T_steps, 3)
        assert "converged" in info
        assert "n_iter" in info
        assert "final_residual" in info

    def test_converges_small_graph(self, triangle_graph):
        """Fixed-point iteration should converge on a simple 3-node graph."""
        solver = _make_solver(triangle_graph, alpha=[1.0, 0.5, 0.2], beta=0.5, tol=1e-5)
        g = _arrivals_at(0, solver.T_steps, 3, rate=0.5)
        _, _, info = solver.fixed_point_iteration(g)
        assert info["converged"], (
            f"Did not converge after {info['n_iter']} iters, "
            f"residual={info['final_residual']:.2e}"
        )

    def test_nonnegative_equilibrium(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[2.0, 1.0, 0.5], beta=0.5)
        g = _arrivals_at(0, solver.T_steps, 3, rate=1.0)
        rho_eq, _, _ = solver.fixed_point_iteration(g)
        assert rho_eq.min().item() >= -1e-5

    def test_zero_arrivals_zero_density(self, triangle_graph):
        """With no arrivals, equilibrium density should be (near) zero."""
        solver = _make_solver(triangle_graph, alpha=[1.0, 0.5, 0.2])
        g = _zero_arrivals(solver.T_steps, 3)
        rho_eq, _, _ = solver.fixed_point_iteration(g)
        assert rho_eq.abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_info_dict_structure(self, triangle_graph):
        solver = _make_solver(triangle_graph, alpha=[1.0, 1.0, 1.0])
        g = _arrivals_at(0, solver.T_steps, 3, rate=1.0)
        _, _, info = solver.fixed_point_iteration(g)
        assert isinstance(info["n_iter"], int)
        assert isinstance(info["converged"], bool)
        assert isinstance(info["final_residual"], float)
        assert info["n_iter"] >= 1

    # ----- Analytical benchmark (EXP-03 core check) -----

    def test_exp03_higher_alpha_attracts_more(self, star_graph_4):
        """EXP-03 benchmark: with beta=0, gamma=0, higher-alpha attractions
        should receive higher equilibrium density (monotone ordering)."""
        # alpha[0]=0 (transit), alpha[1]=2, alpha[2]=1, alpha[3]=0.5
        alpha = [0.0, 2.0, 1.0, 0.5]
        solver = _make_solver(
            star_graph_4, alpha=alpha, beta=0.0, gamma=0.0,
            epsilon=0.1, T=1.0, tol=1e-6,
        )
        # Arrivals at transit node 0 only
        g = _arrivals_at(0, solver.T_steps, 4, rate=1.0)
        rho_eq, u_eq, info = solver.fixed_point_iteration(g)

        # Time-average density at each attraction node
        rho_mean = rho_eq.mean(dim=0)  # (4,)
        # Attractions 1, 2, 3 should be in descending density order
        assert rho_mean[1] > rho_mean[2], (
            f"Expected rho[1]={rho_mean[1]:.4f} > rho[2]={rho_mean[2]:.4f}"
        )
        assert rho_mean[2] > rho_mean[3], (
            f"Expected rho[2]={rho_mean[2]:.4f} > rho[3]={rho_mean[3]:.4f}"
        )

    def test_exp03_congestion_redistributes(self, star_graph_4):
        """EXP-03 benchmark: adding congestion (beta>0) should push tourists
        away from the top attraction, making distribution more equal."""
        alpha = [0.0, 2.0, 1.0, 0.5]
        g = _arrivals_at(0, 12, 4, rate=1.0)  # T=1h, dt=5min → 12 steps

        solver_no_cong = _make_solver(star_graph_4, alpha=alpha, beta=0.0, T=1.0)
        solver_cong = _make_solver(star_graph_4, alpha=alpha, beta=2.0, T=1.0)

        rho_no, _, _ = solver_no_cong.fixed_point_iteration(g)
        rho_yes, _, _ = solver_cong.fixed_point_iteration(g)

        # Node 1 (highest alpha) should have LOWER share with congestion
        mean_no = rho_no.mean(dim=0)
        mean_yes = rho_yes.mean(dim=0)
        # High-alpha node 1 share decreases with congestion
        assert mean_yes[1] < mean_no[1], (
            "Congestion should reduce density at top attraction"
        )
        # Lower-alpha nodes should get higher share under congestion
        assert mean_yes[2] > mean_no[2] or mean_yes[3] > mean_no[3], (
            "Congestion should push some tourists to lower-alpha attractions"
        )

    def test_exp03_analytical_beta0_softmax(self, star_graph_4):
        """EXP-03 analytical check: with beta=0, gamma=0, tourists injected at
        transit at t=0 should mostly move to node 1 (highest alpha) at t=2.

        Timeline with 3 steps:
          t=0: tourists injected at transit (node 0) via g[0,0]
          t=1: rho[1,0] = 1.0 (at transit, not yet moved)
          t=2: rho[2,1] >> others (moved from transit to best attraction)

        Policy at t=1 (transit → attractions):
          Q_cont[0, w] = alpha[w]*dt  (gamma=0, u[2,w]=alpha[w]*dt terminal)
          pi[0→1] = softmax([0, 3*dt, 0, 0, 0] / eps)[1] → dominates with small eps
        """
        eps = 0.05    # small epsilon → near-greedy
        dt = 5 / 60
        T = 3 * dt    # 3 steps: arrive→at_transit→at_attractions

        alpha = [0.0, 3.0, 0.0, 0.0]  # only node 1 is attractive
        solver = _make_solver(
            star_graph_4, alpha=alpha, beta=0.0, gamma=0.0,
            epsilon=eps, dt=dt, T=T, tol=1e-7,
        )
        assert solver.T_steps == 3

        # Inject 1 tourist-unit into transit at t=0: g[0, 0] = 1/dt → rho[1,0] = 1.0
        g = torch.zeros(3, 4)
        g[0, 0] = 1.0 / dt

        rho_eq, u_eq, _ = solver.fixed_point_iteration(g)

        # At t=2: tourists have moved from transit to their chosen attraction
        rho_at_t2 = rho_eq[2]
        total = rho_at_t2.sum()
        assert total > 1e-6, "Expected tourists at t=2"
        frac_at_1 = float(rho_at_t2[1].item()) / float(total.item())
        assert frac_at_1 > 0.5, (
            f"With alpha[1]=3 and eps={eps}, most tourists should be at node 1, "
            f"got fraction={frac_at_1:.3f}"
        )
