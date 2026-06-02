"""Tests for the two-type compliance solver (EXP-09 / Goal D)."""

from __future__ import annotations

import networkx as nx
import torch

from src.models.mfg_solver import MFGSolver


def _star(n=4, length=200.0):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, length=length)
    return G


def _arrivals(T_steps, dt, n=100.0, n_nodes=4):
    t = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g0 = torch.exp(-0.5 * ((t - 2.0) / 1.5) ** 2)
    g0 = g0 / (g0.sum() * dt + 1e-12)
    g = torch.zeros(T_steps, n_nodes)
    g[:, 0] = n * g0
    return g


def _solver():
    G = _star(4)
    params = {"alpha": [0.0, 3.0, 0.3, 0.3], "beta": 0.05, "gamma": 0.0}
    return MFGSolver(G=G, params=params, dt=0.08333, T=6.0, epsilon=0.1,
                     tol=5e-4, max_iter=300, node_order=list(range(4)))


def _eta():
    e = torch.zeros(4, 4)
    e[0, 2] = 0.3   # nudge hub -> node 2 (a less-attractive node)
    e[0, 3] = 0.3
    return e


class TestComplianceLimits:
    def test_phi_one_matches_full_routing(self):
        """phi=1 must equal the standard equilibrium with routing_bonus=eta."""
        solver = _solver()
        g = _arrivals(solver.T_steps, solver.dt)
        eta = _eta()

        rho_phi1, info = solver.fixed_point_iteration_compliance(g, eta, phi=1.0, damping=0.5)
        solver.routing_bonus = eta
        with torch.no_grad():
            rho_ref, _, _ = solver.fixed_point_iteration(g, damping=0.5)
        solver.routing_bonus = torch.zeros(4, 4)

        assert info["converged"]
        assert torch.allclose(rho_phi1, rho_ref, atol=1e-4)

    def test_phi_zero_matches_baseline(self):
        """phi=0 must equal the no-intervention equilibrium (routing_bonus=0)."""
        solver = _solver()
        g = _arrivals(solver.T_steps, solver.dt)
        eta = _eta()

        rho_phi0, _ = solver.fixed_point_iteration_compliance(g, eta, phi=0.0, damping=0.5)
        with torch.no_grad():
            rho_ref, _, _ = solver.fixed_point_iteration(g, damping=0.5)

        assert torch.allclose(rho_phi0, rho_ref, atol=1e-4)

    def test_routing_bonus_restored(self):
        """The solver's routing_bonus attribute is left unchanged after the call."""
        solver = _solver()
        marker = torch.full((4, 4), 0.123)
        solver.routing_bonus = marker
        g = _arrivals(solver.T_steps, solver.dt)
        solver.fixed_point_iteration_compliance(g, _eta(), phi=0.5, damping=0.5)
        assert torch.allclose(solver.routing_bonus, marker)


class TestComplianceMonotonicity:
    def test_more_compliance_reduces_bottleneck_peak(self):
        """If eta diverts from node 1, higher compliance lowers node-1 peak."""
        solver = _solver()
        g = _arrivals(solver.T_steps, solver.dt)
        eta = _eta()  # nudges toward nodes 2,3 and away from the dominant node 1

        peaks = []
        for phi in (0.0, 0.5, 1.0):
            rho, _ = solver.fixed_point_iteration_compliance(g, eta, phi=phi, damping=0.5)
            peaks.append(float(rho[:, 1].max().item()))

        # Monotone non-increasing peak at the bottleneck as compliance rises.
        assert peaks[0] >= peaks[1] >= peaks[2] - 1e-6
        assert peaks[2] < peaks[0]
