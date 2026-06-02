"""Tests for src.models.baseline_random.RandomWalkBaseline."""

from __future__ import annotations

import pytest
import torch
import networkx as nx

from src.models.baseline_random import RandomWalkBaseline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def triangle_graph() -> nx.DiGraph:
    """Fully-connected triangle DiGraph with symmetric edges (length in metres).

    Nodes: 'A', 'B', 'C'
    Edges: A↔B (100 m), B↔C (200 m), A↔C (150 m)
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", length=100.0)
    G.add_edge("B", "A", length=100.0)
    G.add_edge("B", "C", length=200.0)
    G.add_edge("C", "B", length=200.0)
    G.add_edge("A", "C", length=150.0)
    G.add_edge("C", "A", length=150.0)
    return G


def _make_baseline(
    G: nx.DiGraph,
    T: int = 20,
    node_order: list[str] | None = None,
    exit_rate: float = 0.02,
    arrivals: torch.Tensor | None = None,
) -> RandomWalkBaseline:
    node_order = node_order or ["A", "B", "C"]
    N = len(node_order)
    if arrivals is None:
        arrivals = torch.zeros(T, N)
        arrivals[:, 0] = 10.0  # arrivals at node A
    return RandomWalkBaseline(
        G, arrival_rates=arrivals, dt=5 / 60,
        node_order=node_order, exit_rate=exit_rate,
    )


# ---------------------------------------------------------------------------
# Transition matrix tests
# ---------------------------------------------------------------------------


class TestTransitionMatrix:
    def test_shape(self, triangle_graph):
        bl = _make_baseline(triangle_graph)
        assert bl.P.shape == (3, 3)

    def test_row_stochastic(self, triangle_graph):
        bl = _make_baseline(triangle_graph)
        row_sums = bl.P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)

    def test_no_negative(self, triangle_graph):
        bl = _make_baseline(triangle_graph)
        assert bl.P.min().item() >= 0.0

    def test_dtype_float32(self, triangle_graph):
        bl = _make_baseline(triangle_graph)
        assert bl.P.dtype == torch.float32

    def test_sink_node_self_loop(self):
        """A node with no outgoing edges should get a self-loop (P[i,i]=1)."""
        G = nx.DiGraph()
        G.add_edge("A", "B", length=100.0)
        # C has no outgoing edges — it is a sink
        G.add_node("C")
        node_order = ["A", "B", "C"]
        arrivals = torch.zeros(10, 3)
        bl = RandomWalkBaseline(G, arrivals, node_order=node_order)
        # Row 2 (C) must self-loop
        assert abs(float(bl.P[2, 2].item()) - 1.0) < 1e-6

    def test_shorter_edges_get_higher_prob(self, triangle_graph):
        """From A: edge to B (100 m) should get higher prob than edge to C (150 m)."""
        bl = _make_baseline(triangle_graph)
        # node_order = ['A', 'B', 'C'] → indices 0, 1, 2
        assert float(bl.P[0, 1].item()) > float(bl.P[0, 2].item())


# ---------------------------------------------------------------------------
# Simulate tests
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_output_shape(self, triangle_graph):
        T = 30
        bl = _make_baseline(triangle_graph, T=T)
        traj = bl.simulate(seed=42)
        assert traj.shape == (T, 3)

    def test_output_dtype(self, triangle_graph):
        bl = _make_baseline(triangle_graph)
        traj = bl.simulate(seed=42)
        assert traj.dtype == torch.float32

    def test_nonnegative(self, triangle_graph):
        bl = _make_baseline(triangle_graph)
        traj = bl.simulate(seed=42)
        assert traj.min().item() >= 0.0

    def test_deterministic_with_seed(self, triangle_graph):
        bl = _make_baseline(triangle_graph)
        t1 = bl.simulate(seed=7)
        t2 = bl.simulate(seed=7)
        assert torch.allclose(t1, t2)

    def test_different_seeds_same_result(self, triangle_graph):
        """Simulation is deterministic (no stochastic ops), so seed doesn't matter."""
        bl = _make_baseline(triangle_graph)
        t1 = bl.simulate(seed=1)
        t2 = bl.simulate(seed=99)
        # The simulation itself is fully deterministic; seeds only affect
        # torch random state for future stochastic extensions.
        assert torch.allclose(t1, t2)

    def test_zero_arrivals_decays(self, triangle_graph):
        """With no arrivals and exit_rate > 0, density should decay toward 0."""
        T = 50
        node_order = ["A", "B", "C"]
        # Start with some tourists already in the system — inject via initial step
        arrivals = torch.zeros(T, 3)
        arrivals[0, 0] = 1000.0  # big pulse at step 0
        bl = RandomWalkBaseline(
            triangle_graph, arrivals, dt=5 / 60,
            node_order=node_order, exit_rate=0.1,
        )
        traj = bl.simulate(seed=0)
        # Total tourists at last step should be much less than at peak
        assert traj[-1].sum().item() < traj[5].sum().item()

    def test_arrivals_increase_density(self, triangle_graph):
        """With non-zero arrivals, trajectory sum should grow initially."""
        T = 10
        node_order = ["A", "B", "C"]
        arrivals = torch.zeros(T, 3)
        arrivals[:, 0] = 100.0  # constant arrivals at A
        bl = RandomWalkBaseline(
            triangle_graph, arrivals, dt=5 / 60,
            node_order=node_order, exit_rate=0.0,  # no exit so density monotone
        )
        traj = bl.simulate(seed=0)
        # With no exit, total tourists should be strictly increasing
        totals = traj.sum(dim=1)
        assert totals[-1].item() > totals[0].item()

    def test_first_step_is_zero(self, triangle_graph):
        """trajectory[0] records state *before* first movement, so starts at 0."""
        bl = _make_baseline(triangle_graph)
        traj = bl.simulate(seed=0)
        assert traj[0].sum().item() == pytest.approx(0.0, abs=1e-6)
