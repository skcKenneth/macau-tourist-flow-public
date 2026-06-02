"""Unit tests for EXP-07 entrance metering helpers.

Tests use a synthetic 4-node star graph (fast, no OSM required):
  Node 0: transit hub (source of arrivals — the "ferry terminal" analogue)
  Nodes 1–3: attraction nodes

Shared fixture: 100 tourists, Gaussian arrival peak at hour 2,
R1_base recovered parameters from EXP-04.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch


# ---------------------------------------------------------------------------
# Local helpers (avoid importing from run_exp07 until fixture is ready)
# ---------------------------------------------------------------------------

from src.run_exp07 import _compute_metrics, _meter_arrivals


def _make_star_graph(edge_length_m: float = 200.0) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    for i in range(4):
        for j in range(4):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


def _make_arrivals(
    T_steps: int,
    dt: float,
    n_tourists: float,
    peak: float = 2.0,
    sigma: float = 1.5,
    source_col: int = 0,
    n_nodes: int = 4,
) -> torch.Tensor:
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - peak) / sigma) ** 2)
    g_norm = g_raw / (g_raw.sum() * dt + 1e-12)
    g = torch.zeros(T_steps, n_nodes, dtype=torch.float32)
    g[:, source_col] = (n_tourists * g_norm).float()
    return g


@pytest.fixture(scope="module")
def base_setup():
    """4-node star, R1_base recovered parameters, 100 tourists."""
    dt = 0.08333
    T_hours = 14.0
    T_steps = int(round(T_hours / dt))
    g = _make_arrivals(T_steps, dt, 100.0)
    return {
        "G": _make_star_graph(),
        "g": g,
        "dt": dt,
        "T_hours": T_hours,
        "T_steps": T_steps,
        "n_tourists": 100.0,
        "params": {
            "alpha": torch.tensor([0.0, 2.095, 1.022, 0.527], dtype=torch.float32),
            "beta": torch.tensor(0.0133, dtype=torch.float32),
            "gamma": torch.tensor(0.0, dtype=torch.float32),
        },
        "solver_cfg": {
            "dt_hours": dt,
            "T_hours": T_hours,
            "epsilon": 0.1,
            "tol": 5e-4,
            "max_iter": 200,
            "damping": 0.5,
        },
    }


def _make_solver(base_setup):
    from src.models.mfg_solver import MFGSolver
    cfg = base_setup["solver_cfg"]
    return MFGSolver(
        G=base_setup["G"],
        params=base_setup["params"],
        dt=cfg["dt_hours"],
        T=cfg["T_hours"],
        epsilon=cfg["epsilon"],
        tol=cfg["tol"],
        max_iter=cfg["max_iter"],
        node_order=list(range(4)),
    )


def _run_solver(base_setup, g: torch.Tensor) -> torch.Tensor:
    solver = _make_solver(base_setup)
    with torch.no_grad():
        rho, _, _ = solver.fixed_point_iteration(g, damping=base_setup["solver_cfg"]["damping"])
    return rho


# ---------------------------------------------------------------------------
# _meter_arrivals tests
# ---------------------------------------------------------------------------


class TestMeterArrivals:

    def test_meter_preserves_total_arrivals(self, base_setup):
        """Total arrivals at source column must be conserved after metering."""
        g = base_setup["g"]
        dt = base_setup["dt"]
        source_col = 0
        r_star = float(g[:, source_col].max().item()) * 0.5

        g_metered = _meter_arrivals(g, r_star, dt, source_col)

        original_total = float((g[:, source_col] * dt).sum().item())
        metered_total = float((g_metered[:, source_col] * dt).sum().item())
        assert abs(metered_total - original_total) < 1e-3, (
            f"Total arrivals changed: original={original_total:.4f}, "
            f"metered={metered_total:.4f}"
        )

    def test_meter_caps_peak_rate(self, base_setup):
        """All values at source column must be ≤ r_star after metering."""
        g = base_setup["g"]
        dt = base_setup["dt"]
        source_col = 0
        r_star = float(g[:, source_col].max().item()) * 0.5

        g_metered = _meter_arrivals(g, r_star, dt, source_col)

        assert g_metered[:, source_col].max().item() <= r_star + 1e-5, (
            f"Cap exceeded: max={g_metered[:, source_col].max().item():.4f} > r_star={r_star:.4f}"
        )

    def test_meter_no_change_when_cap_above_peak(self, base_setup):
        """If r_star >= current peak, g_metered must equal g (no redistribution)."""
        g = base_setup["g"]
        dt = base_setup["dt"]
        source_col = 0
        # Set r_star well above any arrival value
        r_star = float(g[:, source_col].max().item()) * 2.0

        g_metered = _meter_arrivals(g, r_star, dt, source_col)

        assert torch.allclose(g_metered, g, atol=1e-6), (
            "g_metered should equal g when r_star is above current peak"
        )

    def test_meter_other_columns_unchanged(self, base_setup):
        """Metering source_col must not change any other column."""
        g = base_setup["g"]
        dt = base_setup["dt"]
        source_col = 0
        # Add some dummy arrivals at col 1 (shouldn't happen in practice but tests isolation)
        g_test = g.clone()
        g_test[:, 1] = 0.5
        r_star = float(g_test[:, source_col].max().item()) * 0.5

        g_metered = _meter_arrivals(g_test, r_star, dt, source_col)

        for col in range(1, g_test.shape[1]):
            assert torch.allclose(g_metered[:, col], g_test[:, col], atol=1e-6), (
                f"Column {col} was modified by metering (should be untouched)"
            )


# ---------------------------------------------------------------------------
# Solver-based tests
# ---------------------------------------------------------------------------


class TestMeteringEffect:

    def test_aggressive_cap_reduces_peak_density(self, base_setup):
        """Aggressive metering (R* = 10% of peak) should reduce peak density at node 1.

        Note: The 4-node star toy model has small congestion cost (beta≈0.01), so
        metering reduces peak density only mildly (~2%). We test for >= 1% reduction
        to verify the directional effect is correct without over-constraining toy dynamics.
        """
        g = base_setup["g"]
        dt = base_setup["dt"]
        source_col = 0
        ruins_idx = 1  # bottleneck node in the 4-node star

        # Baseline
        rho_base = _run_solver(base_setup, g)
        m_base = _compute_metrics(rho_base, g, dt, ruins_idx=ruins_idx)

        # Aggressive metering: R* = 10% of peak rate
        r_star = float(g[:, source_col].max().item()) * 0.10
        g_metered = _meter_arrivals(g, r_star, dt, source_col)
        rho_met = _run_solver(base_setup, g_metered)
        m_met = _compute_metrics(rho_met, g_metered, dt, ruins_idx=ruins_idx)

        # At least 1% directional reduction expected (toy model has small congestion cost)
        assert m_met["peak_density_ruins"] < m_base["peak_density_ruins"] * 0.99, (
            f"Expected >= 1% peak reduction with aggressive metering: "
            f"baseline={m_base['peak_density_ruins']:.4f}, "
            f"metered={m_met['peak_density_ruins']:.4f}"
        )

    def test_mild_cap_preserves_attraction_hours(self, base_setup):
        """Mild metering (R* = 80% of peak) should preserve total attraction-hours within 5%."""
        g = base_setup["g"]
        dt = base_setup["dt"]
        source_col = 0
        ruins_idx = 1

        # Baseline
        rho_base = _run_solver(base_setup, g)
        m_base = _compute_metrics(rho_base, g, dt, ruins_idx=ruins_idx)

        # Mild metering: R* = 80% of peak rate
        r_star = float(g[:, source_col].max().item()) * 0.80
        g_metered = _meter_arrivals(g, r_star, dt, source_col)
        rho_met = _run_solver(base_setup, g_metered)
        m_met = _compute_metrics(rho_met, g_metered, dt, ruins_idx=ruins_idx)

        rel_change = abs(m_met["total_att_hours"] - m_base["total_att_hours"]) / (
            m_base["total_att_hours"] + 1e-9
        )
        assert rel_change < 0.05, (
            f"Total attraction-hours changed by {100*rel_change:.1f}% with mild metering "
            f"(expected < 5%): "
            f"baseline={m_base['total_att_hours']:.2f}, metered={m_met['total_att_hours']:.2f}"
        )
