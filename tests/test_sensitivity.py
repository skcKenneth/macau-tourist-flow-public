"""Unit tests for EXP-06 sensitivity analysis helpers.

Tests cover the two main helpers exported from src.run_exp06:
- ``_build_perturbed_params``: applies a scalar multiplier to one parameter
- ``_run_single``: runs MFG fixed-point and returns the three target metrics

Shared fixture
--------------
4-node star graph (1 transit hub + 3 attractions, edges=200 m).
Baseline parameters from EXP-04 R1_base recovered values:
  alpha=[0.0, 2.095, 1.022, 0.527]  beta=0.0133  gamma=0.0
100 tourists, Gaussian peak at +2 h, sigma=1.5 h.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers (duplicated from run_exp06 to avoid import-order issues during test)
# ---------------------------------------------------------------------------

def _make_graph(edge_length_m: float = 200.0) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    for i in range(4):
        for j in range(4):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


def _make_arrivals(T_steps: int, dt: float, n_tourists: float,
                   peak: float, sigma: float) -> torch.Tensor:
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - peak) / sigma) ** 2)
    g_norm = g_raw / (g_raw.sum() * dt + 1e-12)
    g = torch.zeros(T_steps, 4, dtype=torch.float32)
    g[:, 0] = (n_tourists * g_norm).float()
    return g


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_setup():
    """4-node star, R1_base recovered params, 100 tourists."""
    dt = 0.08333
    T_hours = 14.0
    T_steps = int(round(T_hours / dt))
    G = _make_graph()
    g = _make_arrivals(T_steps, dt, 100.0, peak=2.0, sigma=1.5)
    baseline = {
        "alpha": [0.0, 2.095, 1.022, 0.527],
        "beta": 0.0133,
        "gamma": 0.0,
    }
    solver_cfg = {
        "dt_hours": dt,
        "T_hours": T_hours,
        "epsilon": 0.1,
        "tol": 5e-4,
        "max_iter": 200,
        "damping": 0.5,
    }
    return {"G": G, "g": g, "baseline": baseline, "solver_cfg": solver_cfg}


# ---------------------------------------------------------------------------
# Import helpers from run_exp06
# ---------------------------------------------------------------------------

from src.run_exp06 import _build_perturbed_params, _run_single


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildPerturbedParams:
    """Tests for _build_perturbed_params()."""

    def test_alpha1_perturb_changes_only_node1(self, base_setup):
        baseline = base_setup["baseline"]
        result = _build_perturbed_params(baseline, "alpha_1", 1.2, n_nodes=4)
        # Node 1 should be perturbed
        assert abs(result["alpha"][1] - baseline["alpha"][1] * 1.2) < 1e-5
        # Nodes 2, 3 should be unchanged
        assert abs(result["alpha"][2] - baseline["alpha"][2]) < 1e-6
        assert abs(result["alpha"][3] - baseline["alpha"][3]) < 1e-6
        # Beta and gamma unchanged
        assert abs(result["beta"] - baseline["beta"]) < 1e-9
        assert abs(result["gamma"] - baseline["gamma"]) < 1e-9

    def test_beta_perturb_changes_only_beta(self, base_setup):
        baseline = base_setup["baseline"]
        result = _build_perturbed_params(baseline, "beta", 0.9, n_nodes=4)
        assert abs(result["beta"] - baseline["beta"] * 0.9) < 1e-8
        # Alpha unchanged
        for i in range(4):
            assert abs(result["alpha"][i] - baseline["alpha"][i]) < 1e-6

    def test_gamma_uses_baseline_override_when_zero(self, base_setup):
        baseline = base_setup["baseline"]  # gamma=0.0
        # Param entry with baseline_override
        result = _build_perturbed_params(
            baseline, "gamma", 1.2, n_nodes=4, baseline_override=1e-4
        )
        # Should be baseline_override * perturb, not 0 * perturb
        assert abs(result["gamma"] - 1e-4 * 1.2) < 1e-12

    def test_perturb_1_returns_baseline_values(self, base_setup):
        baseline = base_setup["baseline"]
        result = _build_perturbed_params(baseline, "beta", 1.0, n_nodes=4)
        assert abs(result["beta"] - baseline["beta"]) < 1e-9
        for i in range(4):
            assert abs(result["alpha"][i] - baseline["alpha"][i]) < 1e-6


class TestRunSingle:
    """Tests for _run_single()."""

    def test_returns_expected_keys(self, base_setup):
        """_run_single returns dict with all required metric keys, all > 0."""
        from src.models.mfg_solver import MFGSolver

        cfg = base_setup["solver_cfg"]
        G = base_setup["G"]
        g = base_setup["g"]
        params_dict = {
            "alpha": base_setup["baseline"]["alpha"],
            "beta": base_setup["baseline"]["beta"],
            "gamma": base_setup["baseline"]["gamma"],
        }
        dt = cfg["dt_hours"]
        T_hours = cfg["T_hours"]
        T_steps = int(round(T_hours / dt))
        node_order = list(range(4))
        alpha_t = torch.tensor(params_dict["alpha"], dtype=torch.float32)
        solver = MFGSolver(
            G=G,
            params={
                "alpha": alpha_t,
                "beta": torch.tensor(params_dict["beta"], dtype=torch.float32),
                "gamma": torch.tensor(params_dict["gamma"], dtype=torch.float32),
            },
            dt=dt, T=T_hours, epsilon=cfg["epsilon"],
            tol=cfg["tol"], max_iter=cfg["max_iter"],
            node_order=node_order,
        )

        metrics = _run_single(solver, g, params_dict, cfg)

        assert "peak_density_node1" in metrics
        assert "gini" in metrics
        assert "mean_attractions" in metrics
        assert metrics["peak_density_node1"] > 0.0
        assert metrics["gini"] >= 0.0

    def test_baseline_perturbation_matches_direct_solve(self, base_setup):
        """At perturb=1.0 (identity), _run_single matches direct FP solve within 1%."""
        from src.models.mfg_solver import MFGSolver

        cfg = base_setup["solver_cfg"]
        G = base_setup["G"]
        g = base_setup["g"]
        params_dict = {
            "alpha": base_setup["baseline"]["alpha"],
            "beta": base_setup["baseline"]["beta"],
            "gamma": base_setup["baseline"]["gamma"],
        }
        dt = cfg["dt_hours"]
        T_hours = cfg["T_hours"]
        node_order = list(range(4))
        alpha_t = torch.tensor(params_dict["alpha"], dtype=torch.float32)
        solver = MFGSolver(
            G=G,
            params={
                "alpha": alpha_t,
                "beta": torch.tensor(params_dict["beta"], dtype=torch.float32),
                "gamma": torch.tensor(params_dict["gamma"], dtype=torch.float32),
            },
            dt=dt, T=T_hours, epsilon=cfg["epsilon"],
            tol=cfg["tol"], max_iter=cfg["max_iter"],
            node_order=node_order,
        )

        # Direct solve
        with torch.no_grad():
            rho_direct, _, _ = solver.fixed_point_iteration(g, damping=cfg["damping"])
        direct_peak = float(rho_direct[:, 1].max().item())

        # Via _run_single
        metrics = _run_single(solver, g, params_dict, cfg)

        # Allow 1% relative tolerance (solver is deterministic, so should be identical)
        assert abs(metrics["peak_density_node1"] - direct_peak) / (direct_peak + 1e-9) < 0.01

    def test_higher_alpha1_increases_peak_density(self, base_setup):
        """Increasing attractiveness at bottleneck node should raise its peak density."""
        from src.models.mfg_solver import MFGSolver

        cfg = base_setup["solver_cfg"]
        G = base_setup["G"]
        g = base_setup["g"]
        dt = cfg["dt_hours"]
        T_hours = cfg["T_hours"]
        node_order = list(range(4))
        baseline = base_setup["baseline"]

        def _make_solver(params_dict):
            return MFGSolver(
                G=G,
                params={
                    "alpha": torch.tensor(params_dict["alpha"], dtype=torch.float32),
                    "beta": torch.tensor(params_dict["beta"], dtype=torch.float32),
                    "gamma": torch.tensor(params_dict["gamma"], dtype=torch.float32),
                },
                dt=dt, T=T_hours, epsilon=cfg["epsilon"],
                tol=cfg["tol"], max_iter=cfg["max_iter"],
                node_order=node_order,
            )

        # Baseline
        p_base = _build_perturbed_params(baseline, "alpha_1", 1.0, n_nodes=4)
        m_base = _run_single(_make_solver(p_base), g, p_base, cfg)

        # +20% alpha_1
        p_high = _build_perturbed_params(baseline, "alpha_1", 1.2, n_nodes=4)
        m_high = _run_single(_make_solver(p_high), g, p_high, cfg)

        assert m_high["peak_density_node1"] > m_base["peak_density_node1"], (
            f"Expected peak_density_node1 to increase with higher alpha_1: "
            f"baseline={m_base['peak_density_node1']:.4f} "
            f"perturbed={m_high['peak_density_node1']:.4f}"
        )

    def test_higher_beta_does_not_increase_peak_density(self, base_setup):
        """Increasing congestion sensitivity should not increase bottleneck peak density.

        Higher beta raises the cost of visiting the crowded bottleneck, causing
        rational tourists to redistribute toward less-crowded alternatives.
        At the toy scale the effect is small but directionally it should be
        non-positive (peak <= baseline peak, allowing 2% numerical tolerance).
        """
        from src.models.mfg_solver import MFGSolver

        cfg = base_setup["solver_cfg"]
        G = base_setup["G"]
        g = base_setup["g"]
        dt = cfg["dt_hours"]
        T_hours = cfg["T_hours"]
        node_order = list(range(4))
        baseline = base_setup["baseline"]

        def _make_solver(params_dict):
            return MFGSolver(
                G=G,
                params={
                    "alpha": torch.tensor(params_dict["alpha"], dtype=torch.float32),
                    "beta": torch.tensor(params_dict["beta"], dtype=torch.float32),
                    "gamma": torch.tensor(params_dict["gamma"], dtype=torch.float32),
                },
                dt=dt, T=T_hours, epsilon=cfg["epsilon"],
                tol=cfg["tol"], max_iter=cfg["max_iter"],
                node_order=node_order,
            )

        # Baseline
        p_base = _build_perturbed_params(baseline, "beta", 1.0, n_nodes=4)
        m_base = _run_single(_make_solver(p_base), g, p_base, cfg)

        # +20% beta (stronger congestion)
        p_high = _build_perturbed_params(baseline, "beta", 1.2, n_nodes=4)
        m_high = _run_single(_make_solver(p_high), g, p_high, cfg)

        # Allow 2% numerical tolerance — effect is small at toy scale
        tol_rel = 0.02
        assert m_high["peak_density_node1"] <= m_base["peak_density_node1"] * (1 + tol_rel), (
            f"Expected peak_density_node1 to stay level or decrease with higher beta: "
            f"baseline={m_base['peak_density_node1']:.4f} "
            f"perturbed={m_high['peak_density_node1']:.4f}"
        )
