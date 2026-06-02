"""End-to-end pipeline smoke test (no external data, no OSM download).

Exercises the full modelling chain on a small synthetic star graph so a single
fast test catches breakage that unit tests (which cover modules in isolation)
would miss:

    MFGSolver fixed point
        -> CalibrationEstimator.fit            (calibration)
        -> RoutingOptimizer.optimise           (EXP-08 intervention)
        -> EntranceMeteringOptimizer.sweep     (EXP-07 intervention)
        -> transition_flows + flow metrics     (cross-cutting metrics 6 & 7)

Marked ``integration`` so it can be opted into / out of in CI; it still runs by
default with ``pytest`` and is excluded only by ``-m "not integration"``.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from src.calibration.estimator import CalibrationEstimator, MFGParameters
from src.evaluation.metrics import (
    calibration_mae,
    mean_attractions_visited,
    mean_walking_distance,
)
from src.models.mfg_solver import MFGSolver
from src.optimization.interventions import (
    EntranceMeteringOptimizer,
    RoutingOptimizer,
)

pytestmark = pytest.mark.integration

N_NODES = 4
ATTRACTION_MASK = torch.tensor([False, True, True, True])  # node 0 = transit hub


def _star_graph(edge_length_m: float = 200.0) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(N_NODES))
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


def _make_solver(alpha, beta=0.01, gamma=0.001) -> MFGSolver:
    return MFGSolver(
        G=_star_graph(),
        params={
            "alpha": torch.as_tensor(alpha, dtype=torch.float32),
            "beta": torch.tensor(float(beta), dtype=torch.float32),
            "gamma": torch.tensor(float(gamma), dtype=torch.float32),
        },
        dt=0.25,
        T=6.0,
        epsilon=0.1,
        tol=1e-3,
        max_iter=300,
        node_order=list(range(N_NODES)),
    )


def _arrivals(T_steps: int, dt: float, n_tourists: float = 40.0) -> torch.Tensor:
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    shape = torch.exp(-0.5 * ((t_vec - 2.0) / 1.0) ** 2)
    shape = shape / (shape.sum() * dt + 1e-12)
    g = torch.zeros(T_steps, N_NODES, dtype=torch.float32)
    g[:, 0] = (n_tourists * shape).float()
    return g


def test_pipeline_calibration_then_interventions_then_metrics():
    torch.manual_seed(0)

    # ── 1. Generate synthetic "observed" data from a known true model ─────────
    true_alpha = torch.tensor([0.0, 2.0, 1.0, 0.5], dtype=torch.float32)
    true_solver = _make_solver(true_alpha)
    n_tourists = 40.0
    g = _arrivals(true_solver.T_steps, true_solver.dt, n_tourists)
    with torch.no_grad():
        rho_obs, _, base_info = true_solver.fixed_point_iteration(g, damping=0.5)
    assert base_info["converged"], "baseline equilibrium must converge"
    assert torch.isfinite(rho_obs).all()

    # ── 2. Calibrate from a deliberately wrong init; loss must improve ────────
    fit_solver = _make_solver(torch.ones(N_NODES))
    params = MFGParameters(
        n_nodes=N_NODES,
        alpha_init=torch.ones(N_NODES),
        beta_init=0.05,
        gamma_init=0.001,
    )
    estimator = CalibrationEstimator(
        solver=fit_solver,
        params=params,
        observations={"rho_obs": rho_obs, "g": g},
    )
    result = estimator.fit(n_epochs=15, lr=0.05, log_every=100, damping=0.5)
    assert len(result["loss_history"]) == 15
    assert result["loss_history"][-1] < result["loss_history"][0], (
        "calibration should reduce the loss"
    )
    fitted = result["final_params"]
    assert all(a >= 0.0 for a in fitted["alpha"])
    assert fitted["beta"] > 0.0 and fitted["gamma"] > 0.0

    # ── 3. Build a calibrated solver and confirm a sane spatial fit ───────────
    cal_solver = _make_solver(fitted["alpha"], fitted["beta"], fitted["gamma"])
    with torch.no_grad():
        rho_cal, u_cal, cal_info = cal_solver.fixed_point_iteration(g, damping=0.5)
    assert cal_info["converged"]
    share_pred = rho_cal[:, 1:].sum(0) / (rho_cal[:, 1:].sum() + 1e-12)
    share_obs = rho_obs[:, 1:].sum(0) / (rho_obs[:, 1:].sum() + 1e-12)
    mae = calibration_mae(share_pred, share_obs)
    assert mae < 0.4, f"calibrated attraction-share MAE unexpectedly high: {mae:.3f}"

    # ── 4. Routing intervention (EXP-08) on the calibrated solver ─────────────
    routing = RoutingOptimizer(
        solver=cal_solver,
        g=g,
        bottleneck_idx=[1, 2, 3],
        attraction_count=3,
        damping=0.5,
    )
    routing_out = routing.optimise(n_steps=5, lr=0.05, log_every=100)
    assert routing_out["eta"].shape == (N_NODES, N_NODES)
    assert len(routing_out["loss_history"]) == 5
    assert routing_out["peak_baseline"] > 0.0
    # The solver is left in a clean state (routing_bonus reset to zeros).
    assert torch.count_nonzero(cal_solver.routing_bonus) == 0

    # ── 5. Entrance metering intervention (EXP-07) on the calibrated solver ───
    metering = EntranceMeteringOptimizer(
        solver=cal_solver,
        source_col=0,
        bottleneck_idx=1,
        attraction_count=3,
        damping=0.5,
    )
    records = metering.pareto_frontier(g, n_points=4)
    assert len(records) == 4
    for rec in records:
        assert {"r_star", "peak_reduction_pct", "visit_reduction_pct"} <= set(rec)
    # An aggressive cap should not increase the peak relative to the loosest cap.
    assert records[0]["peak_density_ruins"] <= records[-1]["peak_density_ruins"] + 1e-3

    # ── 6. Cross-cutting flow metrics (metrics 6 & 7) ─────────────────────────
    flows = cal_solver.transition_flows(rho_cal, u_cal)
    assert flows.shape == (cal_solver.T_steps - 1, N_NODES, N_NODES)
    visited = mean_attractions_visited(flows, ATTRACTION_MASK, n_tourists)
    walked = mean_walking_distance(flows, cal_solver.D, n_tourists)
    assert visited > 0.0
    assert walked > 0.0
