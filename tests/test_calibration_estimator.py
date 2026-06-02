"""Unit tests for src/calibration/estimator.py.

All tests use a 4-node K4 graph (1 transit + 3 attractions) to stay
consistent with the EXP-03/04 toy-graph convention.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from src.calibration.estimator import CalibrationEstimator, MFGParameters
from src.models.mfg_solver import MFGSolver


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_k4_graph(edge_length_m: float = 200.0) -> nx.DiGraph:
    """Fully-connected directed 4-node graph (K4)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    for i in range(4):
        for j in range(4):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


def _make_arrival_tensor(T_steps: int, N: int, dt: float, n_tourists: float) -> torch.Tensor:
    """Simple Gaussian arrival at node 0 only."""
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - 2.0) / 1.5) ** 2)
    g_norm = g_raw / (g_raw.sum() * dt + 1e-12)
    arrivals = torch.zeros(T_steps, N, dtype=torch.float32)
    arrivals[:, 0] = (n_tourists * g_norm).float()
    return arrivals


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_setup():
    """4-node K4, known true params, synthetic rho_obs, CalibrationEstimator ready."""
    dt = 5 / 60
    T_hours = 14.0
    T_steps = int(round(T_hours / dt))
    N = 4
    node_order = list(range(N))

    G = _make_k4_graph()
    alpha_star = [0.0, 2.0, 1.0, 0.5]
    beta_star = 0.01
    gamma_star = 0.0

    params_true = {
        "alpha": torch.tensor(alpha_star, dtype=torch.float32),
        "beta": torch.tensor(float(beta_star), dtype=torch.float32),
        "gamma": torch.tensor(float(gamma_star), dtype=torch.float32),
    }
    solver = MFGSolver(
        G=G, params=params_true, dt=dt, T=T_hours,
        epsilon=0.1, tol=1e-5, max_iter=200, node_order=node_order,
    )
    g = _make_arrival_tensor(T_steps, N, dt, n_tourists=1000.0)

    # Generate synthetic rho_obs from true params
    with torch.no_grad():
        rho_obs, _, _ = solver.fixed_point_iteration(g, damping=0.5)

    # MFGParameters: alpha_init non-zero for all nodes (transit gets 1e-6)
    alpha_init = torch.tensor([1e-6, 2.0, 1.0, 0.5], dtype=torch.float32)
    params = MFGParameters(N, alpha_init=alpha_init, beta_init=beta_star, gamma_init=1e-6)

    observations = {"rho_obs": rho_obs, "g": g}
    return solver, params, rho_obs, g, observations


# ---------------------------------------------------------------------------
# MFGParameters tests
# ---------------------------------------------------------------------------


class TestMFGParameters:
    def test_alpha_positive(self):
        p = MFGParameters(4, alpha_init=torch.tensor([1e-6, 2.0, 1.0, 0.5]))
        assert (p.alpha > 0).all()

    def test_beta_positive(self):
        p = MFGParameters(4, beta_init=0.01)
        assert float(p.beta.detach()) > 0

    def test_gamma_positive(self):
        p = MFGParameters(4, gamma_init=1e-4)
        assert float(p.gamma.detach()) > 0

    def test_as_dict_keys(self):
        p = MFGParameters(4)
        d = p.as_dict()
        assert set(d.keys()) == {"alpha", "beta", "gamma"}

    def test_parameters_are_learnable(self):
        p = MFGParameters(4)
        param_list = list(p.parameters())
        assert len(param_list) == 3  # log_alpha, log_beta, log_gamma


# ---------------------------------------------------------------------------
# CalibrationEstimator.loss() tests
# ---------------------------------------------------------------------------


class TestLoss:
    def test_loss_zero_when_pred_equals_obs(self, simple_setup):
        solver, params, rho_obs, g, observations = simple_setup
        est = CalibrationEstimator(solver, params, observations, lambda_reg=0.0)
        loss_val = est.loss(rho_obs, rho_obs)
        assert float(loss_val.item()) < 1e-6

    def test_loss_positive_when_different(self, simple_setup):
        solver, params, rho_obs, g, observations = simple_setup
        est = CalibrationEstimator(solver, params, observations, lambda_reg=0.0)
        rho_shifted = rho_obs + 1.0
        assert float(est.loss(rho_shifted, rho_obs).item()) > 0.0

    def test_loss_has_regularisation(self, simple_setup):
        solver, params, rho_obs, g, observations = simple_setup
        est = CalibrationEstimator(solver, params, observations, lambda_reg=1.0)
        # MSE term is 0 (pred=obs), but regularisation on alpha should be >0
        loss_val = est.loss(rho_obs, rho_obs)
        assert float(loss_val.item()) > 0.0

    def test_loss_differentiable(self, simple_setup):
        """Gradient flows through HJB+FP back to log_alpha and log_beta."""
        solver, params, rho_obs, g, observations = simple_setup
        est = CalibrationEstimator(solver, params, observations, lambda_reg=1e-4)

        # One-step forward with autograd
        solver.params = {
            "alpha": params.alpha,
            "beta": params.beta,
            "gamma": params.gamma,
        }
        rho_fp = torch.zeros_like(rho_obs)
        u = solver.solve_hjb_backward(rho_fp)
        rho_pred = solver.solve_fp_forward(u, g)

        loss_val = est.loss(rho_pred, rho_obs)
        loss_val.backward()

        assert params.log_alpha.grad is not None, "log_alpha has no gradient"
        assert params.log_beta.grad is not None, "log_beta has no gradient"
        assert not torch.isnan(params.log_alpha.grad).any()


# ---------------------------------------------------------------------------
# CalibrationEstimator.fit() tests
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_correct_keys(self, simple_setup):
        solver, params, rho_obs, g, observations = simple_setup
        est = CalibrationEstimator(solver, params, observations)
        result = est.fit(n_epochs=3, log_every=10)
        assert set(result.keys()) == {"loss_history", "final_params", "n_epochs"}

    def test_fit_loss_history_length(self, simple_setup):
        solver, params, rho_obs, g, observations = simple_setup
        est = CalibrationEstimator(solver, params, observations)
        n = 7
        result = est.fit(n_epochs=n, log_every=100)
        assert len(result["loss_history"]) == n

    def test_fit_reduces_loss_over_epochs(self):
        """Loss should decrease over epochs (uses tiny T=3 for speed)."""
        # Build a minimal solver (T=3 steps) to keep unit test fast.
        dt = 5 / 60
        T_hours = 3 * dt  # exactly 3 time steps
        N = 4
        node_order = list(range(N))

        G = _make_k4_graph()
        alpha_star = [0.0, 2.0, 1.0, 0.5]
        beta_star = 0.01

        params_true = {
            "alpha": torch.tensor(alpha_star, dtype=torch.float32),
            "beta": torch.tensor(float(beta_star), dtype=torch.float32),
            "gamma": torch.tensor(0.0, dtype=torch.float32),
        }
        solver = MFGSolver(
            G=G, params=params_true, dt=dt, T=T_hours,
            epsilon=0.1, tol=1e-5, max_iter=20, node_order=node_order,
        )
        g = _make_arrival_tensor(3, N, dt, n_tourists=1000.0)

        with torch.no_grad():
            rho_obs, _, _ = solver.fixed_point_iteration(g, damping=0.5)

        # Perturb params away from truth
        alpha_init = torch.tensor([1e-6, 2.0 * 1.6, 1.0 * 1.6, 0.5 * 1.6], dtype=torch.float32)
        params = MFGParameters(N, alpha_init=alpha_init, beta_init=beta_star * 1.6, gamma_init=1e-6)

        observations = {"rho_obs": rho_obs, "g": g}
        est = CalibrationEstimator(solver, params, observations, lambda_reg=1e-4)
        result = est.fit(n_epochs=30, lr=5e-3, log_every=100)

        first = result["loss_history"][0]
        last = result["loss_history"][-1]
        assert last < first, f"Loss did not decrease: {first:.4e} -> {last:.4e}"
