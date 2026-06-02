"""PyTorch-based parameter estimation for the MFG model.

Uses a differentiable forward simulator (MFGSolver) as the inner loop of an
Adam optimisation that minimises the discrepancy between predicted and observed
per-attraction visitor densities.

Unknown parameters (see docs/03_methodology.md §Phase 3):
- alpha_v : per-node attractiveness weight  (N_nodes-dim vector, positive)
- beta     : congestion cost coefficient     (scalar, positive)
- gamma    : walking cost coefficient        (scalar, positive)

Gradient strategy — "one-step consistency":
    Each epoch:
    1. Run fixed-point to convergence (torch.no_grad) to get stable rho_fp.
    2. Run ONE HJB+FP step with autograd active, injecting learnable params.
    3. Loss = MSE(rho_pred, rho_obs) + lambda_reg * mean(alpha^2)
    4. Backprop through step 2 to update log_alpha, log_beta, log_gamma.

All three parameters have well-defined gradients:
- alpha enters HJB as u[t] = alpha*dt - beta*rho*dt + cont → flows to FP softmax
- beta enters the same HJB term (multiplicative with rho)
- gamma enters Q_cont = -gamma*D + u[t+1] in both HJB and FP policy
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MFGParameters(nn.Module):
    """Learnable parameters for the MFG model.

    Wraps alpha (per-node), beta, and gamma as ``nn.Parameter`` tensors so
    that PyTorch autograd can compute gradients through the MFG forward pass.

    All parameters use log-space parameterisation to enforce positivity:
    ``alpha = exp(log_alpha)``, etc.

    Args:
        n_nodes: Number of graph nodes.
        alpha_init: Initial attractiveness weights (positive). If None,
            initialised uniformly to 1.0.
        beta_init: Initial congestion coefficient. Defaults to 1.0.
        gamma_init: Initial walking cost coefficient. Defaults to 0.1.
    """

    def __init__(
        self,
        n_nodes: int,
        alpha_init: torch.Tensor | None = None,
        beta_init: float = 1.0,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if alpha_init is None:
            alpha_init = torch.ones(n_nodes)
        alpha_init = alpha_init.float().clamp(min=1e-8)
        self.log_alpha = nn.Parameter(torch.log(alpha_init))
        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).clamp(min=1e-8).log())
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).clamp(min=1e-8).log())

    @property
    def alpha(self) -> torch.Tensor:
        """Positive attractiveness weights (N_nodes,)."""
        return torch.exp(self.log_alpha)

    @property
    def beta(self) -> torch.Tensor:
        """Positive congestion coefficient (scalar)."""
        return torch.exp(self.log_beta)

    @property
    def gamma(self) -> torch.Tensor:
        """Positive walking cost coefficient (scalar)."""
        return torch.exp(self.log_gamma)

    def as_dict(self) -> dict[str, Any]:
        """Return parameter values as a plain dict (for logging/saving).

        Returns:
            Dict with keys ``alpha``, ``beta``, ``gamma`` as Python floats/lists.
        """
        return {
            "alpha": self.alpha.detach().cpu().tolist(),
            "beta": float(self.beta.detach()),
            "gamma": float(self.gamma.detach()),
        }


class CalibrationEstimator:
    """End-to-end PyTorch calibration pipeline for the MFG model.

    Runs a gradient-based optimisation loop (Adam + exponential LR decay)
    that minimises the one-step consistency loss between the MFG predicted
    density and the observed visitor distribution.

    Args:
        solver: An ``MFGSolver`` instance. Its ``params`` dict will be
            overwritten each epoch with the current learnable parameters.
        params: ``MFGParameters`` module containing the learnable parameters.
        observations: Dict with keys:
            - ``rho_obs``: Observed density tensor (T_steps, N_nodes).
            - ``g``: Exogenous arrival rate tensor (T_steps, N_nodes).
        lambda_reg: L2 regularisation weight on alpha. Defaults to 1e-4.
    """

    def __init__(
        self,
        solver: Any,
        params: MFGParameters,
        observations: dict[str, torch.Tensor],
        lambda_reg: float = 1e-4,
    ) -> None:
        self.solver = solver
        self.params = params
        self.observations = observations
        self.lambda_reg = lambda_reg

    def loss(
        self,
        rho_pred: torch.Tensor,
        rho_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute calibration loss: MSE + L2 regularisation on alpha.

        Args:
            rho_pred: Predicted density (T_steps, N_nodes).
            rho_obs: Observed density (T_steps, N_nodes).

        Returns:
            Scalar loss tensor (differentiable w.r.t. self.params).
        """
        mse = F.mse_loss(rho_pred, rho_obs)
        reg = self.lambda_reg * (self.params.alpha ** 2).mean()
        return mse + reg

    def fit(
        self,
        n_epochs: int = 500,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        grad_clip: float = 1.0,
        log_every: int = 50,
        damping: float = 0.5,
    ) -> dict[str, Any]:
        """Run Adam optimisation loop.

        Each epoch:
        1. Run fixed-point to convergence (no_grad, with damping) for stable rho.
        2. Run one HJB+FP step with autograd active → rho_pred.
        3. Compute loss, backprop, clip gradients, update params.

        Args:
            n_epochs: Number of gradient steps. Defaults to 500.
            lr: Initial learning rate for Adam. Defaults to 1e-3.
            lr_decay: Multiplicative LR decay per epoch (ExponentialLR gamma).
                Defaults to 0.99.
            grad_clip: Gradient clipping L2 norm. Defaults to 1.0.
            log_every: Log loss every this many epochs. Defaults to 50.
            damping: Damping coefficient for fixed-point iteration (0, 1].
                Defaults to 0.5 for stability. Passed to
                ``solver.fixed_point_iteration(damping=damping)``.

        Returns:
            Dict with keys:
            - ``loss_history``: List[float] of loss per epoch.
            - ``final_params``: Final parameter values (from params.as_dict()).
            - ``n_epochs``: Number of epochs actually run.
        """
        optimizer = torch.optim.Adam(self.params.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

        g = self.observations["g"].float()
        rho_obs = self.observations["rho_obs"].float()

        loss_history: list[float] = []

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # ── Step 1: fixed-point to convergence (no grad) ─────────────────
            with torch.no_grad():
                self.solver.params = {
                    "alpha": self.params.alpha.detach(),
                    "beta": self.params.beta.detach(),
                    "gamma": self.params.gamma.detach(),
                }
                rho_fp, _, fp_info = self.solver.fixed_point_iteration(g, damping=damping)

            # ── Step 2: one HJB+FP step with autograd ────────────────────────
            self.solver.params = {
                "alpha": self.params.alpha,
                "beta": self.params.beta,
                "gamma": self.params.gamma,
            }
            u = self.solver.solve_hjb_backward(rho_fp)
            rho_pred = self.solver.solve_fp_forward(u, g)

            # ── Step 3: loss, backward, clip, step ───────────────────────────
            loss_val = self.loss(rho_pred, rho_obs)
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.params.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            loss_history.append(float(loss_val.item()))

            if (epoch + 1) % log_every == 0:
                alpha_str = "[" + ", ".join(
                    f"{x:.3f}" for x in self.params.alpha.detach().tolist()
                ) + "]"
                logger.info(
                    "Epoch %d/%d: loss=%.4e | alpha=%s | beta=%.5f | gamma=%.6f"
                    " | fp_iter=%d | fp_conv=%s",
                    epoch + 1, n_epochs,
                    loss_val.item(),
                    alpha_str,
                    float(self.params.beta.detach()),
                    float(self.params.gamma.detach()),
                    fp_info["n_iter"],
                    fp_info["converged"],
                )

        return {
            "loss_history": loss_history,
            "final_params": self.params.as_dict(),
            "n_epochs": n_epochs,
        }
