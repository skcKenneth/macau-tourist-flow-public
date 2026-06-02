"""Competing baseline models for the attraction-share prediction task (EXP-10).

These give an apples-to-apples comparison against the MFG on the **same**
real-DSEC held-out split (Goal C of the research-hardening program). All models
predict the normalized spatial distribution of visitors across the heritage
attractions from the per-source arrival weights; none has congestion feedback.

Models:
- ``GravityModel``: classic spatial-interaction model. The accessibility of
  attraction ``v`` is ``A_v = sum_s w_s * exp(-theta * d_sv)`` (distance decay from
  each source with weight ``w_s``); predicted share ``∝ mass_v * A_v``.
- ``MultinomialLogitModel``: static random-utility discrete choice. From source
  ``s`` a tourist picks attraction ``v`` with probability
  ``softmax_v(alpha_v - gamma * d_sv)``; the population share mixes these by ``w_s``.
  No congestion, no temporal dynamics — the natural "MFG minus the mean-field
  coupling" comparison.

Both are differentiable ``nn.Module``s fit by Adam to the monthly spatial shares,
mirroring the MFG calibration protocol. Distances are passed in metres and scaled
to km internally so decay parameters are O(1).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GravityModel(nn.Module):
    """Distance-decay spatial-interaction model for attraction shares.

    Args:
        n_attractions: Number of attraction nodes (output dimension).
        dist_source_attraction: Tensor (n_sources, n_attractions) of walking
            distances in metres from each source node to each attraction.
        theta_init: Initial distance-decay coefficient (per km).
    """

    def __init__(
        self,
        n_attractions: int,
        dist_source_attraction: torch.Tensor,
        theta_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("D_km", dist_source_attraction.float() / 1000.0)
        self.log_mass = nn.Parameter(torch.zeros(n_attractions))
        self.log_theta = nn.Parameter(torch.tensor(float(theta_init)).clamp(min=1e-6).log())

    def predict(self, source_weights: torch.Tensor) -> torch.Tensor:
        """Predicted normalized attraction shares for given source weights.

        Args:
            source_weights: Tensor (n_sources,) of (not necessarily normalized)
                arrival weights per source node.

        Returns:
            Tensor (n_attractions,) summing to 1.
        """
        w = source_weights.float()
        w = w / (w.sum() + 1e-12)
        theta = torch.exp(self.log_theta)
        access = (w.unsqueeze(1) * torch.exp(-theta * self.D_km)).sum(dim=0)  # (n_att,)
        raw = torch.exp(self.log_mass) * access
        return raw / (raw.sum() + 1e-12)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mass": torch.exp(self.log_mass).detach().tolist(),
            "theta": float(torch.exp(self.log_theta).detach()),
        }


class MultinomialLogitModel(nn.Module):
    """Static multinomial-logit discrete-choice model (no congestion feedback).

    Args:
        n_attractions: Number of attraction nodes (output dimension).
        dist_source_attraction: Tensor (n_sources, n_attractions) of walking
            distances in metres from each source node to each attraction.
        gamma_init: Initial walking-cost coefficient (per km).
    """

    def __init__(
        self,
        n_attractions: int,
        dist_source_attraction: torch.Tensor,
        gamma_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("D_km", dist_source_attraction.float() / 1000.0)
        self.alpha = nn.Parameter(torch.zeros(n_attractions))
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).clamp(min=1e-6).log())

    def predict(self, source_weights: torch.Tensor) -> torch.Tensor:
        """Population attraction shares: source-mixed multinomial-logit choices."""
        w = source_weights.float()
        w = w / (w.sum() + 1e-12)
        gamma = torch.exp(self.log_gamma)
        utility = self.alpha.unsqueeze(0) - gamma * self.D_km  # (n_sources, n_att)
        choice = torch.softmax(utility, dim=1)                 # (n_sources, n_att)
        shares = (w.unsqueeze(1) * choice).sum(dim=0)          # (n_att,)
        return shares / (shares.sum() + 1e-12)

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha.detach().tolist(),
            "gamma": float(torch.exp(self.log_gamma).detach()),
        }


def fit_static_model(
    model: nn.Module,
    train_items: list[dict[str, torch.Tensor]],
    n_epochs: int = 500,
    lr: float = 5e-2,
    lr_decay: float = 0.999,
) -> list[float]:
    """Fit a static share model (gravity / MNL) by Adam on monthly spatial MSE.

    Args:
        model: A model exposing ``predict(source_weights) -> shares``.
        train_items: List of dicts with ``source_weights`` and ``obs`` tensors.
        n_epochs: Number of gradient steps.
        lr: Initial Adam learning rate.
        lr_decay: ExponentialLR multiplicative decay per step.

    Returns:
        Loss history (mean per-month MSE per epoch).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    history: list[float] = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for item in train_items:
            pred = model.predict(item["source_weights"])
            loss = loss + F.mse_loss(pred, item["obs"])
        loss = loss / max(len(train_items), 1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(float(loss.item()))
    return history


def evaluate_static_model(
    model: nn.Module,
    val_items: list[dict[str, torch.Tensor]],
) -> float:
    """Mean held-out spatial MAE of a static share model."""
    maes = []
    with torch.no_grad():
        for item in val_items:
            pred = model.predict(item["source_weights"])
            maes.append(float((pred - item["obs"]).abs().mean().item()))
    return sum(maes) / max(len(maes), 1)
