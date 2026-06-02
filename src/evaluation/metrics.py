"""Evaluation metrics for the MFG model and intervention experiments.

All functions accept PyTorch tensors as primary inputs (for differentiability
where needed) but return plain Python floats or numpy arrays for reporting.

Cross-cutting metrics logged for every model variant (from docs/05_experiment_plan.md):
1. Calibration MAE (per attraction, normalised density)
2. Wall-clock runtime for forward solve
3. Number of fixed-point iterations to convergence
4. Peak density at top-3 attractions
5. Gini coefficient of density distribution
6. Mean attractions visited per tourist
7. Mean total walking distance per tourist

Metrics 1, 4, 5 implemented in Week 2 (pulled forward for EXP-02).
Metrics 6, 7 are mean-field proxies computed from the per-step movement flows
(``MFGSolver.transition_flows``): attraction *entries* per tourist and walking
distance per tourist. They do not require individual trajectories.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def peak_density(rho: torch.Tensor, node_idx: int) -> float:
    """Maximum density at a single node across all time steps.

    Args:
        rho: Density tensor of shape (T_steps, N_nodes).
        node_idx: Column index of the target node.

    Returns:
        Scalar peak density value (float).
    """
    return float(rho[:, node_idx].max().item())


def calibration_mae(
    rho_pred: torch.Tensor,
    rho_obs: torch.Tensor,
    per_node: bool = False,
) -> float | np.ndarray:
    """Mean absolute error between predicted and observed normalised density.

    Both inputs should be 1-D tensors of shape (N_nodes,) representing the
    time-averaged (equilibrium) fraction at each node, normalized to sum to 1.
    Alternatively both can be 2-D (T_steps, N_nodes) for per-step MAE.

    Args:
        rho_pred: Predicted density (N_nodes,) or (T_steps, N_nodes).
        rho_obs: Observed density, same shape as rho_pred.
        per_node: If True, return per-node MAE array instead of the scalar mean.

    Returns:
        Overall MAE (float) or per-node MAE (np.ndarray of shape N_nodes).
    """
    if rho_pred.shape != rho_obs.shape:
        raise ValueError(
            f"Shape mismatch: rho_pred {rho_pred.shape} != rho_obs {rho_obs.shape}"
        )

    # Normalization check (warn only — caller is responsible for normalizing)
    def _check_normalized(t: torch.Tensor, name: str) -> None:
        if t.dim() == 1:
            s = float(t.sum().item())
            if abs(s - 1.0) > 0.01:
                logger.warning("%s sums to %.4f, expected ~1.0", name, s)
        else:
            row_sums = t.sum(dim=-1)
            bad = (row_sums - 1.0).abs().max().item()
            if bad > 0.01:
                logger.warning("%s max row-sum deviation from 1: %.4f", name, bad)

    _check_normalized(rho_pred, "rho_pred")
    _check_normalized(rho_obs, "rho_obs")

    diff = (rho_pred - rho_obs).abs()

    if per_node:
        if diff.dim() == 2:
            return diff.mean(dim=0).numpy()
        return diff.numpy()

    return float(diff.mean().item())


def gini_coefficient(rho: torch.Tensor, t_idx: int) -> float:
    """Gini coefficient of the density distribution at a single time step.

    Higher Gini = more spatially concentrated tourist distribution.
    A Gini of 0 means perfectly uniform; 1 means all tourists at one node.

    Args:
        rho: Density tensor of shape (T_steps, N_nodes).
        t_idx: Time step index to evaluate.

    Returns:
        Gini coefficient in [0, 1]. Returns 0.0 if all densities are zero.
    """
    x = rho[t_idx].float()
    total = float(x.sum().item())
    if total < 1e-12:
        return 0.0

    # Normalize to fractions
    x = x / total

    # Sort ascending for the standard linear-time formula:
    # G = 1 - (2 / n) * sum_i((n + 1 - i) * x_sorted_i) / sum(x)
    # where i is 1-indexed and x is already normalized (sum=1).
    x_sorted, _ = x.sort()
    n = x_sorted.shape[0]
    # 1-indexed ranks: n, n-1, ..., 1  (highest rank → smallest value)
    ranks = torch.arange(n, 0, -1, dtype=torch.float32)
    numerator = float((ranks * x_sorted).sum().item())
    # G = 1 - 2 * numerator / n  (since sum(x_sorted) = 1 after normalization)
    return float(1.0 - 2.0 * numerator / n)


def gini_timeseries(rho: torch.Tensor) -> np.ndarray:
    """Compute Gini coefficient at every time step.

    Args:
        rho: Density tensor of shape (T_steps, N_nodes).

    Returns:
        Array of shape (T_steps,) with Gini coefficient at each time step.
    """
    T = rho.shape[0]
    return np.array([gini_coefficient(rho, t) for t in range(T)])


def mean_attractions_visited(
    flow_matrix: torch.Tensor,
    attraction_mask: torch.Tensor,
    n_tourists: float,
) -> float:
    """Mean number of heritage-attraction entries per tourist (mean-field proxy).

    A pure density trajectory cannot recover *distinct* attractions per tourist,
    but the per-step movement flows can. This counts every move into a heritage
    attraction node (excluding "stay" self-loops) and normalises by the number of
    tourists that entered the system, giving the average count of attraction
    arrivals per tourist over the day. Re-entries are counted (a tourist who
    leaves and returns registers two entries), so it is an upper-leaning proxy for
    "distinct attractions visited".

    Args:
        flow_matrix: Movement flows (T_steps - 1, N_nodes, N_nodes) from
            ``MFGSolver.transition_flows``; ``flow[t, v, w]`` is the number of
            tourists moving v->w at step t.
        attraction_mask: Boolean tensor (N_nodes,), True for heritage attraction
            nodes (excludes transit nodes).
        n_tourists: Total tourists that entered the system (sum of arrivals).

    Returns:
        Mean attraction entries per tourist (float).
    """
    if flow_matrix.dim() != 3 or flow_matrix.shape[-1] != flow_matrix.shape[-2]:
        raise ValueError(
            f"flow_matrix must be (T-1, N, N); got {tuple(flow_matrix.shape)}"
        )
    n = flow_matrix.shape[-1]
    mask = attraction_mask.bool()
    if mask.shape != (n,):
        raise ValueError(f"attraction_mask must be ({n},); got {tuple(mask.shape)}")

    # Drop self-loops (stays are not "visits") and sum entries into each node.
    moves = flow_matrix * (1.0 - torch.eye(n, dtype=flow_matrix.dtype))
    inflow_per_node = moves.sum(dim=(0, 1))  # (N,) total entries into each w
    entries = float(inflow_per_node[mask].sum().item())
    return entries / (float(n_tourists) + 1e-12)


def mean_walking_distance(
    flow_matrix: torch.Tensor,
    edge_lengths: torch.Tensor,
    n_tourists: float,
) -> float:
    """Mean total walking distance per tourist (metres), from movement flows.

    Weights each edge traversal by its walking length and normalises by the
    number of tourists. Self-loops contribute zero (``D[v, v] = 0``) and
    unreachable edges (``D = inf``) carry ~zero flow; both are masked to avoid
    ``inf * 0`` NaNs.

    Args:
        flow_matrix: Movement flows (T_steps - 1, N_nodes, N_nodes) from
            ``MFGSolver.transition_flows``.
        edge_lengths: Walking-distance matrix (N_nodes, N_nodes) in metres
            (``MFGSolver.D``); may contain ``inf`` for absent edges.
        n_tourists: Total tourists that entered the system (sum of arrivals).

    Returns:
        Mean walking distance per tourist in metres (float).
    """
    if flow_matrix.dim() != 3:
        raise ValueError(
            f"flow_matrix must be (T-1, N, N); got {tuple(flow_matrix.shape)}"
        )
    d = torch.where(
        torch.isinf(edge_lengths), torch.zeros_like(edge_lengths), edge_lengths
    )
    total_metres = float((flow_matrix * d).sum().item())
    return total_metres / (float(n_tourists) + 1e-12)


def top_k_peak_densities(
    rho: torch.Tensor,
    node_ids: list[str],
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return the top-k nodes by peak density.

    Args:
        rho: Density tensor (T_steps, N_nodes).
        node_ids: List of node ID strings (length N_nodes, canonical order).
        k: Number of top nodes to return. Defaults to 3.

    Returns:
        List of (node_id, peak_density) tuples, sorted descending by peak density.
    """
    peaks = rho.max(dim=0).values  # (N_nodes,)
    k = min(k, len(node_ids))
    top = peaks.topk(k)
    return [(node_ids[int(i)], float(peaks[int(i)].item())) for i in top.indices]
