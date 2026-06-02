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
Metrics 6, 7 deferred to Week 7 (depend on trajectory representation TBD).
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
) -> float:
    """Mean number of distinct heritage attractions visited per tourist trajectory.

    Args:
        flow_matrix: Tourist flow tensor (T_steps, N_nodes) or cumulative
            visit count per node — exact format TBD in Week 7.
        attraction_mask: Boolean tensor (N_nodes,), True for heritage
            attraction nodes (excludes transit nodes).

    Returns:
        Mean number of distinct attractions visited per tourist.

    Raises:
        NotImplementedError: Until Week 7 implementation.
    """
    # TODO (Week 7): implement after trajectory representation is finalised.
    raise NotImplementedError("mean_attractions_visited — implement in Week 7.")


def mean_walking_distance(
    flow_matrix: torch.Tensor,
    edge_lengths: torch.Tensor,
) -> float:
    """Mean total walking distance per tourist (metres).

    Args:
        flow_matrix: Tourist flow tensor — exact format TBD in Week 7.
        edge_lengths: Edge length matrix (N_nodes, N_nodes) in metres.

    Returns:
        Mean walking distance per tourist in metres.

    Raises:
        NotImplementedError: Until Week 7 implementation.
    """
    # TODO (Week 7): implement after trajectory representation is finalised.
    raise NotImplementedError("mean_walking_distance — implement in Week 7.")


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
