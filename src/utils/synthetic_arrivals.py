"""Synthetic arrival profiles for EXP-02 (and future experiments until real data is available).

These functions stand in for the DSEC/MGTO data pipeline until Week 2 data
acquisition is complete. When real data arrives, replace calls to
``build_arrival_tensor`` and ``build_observed_distribution`` with outputs
from ``src.utils.data_loader``.

Temporal profile: Gaussian peak centred at ``peak_time_hours`` (hours after
08:00 start) with standard deviation ``sigma_hours``. Only transit nodes
receive exogenous arrivals; attraction nodes are always zero.

Observed distribution proxy: ``annual_visitors_est`` from ``attractions.py``,
restricted to heritage attraction nodes and normalized to sum to 1.
"""

from __future__ import annotations

import logging

import torch

from src.utils.attractions import ATTRACTION_IDS, ATTRACTION_NODES, NODE_INDEX, TRANSIT_IDS

logger = logging.getLogger(__name__)


def build_arrival_tensor(
    node_order: list[str],
    T_steps: int,
    dt: float,
    n_tourists: int,
    peak_time_hours: float,
    sigma_hours: float,
    transit_shares: dict[str, float],
) -> torch.Tensor:
    """Build a (T_steps, N_nodes) tensor of synthetic arrival rates.

    Arrivals are non-zero only at transit nodes. The temporal profile is a
    Gaussian centred at ``peak_time_hours`` (hours after simulation start)
    with standard deviation ``sigma_hours``, scaled so that integrating over
    the day yields ``n_tourists * share_v`` total arrivals at transit node v.

    Args:
        node_order: List of node_id strings in canonical order (length N_nodes).
        T_steps: Number of simulation time steps.
        dt: Time step in hours (e.g. 5/60 for 5-minute steps).
        n_tourists: Total tourists injected across the whole day (all transit).
        peak_time_hours: Hours after simulation start for the Gaussian peak.
        sigma_hours: Standard deviation of the Gaussian in hours.
        transit_shares: Dict mapping transit node_id → fraction of n_tourists.
            Values must sum to 1.0 (checked within 1e-3 tolerance).

    Returns:
        Float32 tensor of shape (T_steps, N_nodes) giving exogenous arrival
        rate in tourists/hour at each node and time step. Non-transit entries
        are always exactly zero.

    Raises:
        ValueError: If transit_shares do not sum to ~1.0, or if a key in
            transit_shares is not found in node_order.
    """
    total_share = sum(transit_shares.values())
    if abs(total_share - 1.0) > 1e-3:
        raise ValueError(
            f"transit_shares must sum to 1.0, got {total_share:.6f}"
        )

    for node_id in transit_shares:
        if node_id not in node_order:
            raise ValueError(
                f"transit_shares key '{node_id}' not found in node_order"
            )
        if node_id not in TRANSIT_IDS:
            logger.warning(
                "'%s' in transit_shares is not a registered transit node", node_id
            )

    N = len(node_order)
    arrivals = torch.zeros(T_steps, N, dtype=torch.float32)

    # Gaussian temporal profile (unnormalized)
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - peak_time_hours) / sigma_hours) ** 2)

    # Normalize so that sum(g_norm) * dt ≈ 1  →  integrating gives 1 "tourist-unit"
    g_norm = g_raw / (g_raw.sum() * dt)

    node_to_idx = {nid: i for i, nid in enumerate(node_order)}
    for node_id, share in transit_shares.items():
        col = node_to_idx[node_id]
        arrivals[:, col] = n_tourists * share * g_norm

    logger.info(
        "Arrival tensor built: shape=%s, total_injected≈%.0f tourists",
        list(arrivals.shape),
        float((arrivals.sum() * dt).item()),
    )
    return arrivals


def build_observed_distribution(node_order: list[str]) -> torch.Tensor:
    """Build the normalized observed visitor distribution for attraction nodes.

    Uses ``annual_visitors_est`` from AttractionNode metadata as a proxy for
    long-run average visitor proportions. Transit nodes are excluded because
    their visitor counts reflect pass-through traffic, not attraction visits.

    Args:
        node_order: List of node_id strings in canonical order (length N_nodes).
            Used only to determine which indices are attraction nodes.

    Returns:
        Float32 tensor of shape (N_attraction_nodes,) giving the normalized
        fraction of visitors at each attraction node (sums to 1.0). Ordering
        follows the attraction-only subset of node_order.

    Note:
        This is a proxy distribution. Replace with MGTO attraction-level data
        once available (see ``src.utils.data_loader.load_attraction_counts``).
    """
    node_by_id = {n.node_id: n for n in ATTRACTION_NODES}

    # Collect attraction nodes in the order they appear in node_order
    att_counts: list[float] = []
    att_ids_in_order: list[str] = []
    for nid in node_order:
        if nid in ATTRACTION_IDS:
            att_counts.append(float(node_by_id[nid].annual_visitors_est))
            att_ids_in_order.append(nid)

    if not att_counts:
        raise ValueError("No attraction nodes found in node_order")

    counts = torch.tensor(att_counts, dtype=torch.float32)
    dist = counts / counts.sum()

    logger.info(
        "Observed distribution built from annual_visitors_est (%d attractions). "
        "Top node: %s (%.1f%%)",
        len(att_ids_in_order),
        att_ids_in_order[int(dist.argmax().item())],
        float(dist.max().item()) * 100,
    )
    return dist
