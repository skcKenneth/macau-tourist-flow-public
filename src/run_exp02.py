"""EXP-02: Random-Walk Baseline Simulation.

Hypothesis:
    A random walk (distance-weighted transitions, no congestion/optimization)
    does NOT match observed visitor concentration at popular attractions —
    quantifying the motivation for the MFG formulation.

Success criterion:
    MAE > 0.1 in normalized density between the random-walk equilibrium
    distribution and the proxy observed distribution derived from
    annual_visitors_est in attractions.py.

Usage::

    python -m src.run_exp02
    python -m src.run_exp02 --config configs/exp02_baseline.yaml

Outputs (in experiments/YYYYMMDD_EXP-02_baseline_random_walk/):
    density_heatmap.png / .pdf    — (T_steps, N_nodes) density time series
    comparison_bar.png / .pdf     — simulated vs observed per attraction
    metrics.csv                   — per-attraction MAE, Gini, peak densities
    summary.txt                   — PASS/FAIL verdict + key numbers
    config.yaml                   — parameter snapshot

Note:
    DSEC/MGTO data not yet available (Week 2 goal). This run uses:
    - Synthetic Gaussian arrivals at transit nodes (src.utils.synthetic_arrivals)
    - Observed distribution: annual_visitors_est from attractions.py (proxy)
    When real data is available, replace via data_loader and re-run.

See docs/05_experiment_plan.md §EXP-02 for full specification.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subgraph construction
# ---------------------------------------------------------------------------


def _build_subgraph(G_osm, snap_map: dict[str, int], node_order: list[str]):
    """Build the 13-node fully-connected directed subgraph for simulation.

    Uses all-pairs shortest paths on the undirected OSM graph to compute
    realistic walking distances between attraction/transit nodes. Returns a
    DiGraph with integer node IDs 0..12 (matching NODE_INDEX canonical order)
    and directed edges weighted by shortest walking distance in metres.

    Args:
        G_osm: Full OSM MultiDiGraph (981 nodes).
        snap_map: Dict mapping node_id → OSM integer node ID.
        node_order: List of 13 node_id strings in canonical NODE_INDEX order.

    Returns:
        nx.DiGraph with 13 nodes (integer IDs 0..12) and 13×12 directed edges,
        each with attribute ``length`` (metres).

    Raises:
        RuntimeError: If any pair of nodes has no path in the OSM graph.
    """
    import networkx as nx

    G_undir = G_osm.to_undirected()

    G_sub = nx.DiGraph()
    G_sub.add_nodes_from(range(len(node_order)))

    logger.info("Computing all-pairs shortest paths for %d nodes...", len(node_order))
    t0 = time.perf_counter()

    for i, src_id in enumerate(node_order):
        osm_src = snap_map[src_id]
        for j, dst_id in enumerate(node_order):
            if i == j:
                continue
            osm_dst = snap_map[dst_id]
            try:
                d = nx.shortest_path_length(
                    G_undir, source=osm_src, target=osm_dst, weight="length"
                )
            except nx.NetworkXNoPath:
                raise RuntimeError(
                    f"No walking path between '{src_id}' (OSM {osm_src}) "
                    f"and '{dst_id}' (OSM {osm_dst}). "
                    "Check bbox coverage or graph connectivity."
                ) from None
            G_sub.add_edge(i, j, length=float(d))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Subgraph built in %.1f s: %d nodes, %d edges",
        elapsed,
        G_sub.number_of_nodes(),
        G_sub.number_of_edges(),
    )
    return G_sub


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run(cfg: dict) -> bool:
    """Execute EXP-02 and return True if the hypothesis is confirmed (MAE > 0.1).

    Args:
        cfg: Parsed configuration dictionary (matches configs/exp02_baseline.yaml).

    Returns:
        True if MAE > 0.1 (random walk fails to match observed — MFG justified).
    """
    import matplotlib
    matplotlib.use("Agg")

    from src.utils import attractions, graph_loader, io
    from src.utils.synthetic_arrivals import (
        build_arrival_tensor,
        build_observed_distribution,
    )
    from src.models.baseline_random import RandomWalkBaseline
    from src.evaluation.metrics import (
        calibration_mae,
        gini_timeseries,
        peak_density,
        top_k_peak_densities,
    )

    exp_cfg = cfg["experiment"]
    graph_cfg = cfg["graph"]
    snap_cfg = cfg["snapping"]
    sim_cfg = cfg["simulation"]
    arr_cfg = cfg["arrivals"]
    out_cfg = cfg["output"]

    # ── Experiment directory ─────────────────────────────────────────────────
    outdir = io.make_experiment_dir(
        base=Path(out_cfg["base_dir"]),
        name=f"{exp_cfg['id']}_{exp_cfg['name']}",
    )
    logger.info("Output directory: %s", outdir)

    # ── Seeds + metadata ─────────────────────────────────────────────────────
    seed = exp_cfg["seed"]
    io.set_all_seeds(seed)
    git_hash = io.get_git_hash()
    logger.info("Git hash: %s  Seed: %d", git_hash, seed)

    cfg_to_save = dict(cfg)
    cfg_to_save["_meta"] = {"git_hash": git_hash}
    io.save_config(cfg_to_save, outdir)

    # ── Load OSM graph ───────────────────────────────────────────────────────
    cache_path = Path(graph_cfg.get("cache_path", "data/raw/osm"))
    G_osm = graph_loader.load_macau_graph(
        bbox={
            "north": 22.220, "south": 22.183,
            "east": 113.550, "west": 113.525,
        },
        network_type="walk",
        simplify=True,
        retain_all=False,
        consolidate_tolerance_m=graph_cfg.get("consolidate_tolerance_m", 15.0),
        walk_speed_kmh=graph_cfg.get("walk_speed_kmh", 5.0),
        cache_path=cache_path,
    )
    logger.info(
        "OSM graph: %d nodes, %d edges", G_osm.number_of_nodes(), G_osm.number_of_edges()
    )

    # ── Snap attractions ─────────────────────────────────────────────────────
    snap_map = attractions.snap_to_graph(
        G_osm,
        max_dist_attraction_m=snap_cfg.get("max_dist_attraction_m", 100.0),
        max_dist_transit_m=snap_cfg.get("max_dist_transit_m", 300.0),
    )
    graph_loader.add_attraction_attributes(G_osm, snap_map)

    # ── Build 13-node simulation subgraph ────────────────────────────────────
    node_order = [n.node_id for n in attractions.ATTRACTION_NODES]  # canonical order
    G_sub = _build_subgraph(G_osm, snap_map, node_order)

    # ── Simulation parameters ────────────────────────────────────────────────
    dt = sim_cfg["dt_hours"]
    T_hours = sim_cfg["T_hours"]
    T_steps = round(T_hours / dt)
    n_tourists = sim_cfg["n_tourists"]
    exit_rate = sim_cfg["exit_rate_per_step"]

    # ── Synthetic arrivals ───────────────────────────────────────────────────
    arrival_rates = build_arrival_tensor(
        node_order=node_order,
        T_steps=T_steps,
        dt=dt,
        n_tourists=n_tourists,
        peak_time_hours=arr_cfg["peak_time_hours"],
        sigma_hours=arr_cfg["sigma_hours"],
        transit_shares=arr_cfg["transit_shares"],
    )

    # ── Observed distribution (proxy) ────────────────────────────────────────
    obs_dist = build_observed_distribution(node_order)  # (N_att,)

    # ── Run simulation ───────────────────────────────────────────────────────
    logger.info("Running random-walk simulation (%d steps, %d nodes)...", T_steps, 13)
    baseline = RandomWalkBaseline(
        G=G_sub,
        arrival_rates=arrival_rates,
        dt=dt,
        node_order=list(range(13)),  # integer node IDs 0..12
        exit_rate=exit_rate,
    )
    t_sim_start = time.perf_counter()
    rho_raw = baseline.simulate(seed=seed)   # (T, 13) raw counts
    sim_elapsed = time.perf_counter() - t_sim_start
    logger.info("Simulation done in %.3f s", sim_elapsed)

    # ── Normalize density ────────────────────────────────────────────────────
    row_sums = rho_raw.sum(dim=1, keepdim=True).clamp(min=1e-9)
    rho_norm = rho_raw / row_sums  # (T, 13) fraction of in-system tourists

    # ── Cumulative density: total tourist-time at each node ─────────────────
    # Comparable to annual_visitors_est (which also measures cumulative presence).
    # More robust than an end-of-day snapshot which reflects only the tail of
    # the distribution after most tourists have exited.
    att_indices = [i for i, nid in enumerate(node_order)
                   if nid in attractions.ATTRACTION_IDS]
    cumulative = rho_raw.sum(dim=0)            # (N,) tourist-steps per node
    att_cumulative = cumulative[att_indices]
    rho_eq_att = att_cumulative / att_cumulative.sum().clamp(min=1e-9)

    # Also compute end-of-day snapshot for secondary comparison
    rho_eq_eod = rho_norm[-30:].mean(dim=0)
    att_eod = rho_eq_eod[att_indices]
    rho_eq_att_eod = att_eod / att_eod.sum().clamp(min=1e-9)

    # ── Compute metrics ───────────────────────────────────────────────────────
    mae = calibration_mae(rho_eq_att, obs_dist)
    per_node_mae_arr = calibration_mae(rho_eq_att, obs_dist, per_node=True)
    mae_eod = calibration_mae(rho_eq_att_eod, obs_dist)
    gini_ts = gini_timeseries(rho_norm)
    gini_eq = float(gini_ts[-30:].mean())
    top3 = top_k_peak_densities(rho_norm, node_ids=node_order, k=3)

    hyp_cfg = cfg.get("hypothesis", {})
    mae_threshold = float(hyp_cfg.get("mae_threshold", 0.05))
    hypothesis_pass = mae > mae_threshold
    logger.info(
        "MAE (cumulative) = %.4f, threshold=%.2f  [hypothesis %s]",
        mae, mae_threshold,
        "SUPPORTED" if hypothesis_pass else "NOT SUPPORTED",
    )

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_cfg = out_cfg.get("figures", {})
    _generate_heatmap(rho_norm, node_order, dt, fig_cfg, outdir, io)
    _generate_comparison_bar(
        rho_eq_att, obs_dist, attractions.ATTRACTION_IDS,
        per_node_mae_arr, mae, hypothesis_pass, fig_cfg, outdir, io,
    )

    # ── metrics.csv ──────────────────────────────────────────────────────────
    _write_metrics_csv(
        rho_eq_att, obs_dist, per_node_mae_arr,
        attractions.ATTRACTION_IDS, mae, gini_eq, outdir,
    )

    # ── summary.txt ──────────────────────────────────────────────────────────
    verdict = "HYPOTHESIS SUPPORTED" if hypothesis_pass else "HYPOTHESIS NOT SUPPORTED"
    sep = "=" * 60
    top3_lines = "\n".join(
        f"  {k+1}. {nid:<25} peak_density = {pd:.4f}"
        for k, (nid, pd) in enumerate(top3)
    )
    summary = "\n".join([
        f"EXP-02 Random-Walk Baseline -- [{verdict}]",
        sep,
        f"Hypothesis: random-walk cumulative MAE > {mae_threshold:.2f} in normalised density.",
        f"Result:     MAE (cumulative) = {mae:.4f}  [{'PASS' if hypothesis_pass else 'FAIL'}]",
        f"            MAE (end-of-day) = {mae_eod:.4f}  [secondary]",
        "",
        f"Top-3 peak density nodes (in-system fraction):",
        top3_lines,
        "",
        f"Gini coefficient (equilibrium avg, last 30 steps): {gini_eq:.4f}",
        f"  (0 = uniform distribution, 1 = all tourists at one node)",
        "",
        "Simulation parameters:",
        f"  dt         = {dt:.5f} h  ({dt*60:.1f} min)",
        f"  T          = {T_hours:.1f} h  ({T_steps} steps)",
        f"  n_tourists = {n_tourists}",
        f"  exit_rate  = {exit_rate}",
        f"  sim_time   = {sim_elapsed:.3f} s",
        "",
        "Comparison metric: cumulative tourist-time per node (normalized),",
        "  comparable to annual_visitors_est which also measures total visits.",
        "Observed distribution: proxy from annual_visitors_est (to be updated",
        "  with real MGTO data once available -- see src.utils.data_loader).",
        "",
        f"Git hash: {git_hash}",
        f"Seed: {seed}",
        "",
    ])
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    return hypothesis_pass


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _generate_heatmap(rho_norm, node_order, dt, fig_cfg, outdir, io_mod) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from src.utils.attractions import ATTRACTION_NODES, NODE_BY_ID

    figsize = fig_cfg.get("figsize_heatmap", [14, 6])
    cmap = fig_cfg.get("cmap_heatmap", "YlOrRd")
    dpi = fig_cfg.get("dpi", 300)

    data = rho_norm.numpy().T  # (N_nodes, T_steps)
    T = data.shape[1]
    t_hours = np.arange(T) * dt  # hours since 08:00

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        origin="upper",
        extent=[t_hours[0], t_hours[-1], len(node_order) - 0.5, -0.5],
    )
    plt.colorbar(im, ax=ax, label="Normalized tourist density (fraction of in-system)")

    # y-tick labels: short node names
    node_labels = []
    for nid in node_order:
        node = NODE_BY_ID.get(str(nid))
        if node is not None:
            node_labels.append(node.name_en)
        else:
            node_labels.append(str(nid))
    ax.set_yticks(range(len(node_order)))
    ax.set_yticklabels(node_labels, fontsize=8)

    # x-axis: hours → HH:MM labels
    tick_hours = np.arange(0, t_hours[-1] + 0.1, 1.0)
    ax.set_xticks(tick_hours)
    ax.set_xticklabels(
        [f"{int(8 + h):02d}:00" for h in tick_hours], fontsize=8, rotation=45
    )

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Node")
    ax.set_title(
        "EXP-02: Random-Walk Density Time Series\n"
        "(density = fraction of in-system tourists at each node)",
        fontsize=11,
    )

    fig.tight_layout()
    io_mod.save_figure(fig, outdir, "density_heatmap", dpi=dpi)
    plt.close(fig)
    logger.info("Heatmap saved.")


def _generate_comparison_bar(
    rho_eq_att, obs_dist, attraction_ids, per_node_mae, mae,
    hypothesis_pass, fig_cfg, outdir, io_mod,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from src.utils.attractions import NODE_BY_ID

    figsize = fig_cfg.get("figsize_bar", [11, 6])
    dpi = fig_cfg.get("dpi", 300)

    x = np.arange(len(attraction_ids))
    width = 0.38

    short_names = []
    for nid in attraction_ids:
        node = NODE_BY_ID.get(nid)
        short_names.append(node.name_en if node else nid)

    fig, ax = plt.subplots(figsize=figsize)
    bars_sim = ax.bar(x - width / 2, rho_eq_att.numpy(), width,
                      label="Random-walk — cumulative density (simulated)", color="#4c72b0", alpha=0.85)
    bars_obs = ax.bar(x + width / 2, obs_dist.numpy(), width,
                      label="Observed proxy (annual_visitors_est)", color="#dd8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized visitor fraction")
    ax.set_title(
        f"EXP-02 Random-Walk vs Observed — MAE = {mae:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(float(obs_dist.max()), float(rho_eq_att.max())) * 1.25)

    # Verdict annotation
    color = "#2ca02c" if hypothesis_pass else "#d62728"
    verdict_text = (
        "Hypothesis SUPPORTED\n(random walk ≠ observed)" if hypothesis_pass
        else "Hypothesis NOT SUPPORTED\n(random walk ≈ observed)"
    )
    ax.annotate(
        verdict_text,
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right", va="top",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5),
    )

    fig.tight_layout()
    io_mod.save_figure(fig, outdir, "comparison_bar", dpi=dpi)
    plt.close(fig)
    logger.info("Comparison bar chart saved.")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_metrics_csv(
    rho_eq_att, obs_dist, per_node_mae, attraction_ids, mae, gini_eq, outdir
) -> None:
    csv_path = outdir / "metrics.csv"
    from src.utils.attractions import NODE_BY_ID

    rows = []
    for i, nid in enumerate(attraction_ids):
        node = NODE_BY_ID.get(nid)
        rows.append({
            "node_id": nid,
            "name_en": node.name_en if node else "",
            "sim_fraction": round(float(rho_eq_att[i].item()), 6),
            "obs_fraction": round(float(obs_dist[i].item()), 6),
            "abs_error": round(float(per_node_mae[i]), 6),
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["node_id", "name_en", "sim_fraction", "obs_fraction", "abs_error"]
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "node_id": "SUMMARY", "name_en": "cumulative_density",
            "sim_fraction": "", "obs_fraction": "",
            "abs_error": f"MAE={mae:.4f}  Gini_eq={gini_eq:.4f}",
        })

    logger.info("metrics.csv written: %s", csv_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-02: Random-Walk Baseline Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp02_baseline.yaml"),
        help="Path to YAML config (default: configs/exp02_baseline.yaml)",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = _parse_args()
    cfg = _load_config(args.config)
    success = run(cfg)
    sys.exit(0 if success else 1)
