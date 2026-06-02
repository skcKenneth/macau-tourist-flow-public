"""EXP-08: Routing-recommendation intervention on the real Macau graph.

Models an informational nudge (signage / app recommendation) as an additive
"perceived" bonus eta_uv on graph edges, entering the choice value
``Q_cont[v, w] = -gamma*D[v,w] + eta[v,w] + u[t+1, w]`` in both the HJB max and
the FP softmax policy. The bonus is optimised by gradient descent through the
calibrated differentiable MFG simulator to reduce peak density at the headline
bottleneck (Ruins of St. Paul's), while a visit-preservation penalty keeps total
attraction-hours from dropping by more than ``delta``.

Modelling note (state in the report): eta is a *behavioural nudge* — agents
perceive recommended edges as more valuable and best-respond. Realised density
follows that policy; the experience-cost metric (total attraction-hours) is
measured on realised density with the true alpha, so it captures the genuine
cost of diverting tourists.

Hypothesis: an optimised routing bonus reduces peak density at Ruins of St.
Paul's by >= 10% without reducing total attraction-hours by more than 10%.

Reuses EXP-05 data helpers and EXP-07 solver/metric helpers (no duplication).
Optimises eta on the peak month (Aug 2025), then transfers it to a moderate
month (Jan 2025) as a robustness check, and runs a top-edge ablation plus a
shuffled-eta control to show where the gain comes from.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

# ---------------------------------------------------------------------------
# Imports from EXP-05 / EXP-07 (reuse — do not duplicate)
# ---------------------------------------------------------------------------
from src.run_exp05 import (
    ATTRACTION_COUNT,
    _build_real_arrival_tensor,
    _daily_source_counts,
    _load_and_build_graph,
    _period,
)
from src.run_exp07 import (
    _compute_metrics,
    _load_fitted_params,
    _make_solver,
    _params_to_solver_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_arrival_tensor_for_month(
    month_str: str,
    arrivals_df,
    node_order: list[str],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Build the baseline Gaussian arrival tensor for one month (DSEC-derived)."""
    sim_cfg = cfg["simulation"]
    dt = float(sim_cfg["dt_hours"])
    T_steps = round(float(sim_cfg["T_hours"]) / dt)
    daily_counts = _daily_source_counts(
        arrivals_df, _period(month_str), float(sim_cfg["population_scale"])
    )
    return _build_real_arrival_tensor(
        node_order=node_order,
        T_steps=T_steps,
        dt=dt,
        daily_source_counts=daily_counts,
        peak_time_hours=float(sim_cfg["peak_time_hours"]),
        sigma_hours=float(sim_cfg["sigma_hours"]),
    )


def _candidate_edges(
    solver,
    max_dist_m: float | None,
    dst_attractions_only: bool = True,
    attraction_count: int = ATTRACTION_COUNT,
) -> list[tuple[int, int]]:
    """Finite off-diagonal edges eligible for a routing bonus.

    A routing recommendation can only point a tourist toward a *destination
    attraction* — you cannot meaningfully "recommend" parking at a transit node.
    With ``dst_attractions_only`` (default), candidate destinations are limited to
    the leading ``attraction_count`` attraction columns, which removes the
    degenerate exploit of keeping tourists circulating among transit nodes (and
    so out of every attraction). Sources may be any node (a tourist at a transit
    node or at an attraction choosing which attraction to visit next).
    """
    D = solver.D
    N = solver.N_nodes
    edges: list[tuple[int, int]] = []
    for i in range(N):
        for j in range(N):
            if i == j or not torch.isfinite(D[i, j]):
                continue
            if dst_attractions_only and j >= attraction_count:
                continue
            if max_dist_m is not None and float(D[i, j].item()) > max_dist_m:
                continue
            edges.append((i, j))
    return edges


def _solve_with_eta(
    solver, g: torch.Tensor, eta: torch.Tensor, damping: float
) -> torch.Tensor:
    """Run the fixed-point solver under a given routing bonus (no grad)."""
    with torch.no_grad():
        solver.routing_bonus = eta
        rho, _, _ = solver.fixed_point_iteration(g, damping=damping)
    return rho


def _reductions(m_base: dict[str, float], m_eta: dict[str, float]) -> tuple[float, float]:
    """Return (peak_reduction_pct, visit_reduction_pct) of eta vs baseline."""
    peak_red = 100.0 * (m_base["peak_density_ruins"] - m_eta["peak_density_ruins"]) / (
        m_base["peak_density_ruins"] + 1e-12
    )
    visit_red = 100.0 * (m_base["total_att_hours"] - m_eta["total_att_hours"]) / (
        m_base["total_att_hours"] + 1e-12
    )
    return peak_red, visit_red


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> bool:
    """Execute EXP-08 and return True if the hypothesis is satisfied."""
    import matplotlib
    matplotlib.use("Agg")

    from src.evaluation.metrics import top_k_peak_densities
    from src.optimization.interventions import RoutingOptimizer
    from src.utils import attractions, io

    from src.utils.data_loader import load_arrivals_monthly

    exp_cfg = cfg["experiment"]
    routing_cfg = cfg["routing"]
    success_cfg = cfg["success"]
    damping = float(cfg["solver"]["damping"])
    dt = float(cfg["simulation"]["dt_hours"])

    io.set_all_seeds(int(exp_cfg["seed"]))
    outdir = io.make_experiment_dir(
        base=Path(cfg["output"]["base_dir"]),
        name=f"{exp_cfg['id']}_{exp_cfg['name']}",
    )
    io.save_config(cfg, outdir)
    logger.info("Output directory: %s", outdir)

    # Data + graph
    arrivals_df = load_arrivals_monthly(cfg["data"]["arrivals_path"])
    node_order = [n.node_id for n in attractions.ATTRACTION_NODES]

    fitted_params_path = Path(cfg["data"]["fitted_params_path"])
    fitted = _load_fitted_params(fitted_params_path)
    shutil.copy(fitted_params_path, outdir / "fitted_params.yaml")
    logger.info("Loaded EXP-05 fitted params: beta=%.6f gamma=%.8f",
                fitted["beta"], fitted["gamma"])

    logger.info("Loading Macau OSM graph …")
    t_graph = time.perf_counter()
    G = _load_and_build_graph(cfg, node_order)
    logger.info("Graph loaded in %.1f s", time.perf_counter() - t_graph)

    params_dict = _params_to_solver_dict(fitted, n_nodes=len(node_order))
    solver = _make_solver(G, params_dict, cfg)

    bottleneck_node = str(routing_cfg["bottleneck_node"])
    if bottleneck_node not in node_order:
        raise ValueError(f"bottleneck_node {bottleneck_node!r} not in node_order")
    bottleneck_idx = node_order.index(bottleneck_node)

    # Bottleneck SET B for the system-wide max objective. Minimising the joint
    # peak over all attractions prevents the optimiser from merely relocating the
    # crowd onto a neighbour (which would create a new hot spot). Configurable;
    # defaults to all attraction nodes.
    bottleneck_set_cfg = routing_cfg.get("bottleneck_set", "all_attractions")
    if bottleneck_set_cfg == "all_attractions":
        bottleneck_idxs = list(range(ATTRACTION_COUNT))
    else:
        bottleneck_idxs = [node_order.index(str(n)) for n in bottleneck_set_cfg]
    logger.info("Bottleneck set B (objective): %s; headline report node: %s",
                [node_order[i] for i in bottleneck_idxs], bottleneck_node)

    max_dist = routing_cfg.get("candidate_max_dist_m")
    dst_only = bool(routing_cfg.get("candidate_dst_attractions_only", True))
    candidate_edges = _candidate_edges(
        solver, max_dist, dst_attractions_only=dst_only,
        attraction_count=ATTRACTION_COUNT,
    )
    logger.info("Routing candidate edges: %d (bottleneck=%s, idx=%d, dst_attractions_only=%s)",
                len(candidate_edges), bottleneck_node, bottleneck_idx, dst_only)

    months = list(cfg["months"])
    opt_month = months[0]

    # ── Optimise eta on the peak month ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Optimising routing bonus on %s", opt_month)
    g_opt = _build_arrival_tensor_for_month(opt_month, arrivals_df, node_order, cfg)

    optimizer = RoutingOptimizer(
        solver=solver,
        g=g_opt,
        bottleneck_idx=bottleneck_idxs,
        report_idx=bottleneck_idx,
        candidate_edges=candidate_edges,
        attraction_count=ATTRACTION_COUNT,
        node_labels=node_order,
        eta_max=float(routing_cfg["eta_max"]),
        lambda_l1=float(routing_cfg["lambda_l1"]),
        lambda_visit=float(routing_cfg["lambda_visit"]),
        delta=float(routing_cfg["delta"]),
        peak_temp=float(routing_cfg["peak_temp"]),
        damping=damping,
    )
    t0 = time.perf_counter()
    opt_result = optimizer.optimise(
        n_steps=int(routing_cfg["n_steps"]),
        lr=float(routing_cfg["lr"]),
        lr_decay=float(routing_cfg["lr_decay"]),
        grad_clip=float(routing_cfg["grad_clip"]),
    )
    logger.info("Routing optimisation done in %.1f s", time.perf_counter() - t0)
    eta = opt_result["eta"]
    logger.info(
        "Optimised: report peak_red=%.1f%% (%.4f->%.4f), system peak_red=%.1f%% "
        "(%.4f->%.4f), converged=%s",
        opt_result["peak_density_reduction_pct"],
        opt_result["peak_baseline"], opt_result["peak_optimized"],
        opt_result["system_peak_reduction_pct"],
        opt_result["system_peak_baseline"], opt_result["system_peak_optimized"],
        opt_result["optimized_converged"],
    )
    logger.info("Top recommended edges:")
    for e in opt_result["top_edges"]:
        logger.info("  %s -> %s : eta=%+.4f", e["src"], e["dst"], e["eta"])

    # ── Evaluate baseline vs optimised eta on every month ─────────────────────
    zero_eta = torch.zeros_like(eta)
    month_rows: list[dict[str, Any]] = []
    rho_cache: dict[str, dict[str, torch.Tensor]] = {}

    for month_str in months:
        g_m = _build_arrival_tensor_for_month(month_str, arrivals_df, node_order, cfg)
        rho_base = _solve_with_eta(solver, g_m, zero_eta, damping)
        rho_eta = _solve_with_eta(solver, g_m, eta, damping)
        m_base = _compute_metrics(rho_base, g_m, dt, bottleneck_idx)
        m_eta = _compute_metrics(rho_eta, g_m, dt, bottleneck_idx)
        peak_red, visit_red = _reductions(m_base, m_eta)

        month_rows.append({
            "month": month_str,
            "peak_baseline": m_base["peak_density_ruins"],
            "peak_optimized": m_eta["peak_density_ruins"],
            "peak_reduction_pct": peak_red,
            "att_hours_baseline": m_base["total_att_hours"],
            "att_hours_optimized": m_eta["total_att_hours"],
            "visit_reduction_pct": visit_red,
            "gini_baseline": m_base["gini"],
            "gini_optimized": m_eta["gini"],
        })
        rho_cache[month_str] = {"base": rho_base, "eta": rho_eta, "g": g_m}
        logger.info(
            "[%s] peak_red=%.1f%% visit_red=%.1f%% (peak %.4f -> %.4f)",
            month_str, peak_red, visit_red,
            m_base["peak_density_ruins"], m_eta["peak_density_ruins"],
        )

    # ── Ablation on the optimisation month: zero each top edge one at a time ──
    g_opt_tensor = rho_cache[opt_month]["g"]
    rho_full = rho_cache[opt_month]["eta"]
    peak_full = float(rho_full[:, bottleneck_idx].max().item())
    peak_base_opt = float(rho_cache[opt_month]["base"][:, bottleneck_idx].max().item())
    full_gain = peak_base_opt - peak_full  # absolute peak reduction from full eta

    ablation_rows: list[dict[str, Any]] = []
    for e in opt_result["top_edges"]:
        i = node_order.index(e["src"]) if e["src"] in node_order else int(e["src"])
        j = node_order.index(e["dst"]) if e["dst"] in node_order else int(e["dst"])
        eta_ablated = eta.clone()
        eta_ablated[i, j] = 0.0
        rho_ab = _solve_with_eta(solver, g_opt_tensor, eta_ablated, damping)
        peak_ab = float(rho_ab[:, bottleneck_idx].max().item())
        # Lost gain = how much peak rises when this edge is removed.
        lost_gain = peak_ab - peak_full
        attribution = 100.0 * lost_gain / (full_gain + 1e-12)
        ablation_rows.append({
            "edge": f"{e['src']} -> {e['dst']}",
            "eta": e["eta"],
            "peak_without_edge": peak_ab,
            "lost_gain_abs": lost_gain,
            "gain_attribution_pct": attribution,
        })
        logger.info("  Ablate %s->%s: peak %.4f -> %.4f (%.0f%% of gain)",
                    e["src"], e["dst"], peak_full, peak_ab, attribution)

    # Shuffled-eta control: same bonus values, randomly reassigned to edges.
    perm = torch.randperm(len(candidate_edges))
    eta_shuffled = torch.zeros_like(eta)
    src = torch.tensor([c[0] for c in candidate_edges])
    dst = torch.tensor([c[1] for c in candidate_edges])
    eta_vals = eta[src, dst]
    eta_shuffled[src, dst] = eta_vals[perm]
    rho_ctrl = _solve_with_eta(solver, g_opt_tensor, eta_shuffled, damping)
    peak_ctrl = float(rho_ctrl[:, bottleneck_idx].max().item())
    ctrl_red = 100.0 * (peak_base_opt - peak_ctrl) / (peak_base_opt + 1e-12)
    opt_red = 100.0 * (peak_base_opt - peak_full) / (peak_base_opt + 1e-12)
    logger.info("Control (shuffled eta): peak_red=%.1f%% vs optimised %.1f%%",
                ctrl_red, opt_red)

    solver.routing_bonus = zero_eta  # leave solver clean

    # ── Success check ─────────────────────────────────────────────────────────
    # A credible result must (1) reduce the headline bottleneck peak by the
    # target, (2) not degrade total attraction-hours beyond delta, (3) reduce the
    # SYSTEM-wide max peak (not merely relocate the crowd), (4) not increase the
    # spatial Gini (genuine smoothing), and (5) rest on a converged equilibrium.
    min_peak = float(success_cfg["min_peak_reduction_pct"])
    max_visit = float(success_cfg["max_visit_reduction_pct"])
    opt_row = next(r for r in month_rows if r["month"] == opt_month)
    checks = {
        "headline_peak": opt_row["peak_reduction_pct"] >= min_peak,
        "visits_preserved": opt_row["visit_reduction_pct"] <= max_visit,
        "system_peak_reduced": opt_result["system_peak_reduction_pct"] > 0.0,
        "gini_not_worse": opt_row["gini_optimized"] <= opt_row["gini_baseline"] + 1e-6,
        "converged": opt_result["optimized_converged"],
    }
    passed = all(checks.values())
    logger.info(
        "EXP-08 PASS=%s (opt month %s: peak_red=%.1f%%, system_red=%.1f%%, "
        "visit_red=%.1f%%, Gini %.3f->%.3f). Checks: %s",
        passed, opt_month, opt_row["peak_reduction_pct"],
        opt_result["system_peak_reduction_pct"], opt_row["visit_reduction_pct"],
        opt_row["gini_baseline"], opt_row["gini_optimized"], checks,
    )

    # ── Outputs ────────────────────────────────────────────────────────────────
    _write_csvs(outdir, opt_month, opt_result, candidate_edges, eta, node_order,
                solver, month_rows, ablation_rows)
    _write_figures(outdir, cfg, opt_month, opt_result, rho_cache, bottleneck_idx,
                   bottleneck_node, node_order, ablation_rows, dt)
    _write_summary(outdir, cfg, opt_month, opt_row, month_rows, ablation_rows,
                   opt_result, opt_red, ctrl_red, passed, checks)

    logger.info("=" * 60)
    logger.info("EXP-08 overall PASS=%s. Outputs in: %s", passed, outdir)
    return passed


def _write_csvs(outdir, opt_month, opt_result, candidate_edges, eta, node_order,
                solver, month_rows, ablation_rows):
    yyyymm = opt_month.replace("-", "")

    # Per-month metrics
    with open(outdir / "month_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(month_rows[0].keys()))
        writer.writeheader()
        writer.writerows(month_rows)

    # All nonzero edges with eta and distance
    with open(outdir / f"edges_{yyyymm}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["src", "dst", "eta", "distance_m"]
        )
        writer.writeheader()
        scored = sorted(
            candidate_edges, key=lambda c: abs(float(eta[c[0], c[1]].item())), reverse=True
        )
        for i, j in scored:
            val = float(eta[i, j].item())
            if abs(val) < 1e-4:
                continue
            writer.writerow({
                "src": node_order[i],
                "dst": node_order[j],
                "eta": f"{val:.6f}",
                "distance_m": f"{float(solver.D[i, j].item()):.1f}",
            })

    # Ablation
    with open(outdir / f"ablation_{yyyymm}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)


def _write_figures(outdir, cfg, opt_month, opt_result, rho_cache, bottleneck_idx,
                   bottleneck_node, node_order, ablation_rows, dt):
    import matplotlib.pyplot as plt
    from src.utils import io

    fig_cfg = cfg["output"]["figures"]
    dpi = int(fig_cfg.get("dpi", 300))
    yyyymm = opt_month.replace("-", "")

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_loss", [8, 5])))
    ax.plot(opt_result["loss_history"], color="tab:blue")
    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Loss (smooth peak + penalties)")
    ax.set_title(f"EXP-08 routing optimisation loss — {opt_month}")
    ax.grid(True, alpha=0.3)
    io.save_figure(fig, outdir, "loss_curve", dpi=dpi)
    plt.close(fig)

    # 2. Density evolution at bottleneck (baseline vs optimised)
    rho_base = rho_cache[opt_month]["base"]
    rho_eta = rho_cache[opt_month]["eta"]
    T_steps = rho_base.shape[0]
    t_vec = [i * dt for i in range(T_steps)]
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_density", [12, 5])))
    ax.plot(t_vec, rho_base[:, bottleneck_idx].numpy(), label="Baseline (no routing)",
            color="tab:blue", linewidth=2)
    ax.plot(t_vec, rho_eta[:, bottleneck_idx].numpy(), label="Optimised routing bonus",
            color="tab:orange", linewidth=2, linestyle="--")
    ax.set_xlabel("Time (hours into operating day)")
    ax.set_ylabel(f"Density at {bottleneck_node} (tourists)")
    ax.set_title(f"EXP-08 density evolution — {bottleneck_node} — {opt_month}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    io.save_figure(fig, outdir, f"density_evolution_{yyyymm}", dpi=dpi)
    plt.close(fig)

    # 3. Top recommended edges (signed eta)
    top = opt_result["top_edges"]
    if top:
        labels = [f"{e['src']}\n→ {e['dst']}" for e in top]
        vals = [e["eta"] for e in top]
        colors = ["tab:green" if v >= 0 else "tab:red" for v in vals]
        fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_edges", [9, 6])))
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Routing bonus η (utility units; + nudges toward, − away)")
        ax.set_title(f"EXP-08 top routing recommendations — {opt_month}")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        io.save_figure(fig, outdir, f"top_edges_{yyyymm}", dpi=dpi)
        plt.close(fig)

    # 4. Ablation tornado (gain attribution)
    if ablation_rows:
        labels = [r["edge"] for r in ablation_rows]
        attrib = [r["gain_attribution_pct"] for r in ablation_rows]
        fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_ablation", [9, 6])))
        ax.barh(range(len(attrib)), attrib, color="tab:purple")
        ax.set_yticks(range(len(attrib)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Share of peak-reduction gain lost when edge removed (%)")
        ax.set_title(f"EXP-08 routing-edge ablation — {opt_month}")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        io.save_figure(fig, outdir, f"ablation_{yyyymm}", dpi=dpi)
        plt.close(fig)

    # 5. Per-attraction peak density: baseline vs optimised
    peak_base = rho_base[:, :ATTRACTION_COUNT].max(dim=0).values.numpy()
    peak_eta = rho_eta[:, :ATTRACTION_COUNT].max(dim=0).values.numpy()
    att_ids = node_order[:ATTRACTION_COUNT]
    x = list(range(ATTRACTION_COUNT))
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_bar", [12, 5])))
    ax.bar([i - 0.2 for i in x], peak_base, width=0.4, label="Baseline", color="tab:blue")
    ax.bar([i + 0.2 for i in x], peak_eta, width=0.4, label="Optimised routing", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(att_ids, rotation=45, ha="right")
    ax.set_ylabel("Peak density (tourists)")
    ax.set_title(f"EXP-08 per-attraction peak density — {opt_month}")
    ax.legend()
    fig.tight_layout()
    io.save_figure(fig, outdir, f"peak_by_attraction_{yyyymm}", dpi=dpi)
    plt.close(fig)


def _write_summary(outdir, cfg, opt_month, opt_row, month_rows, ablation_rows,
                   opt_result, opt_red, ctrl_red, passed, checks):
    success_cfg = cfg["success"]
    lines = [
        "EXP-08 Routing Recommendations -- " + ("PASS" if passed else "FAIL"),
        "=" * 60,
        f"Headline bottleneck node: {cfg['routing']['bottleneck_node']}",
        f"Objective: minimise system-wide max peak over set B "
        f"({cfg['routing'].get('bottleneck_set', 'all_attractions')})",
        f"Optimisation month: {opt_month}",
        f"Success criteria: headline peak_reduction >= "
        f"{success_cfg['min_peak_reduction_pct']:.1f}% AND visit_reduction <= "
        f"{success_cfg['max_visit_reduction_pct']:.1f}% AND system peak reduced "
        f"AND Gini not worse AND equilibrium converged",
        f"Checks: {checks}",
        "",
        f"System-wide max peak: {opt_result['system_peak_baseline']:.4f} -> "
        f"{opt_result['system_peak_optimized']:.4f} "
        f"({opt_result['system_peak_reduction_pct']:.1f}% reduction); "
        f"optimised equilibrium converged={opt_result['optimized_converged']}",
        "",
        "Per-month results (optimised eta applied):",
    ]
    for r in month_rows:
        lines.append(
            f"  {r['month']}: peak_reduction={r['peak_reduction_pct']:.1f}%  "
            f"visit_reduction={r['visit_reduction_pct']:.1f}%  "
            f"(peak {r['peak_baseline']:.4f} -> {r['peak_optimized']:.4f}, "
            f"Gini {r['gini_baseline']:.3f} -> {r['gini_optimized']:.3f})"
        )
    lines += [
        "",
        "Top recommended edges (eta):",
    ]
    for e in opt_result["top_edges"]:
        lines.append(f"  {e['src']} -> {e['dst']} : eta={e['eta']:+.4f}")
    lines += [
        "",
        f"Control (shuffled eta, same L1, wrong edges): peak_reduction={ctrl_red:.1f}% "
        f"vs optimised {opt_red:.1f}%",
        "",
        "Edge ablation (gain attribution on optimisation month):",
    ]
    for r in ablation_rows:
        lines.append(
            f"  {r['edge']}: {r['gain_attribution_pct']:.0f}% of gain "
            f"(peak rises to {r['peak_without_edge']:.4f} when removed)"
        )
    lines += ["", f"Overall: {'PASS' if passed else 'FAIL'}"]

    summary_path = outdir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Summary written to %s", summary_path)
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_cfg_path() -> Path:
    return Path(__file__).parent.parent / "configs" / "exp08_routing.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-08: Routing recommendations.")
    parser.add_argument("--config", type=Path, default=_default_cfg_path(),
                        help="Path to YAML config file.")
    args = parser.parse_args(argv)

    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 1

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    passed = run(cfg)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
