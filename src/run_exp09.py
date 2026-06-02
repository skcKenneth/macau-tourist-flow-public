"""EXP-09: Combined intervention + compliance model (Goal D).

Three parts, all on the real 13-node Macau graph with EXP-05 fitted parameters:

Part 1 -- Combined Pareto: sweep entrance metering (on g(t)) crossed with routing
    (bonus eta) and trace the frontier of system-wide peak density at the
    bottlenecks {Ruins of St. Paul's, Senado Square} vs total preserved
    attraction-visits. Shows whether combining beats either lever alone.

Part 2 -- Compliance: the routing result assumes everyone follows the
    recommendation. Replace that with a two-type model in which only a fraction
    phi comply (src/models/mfg_solver.py::fixed_point_iteration_compliance); sweep
    phi in [0.1, 1.0] and report the DEPLOYABLE range vs the phi=1 upper bound.

Part 3 -- Robustness: re-evaluate the deployed policy under the four assumed
    within-day profiles (Goal A) and under +/-20% misspecification of beta;
    report ranges, not single numbers.

Writeup: docs/10_interventions.md.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.run_exp05 import (
    ATTRACTION_COUNT,
    _build_real_arrival_tensor,
    _daily_source_counts,
    _load_and_build_graph,
    _period,
)
from src.run_exp07 import (
    _load_fitted_params,
    _make_solver,
    _meter_arrivals,
    _params_to_solver_dict,
)
from src.run_exp08 import _candidate_edges

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_g(arrivals_df, month, node_order, sim_cfg, profile="gaussian", profile_params=None):
    dt = float(sim_cfg["dt_hours"])
    T_steps = round(float(sim_cfg["T_hours"]) / dt)
    daily = _daily_source_counts(arrivals_df, _period(month), float(sim_cfg["population_scale"]))
    return _build_real_arrival_tensor(
        node_order, T_steps, dt, daily,
        float(sim_cfg["peak_time_hours"]), float(sim_cfg["sigma_hours"]),
        profile=profile, profile_params=profile_params,
    )


def _solve(solver, g, eta, damping):
    with torch.no_grad():
        solver.routing_bonus = eta
        rho, _, info = solver.fixed_point_iteration(g, damping=damping)
    solver.routing_bonus = torch.zeros_like(eta)
    return rho, info


def _system_peak(rho, idxs):
    return float(rho[:, idxs].max().item())


def _visits(rho, dt):
    return float((rho[:, :ATTRACTION_COUNT] * dt).sum().item())


def run(cfg: dict[str, Any]) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.optimization.interventions import RoutingOptimizer
    from src.utils import attractions, io

    from src.utils.data_loader import load_arrivals_monthly

    exp_cfg = cfg["experiment"]
    sim_cfg = cfg["simulation"]
    damping = float(cfg["solver"]["damping"])
    dt = float(sim_cfg["dt_hours"])
    io.set_all_seeds(int(exp_cfg["seed"]))
    outdir = io.make_experiment_dir(base=Path(cfg["output"]["base_dir"]), name=f"{exp_cfg['id']}_{exp_cfg['name']}")
    io.save_config(cfg, outdir)
    logger.info("Output dir: %s", outdir)

    arrivals_df = load_arrivals_monthly(cfg["data"]["arrivals_path"])
    node_order = [n.node_id for n in attractions.ATTRACTION_NODES]
    N = len(node_order)
    G = _load_and_build_graph(cfg, node_order)
    fitted = _load_fitted_params(Path(cfg["data"]["fitted_params_path"]))
    params_dict = _params_to_solver_dict(fitted, n_nodes=N)
    solver = _make_solver(G, params_dict, cfg)

    month = cfg["peak_month"]
    bidx = [node_order.index(b) for b in cfg["bottlenecks"]]
    ruins_idx = node_order.index("ruins_st_pauls")
    ferry_col = node_order.index(cfg["metering"]["source_node"])
    zero_eta = torch.zeros(N, N, dtype=torch.float32)

    g = _build_g(arrivals_df, month, node_order, sim_cfg)

    # Baseline (no intervention)
    rho_base, _ = _solve(solver, g, zero_eta, damping)
    peak0 = _system_peak(rho_base, bidx)
    visits0 = _visits(rho_base, dt)
    ruins0 = float(rho_base[:, ruins_idx].max().item())
    logger.info("Baseline: system_peak=%.4f visits=%.1f ruins_peak=%.4f", peak0, visits0, ruins0)

    # ── Optimise routing eta (full-compliance design) ─────────────────────────
    rcfg = cfg["routing"]
    candidate_edges = _candidate_edges(solver, None, dst_attractions_only=True, attraction_count=ATTRACTION_COUNT)
    logger.info("Optimising routing eta (%d candidate edges) …", len(candidate_edges))
    opt = RoutingOptimizer(
        solver=solver, g=g, bottleneck_idx=list(range(ATTRACTION_COUNT)), report_idx=ruins_idx,
        candidate_edges=candidate_edges, attraction_count=ATTRACTION_COUNT, node_labels=node_order,
        eta_max=float(rcfg["eta_max"]), lambda_l1=float(rcfg["lambda_l1"]),
        lambda_visit=float(rcfg["lambda_visit"]), delta=float(rcfg["delta"]),
        peak_temp=float(rcfg["peak_temp"]), damping=damping,
    )
    res = opt.optimise(n_steps=int(rcfg["n_steps"]), lr=float(rcfg["lr"]),
                       lr_decay=float(rcfg["lr_decay"]), grad_clip=float(rcfg["grad_clip"]))
    eta = res["eta"]

    # ── Part 1: combined metering x routing Pareto ────────────────────────────
    pareto_rows = []
    for frac in cfg["metering"]["r_star_fractions"]:
        g_m = g if frac >= 1.0 else _meter_arrivals(g, float(g[:, ferry_col].max().item()) * float(frac), dt, ferry_col)
        for route_on in (False, True):
            rho, info = _solve(solver, g_m, eta if route_on else zero_eta, damping)
            peak = _system_peak(rho, bidx)
            visits = _visits(rho, dt)
            pareto_rows.append({
                "metering_frac": frac,
                "routing": route_on,
                "system_peak": peak,
                "system_peak_reduction_pct": 100.0 * (peak0 - peak) / (peak0 + 1e-12),
                "preserved_visits_pct": 100.0 * visits / (visits0 + 1e-12),
                "converged": info["converged"],
            })
    logger.info("Part 1 Pareto: %d points", len(pareto_rows))

    # ── Part 2: compliance sweep (optimised eta, no metering) ─────────────────
    phi_rows = []
    for phi in cfg["compliance"]["phi_values"]:
        rho_phi, info = solver.fixed_point_iteration_compliance(g, eta, phi=float(phi), damping=damping)
        ruins_peak = float(rho_phi[:, ruins_idx].max().item())
        phi_rows.append({
            "phi": float(phi),
            "ruins_peak": ruins_peak,
            "ruins_peak_reduction_pct": 100.0 * (ruins0 - ruins_peak) / (ruins0 + 1e-12),
            "converged": info["converged"],
        })
        logger.info("  phi=%.2f -> ruins peak_red=%.1f%% conv=%s",
                    phi, phi_rows[-1]["ruins_peak_reduction_pct"], info["converged"])

    # ── Part 2b: heterogeneous/uncertain compliance distribution ──────────────
    from src.optimization.interventions import (
        compliance_robustness_band,
        sample_beta_compliance,
    )

    dist_cfg = cfg["compliance"].get("distribution")
    dist_band: dict[str, Any] | None = None
    if dist_cfg:
        phi_draws = sample_beta_compliance(
            mean=float(dist_cfg["mean"]),
            concentration=float(dist_cfg["concentration"]),
            n_samples=int(dist_cfg["n_samples"]),
            seed=int(dist_cfg.get("seed", 42)),
        )
        dist_band = compliance_robustness_band(
            solver, g, eta, report_idx=ruins_idx, phi_samples=phi_draws, damping=damping
        )
        logger.info(
            "  Compliance ~ Beta(mean=%.2f, kappa=%.1f): routing peak_red "
            "mean=%.1f%% band[p5,p95]=[%.1f%%, %.1f%%] conv=%.0f%%",
            float(dist_cfg["mean"]), float(dist_cfg["concentration"]),
            dist_band["reduction_mean"], dist_band["reduction_p5"],
            dist_band["reduction_p95"], 100.0 * dist_band["frac_converged"],
        )

    # ── Part 3: robustness (profiles x beta), deployed eta, full compliance ───
    rob_rows = []
    base_beta = float(params_dict["beta"].item())
    for prof in cfg["robustness"]["profiles"]:
        for bscale in cfg["robustness"]["beta_perturbations"]:
            pd = {"alpha": params_dict["alpha"].clone(), "beta": torch.tensor(base_beta * float(bscale)), "gamma": params_dict["gamma"].clone()}
            solver.params = solver._parse_params(pd)
            g_p = _build_g(arrivals_df, month, node_order, sim_cfg, profile=prof["name"])
            rho_b, _ = _solve(solver, g_p, zero_eta, damping)
            rho_r, info = _solve(solver, g_p, eta, damping)
            r0 = float(rho_b[:, ruins_idx].max().item())
            r1 = float(rho_r[:, ruins_idx].max().item())
            rob_rows.append({
                "profile": prof["name"], "beta_scale": float(bscale),
                "ruins_peak_reduction_pct": 100.0 * (r0 - r1) / (r0 + 1e-12),
                "converged": info["converged"],
            })
    solver.params = solver._parse_params(params_dict)  # restore
    logger.info("Part 3 robustness: %d (profile x beta) runs", len(rob_rows))

    _write_outputs(outdir, cfg, pareto_rows, phi_rows, rob_rows, res, node_order, dist_band)
    logger.info("EXP-09 done. Outputs in: %s", outdir)
    return True


def _write_outputs(outdir, cfg, pareto_rows, phi_rows, rob_rows, res, node_order, dist_band=None):
    import matplotlib.pyplot as plt
    from src.utils import io
    fig_cfg = cfg["output"]["figures"]
    dpi = int(fig_cfg.get("dpi", 300))

    for name, rows in (("pareto", pareto_rows), ("compliance_phi", phi_rows), ("robustness", rob_rows)):
        with open(outdir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Pareto figure
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_pareto", [8, 6])))
    for route_on, color, lab in ((False, "tab:blue", "Metering only"), (True, "tab:orange", "Metering + routing")):
        pts = [r for r in pareto_rows if r["routing"] == route_on]
        ax.plot([r["preserved_visits_pct"] for r in pts], [r["system_peak_reduction_pct"] for r in pts],
                "o-", color=color, label=lab)
    ax.set_xlabel("Preserved attraction-visits (% of baseline)")
    ax.set_ylabel("System-wide peak reduction at bottlenecks (%)")
    ax.set_title("EXP-09 Part 1: combined-intervention Pareto frontier")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); io.save_figure(fig, outdir, "pareto_combined", dpi=dpi); plt.close(fig)

    # Compliance distribution CSV (Part 2b)
    if dist_band is not None:
        with open(outdir / "compliance_distribution.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["statistic", "value"])
            for k, v in dist_band.items():
                if k == "reductions":
                    continue
                w.writerow([k, v])

    # Compliance phi figure (deterministic sweep + distribution band overlay)
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_phi", [8, 5])))
    ax.plot([r["phi"] for r in phi_rows], [r["ruins_peak_reduction_pct"] for r in phi_rows],
            "o-", color="tab:green", label="Deterministic phi sweep")
    if dist_band is not None:
        # Shade the p5-p95 routing-benefit band from the Beta-distributed compliance,
        # placed at the population mean compliance on the x-axis.
        x = dist_band["phi_mean_sampled"]
        ax.fill_between([0.0, 1.0], dist_band["reduction_p5"], dist_band["reduction_p95"],
                        color="tab:purple", alpha=0.12,
                        label="Heterogeneous compliance p5-p95")
        ax.errorbar([x], [dist_band["reduction_mean"]],
                    yerr=[[dist_band["reduction_mean"] - dist_band["reduction_p5"]],
                          [dist_band["reduction_p95"] - dist_band["reduction_mean"]]],
                    fmt="s", color="tab:purple", capsize=4,
                    label="Beta-compliance mean +/- band")
    ax.set_xlabel("Compliance fraction phi (who follow the recommendation)")
    ax.set_ylabel("Peak reduction at Ruins of St. Paul's (%)")
    ax.set_title("EXP-09 Part 2: routing benefit vs compliance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); io.save_figure(fig, outdir, "compliance_phi", dpi=dpi); plt.close(fig)

    # Summary
    best_combo = max((r for r in pareto_rows if r["routing"] and r["preserved_visits_pct"] >= 90.0),
                     key=lambda r: r["system_peak_reduction_pct"], default=None)
    meter_only_best = max((r for r in pareto_rows if not r["routing"]),
                          key=lambda r: r["system_peak_reduction_pct"], default=None)
    full_phi = next(r for r in phi_rows if r["phi"] == 1.0)
    low_phi = min(phi_rows, key=lambda r: r["phi"])
    rob_reds = [r["ruins_peak_reduction_pct"] for r in rob_rows]
    rob_conv = all(r["converged"] for r in rob_rows)

    lines = [
        "EXP-09 Combined Intervention + Compliance",
        "=" * 60, "",
        "PART 1 -- combined metering x routing Pareto (system peak at {Ruins, Senado}):",
        f"  Metering-only best system-peak reduction: {meter_only_best['system_peak_reduction_pct']:.1f}% "
        f"(preserving {meter_only_best['preserved_visits_pct']:.0f}% visits)",
    ]
    if best_combo:
        lines.append(f"  Combined best (>=90% visits preserved): {best_combo['system_peak_reduction_pct']:.1f}% "
                     f"at metering_frac={best_combo['metering_frac']}, preserving {best_combo['preserved_visits_pct']:.0f}% visits")
    lines += [
        "",
        "PART 2 -- compliance (deployable range vs perfect-compliance upper bound):",
        f"  Routing peak reduction at full compliance (phi=1): {full_phi['ruins_peak_reduction_pct']:.1f}% (UPPER BOUND)",
        f"  At phi={low_phi['phi']:.2f}: {low_phi['ruins_peak_reduction_pct']:.1f}% (deployable lower end)",
        "  -> report the band, not the single phi=1 number.",
    ]
    if dist_band is not None:
        lines += [
            f"  Heterogeneous compliance ~ Beta(mean={dist_band['phi_mean_sampled']:.2f}): "
            f"routing peak_red mean={dist_band['reduction_mean']:.1f}% "
            f"band[p5,p95]=[{dist_band['reduction_p5']:.1f}%, {dist_band['reduction_p95']:.1f}%] "
            f"(n={dist_band['n_samples']}, converged={100*dist_band['frac_converged']:.0f}%)",
        ]
    lines += [
        "",
        "PART 3 -- robustness of the deployed routing policy (profiles x +/-20% beta):",
        f"  Ruins peak-reduction range: [{min(rob_reds):.1f}%, {max(rob_reds):.1f}%]; all converged={rob_conv}",
        "",
        "Top recommended edges (full-compliance design):",
    ]
    for e in res["top_edges"]:
        lines.append(f"  {e['src']} -> {e['dst']} : eta={e['eta']:+.4f}")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-09: combined intervention + compliance.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent.parent / "configs" / "exp09_combined.yaml")
    args = parser.parse_args(argv)
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 1
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return 0 if run(cfg) else 1


if __name__ == "__main__":
    sys.exit(main())
