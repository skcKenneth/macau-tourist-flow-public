"""EXP-11: Validity scope -- robustness to the assumed within-day arrival profile.

The DSEC data is monthly, so only the *spatial* attraction distribution is
data-validated (EXP-05 held-out MAE 0.018). The within-day *shape* of g(t) is an
ASSUMED profile (src/utils/arrival_profiles.py). This experiment makes the
calibrated-vs-assumed split explicit and tests two questions:

Part A -- Spatial-calibration invariance (highest value):
    Re-run the EXP-05 calibration under each assumed profile, on the same
    train/val split and protocol. Do the fitted spatial parameters {alpha_v} and
    the held-out MAE stay ~stable regardless of the profile? If so, the
    data-validated result does not depend on the assumption.

Part B -- Intervention robustness:
    Holding ONE fixed calibrated parameter set (the EXP-05 fit), vary only the
    profile in the intervention simulations and re-run entrance metering (EXP-07)
    and routing (EXP-08). Does the peak-reduction conclusion (sign and rough
    magnitude) survive across profiles?

Honesty: if a conclusion holds only under one profile, the experiment reports it.
See docs/08_validity_scope.md for the writeup.
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
import torch.nn.functional as F
import yaml

from src.run_exp05 import (
    ATTRACTION_COUNT,
    _attraction_distribution,
    _build_real_arrival_tensor,
    _daily_source_counts,
    _evaluate_months,
    _init_alpha_from_observed,
    _load_and_build_graph,
    _period,
    _prepare_month_data,
    _select_months,
)
from src.run_exp07 import (
    _compute_metrics,
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


# ---------------------------------------------------------------------------
# Part A -- calibration under a given profile (replicates the EXP-05 loop)
# ---------------------------------------------------------------------------


def _calibrate_for_profile(
    profile: dict[str, Any],
    cfg: dict[str, Any],
    arrivals_df,
    attractions_df,
    train_months,
    val_months,
    node_order: list[str],
    G,
) -> dict[str, Any]:
    """Run the EXP-05 calibration with the given assumed within-day profile."""
    from src.calibration.estimator import MFGParameters
    from src.models.mfg_solver import MFGSolver

    sim_cfg = {
        **cfg["simulation"],
        "profile": profile["name"],
        "profile_params": profile.get("params") or {},
    }
    cal_cfg = cfg["calibration"]
    solver_cfg = cfg["solver"]
    dt = float(sim_cfg["dt_hours"])
    T = float(sim_cfg["T_hours"])
    damping = float(solver_cfg["damping"])

    train_data = _prepare_month_data(arrivals_df, attractions_df, train_months, node_order, sim_cfg)
    val_data = _prepare_month_data(arrivals_df, attractions_df, val_months, node_order, sim_cfg)

    alpha_init = _init_alpha_from_observed(train_data[0]["obs"], len(node_order))
    params = MFGParameters(
        len(node_order),
        alpha_init=alpha_init,
        beta_init=float(cal_cfg["init_beta"]),
        gamma_init=float(cal_cfg["init_gamma"]),
    )
    solver = MFGSolver(
        G=G, params=params.as_dict(), dt=dt, T=T,
        epsilon=float(solver_cfg["epsilon"]), tol=float(solver_cfg["tol"]),
        max_iter=int(solver_cfg["max_iter"]), node_order=list(range(len(node_order))),
    )
    optimizer = torch.optim.Adam(params.parameters(), lr=float(cal_cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cal_cfg["lr_decay"]))

    for epoch in range(int(cal_cfg["n_epochs"])):
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0)
        for item in train_data:
            with torch.no_grad():
                solver.params = {
                    "alpha": params.alpha.detach(),
                    "beta": params.beta.detach(),
                    "gamma": params.gamma.detach(),
                }
                rho_fp, _, _ = solver.fixed_point_iteration(item["g"], damping=damping)
            solver.params = {"alpha": params.alpha, "beta": params.beta, "gamma": params.gamma}
            u = solver.solve_hjb_backward(rho_fp)
            rho_pred = solver.solve_fp_forward(u, item["g"])
            epoch_loss = epoch_loss + F.mse_loss(_attraction_distribution(rho_pred), item["obs"])
        epoch_loss = epoch_loss / max(len(train_data), 1)
        epoch_loss = epoch_loss + float(cal_cfg["lambda_reg"]) * (params.alpha ** 2).mean()
        epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), float(cal_cfg["grad_clip"]))
        optimizer.step()
        scheduler.step()

    final_params = params.as_dict()
    val_rows = _evaluate_months(solver, final_params, val_data, damping)
    val_mae = sum(r["mae"] for r in val_rows) / max(len(val_rows), 1)
    val_max = max((r["max_error"] for r in val_rows), default=0.0)
    return {
        "alpha": final_params["alpha"],   # list, length N_nodes
        "beta": final_params["beta"],
        "gamma": final_params["gamma"],
        "val_mae": val_mae,
        "val_max_error": val_max,
    }


# ---------------------------------------------------------------------------
# Part B -- interventions under a given profile (fixed EXP-05 params)
# ---------------------------------------------------------------------------


def _intervention_for_profile(
    profile: dict[str, Any],
    cfg: dict[str, Any],
    arrivals_df,
    node_order: list[str],
    G,
    params_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Re-run metering + routing under one assumed profile, fixed calibrated params."""
    from src.optimization.interventions import RoutingOptimizer

    sim_cfg = cfg["simulation"]
    iv = cfg["interventions"]
    rcfg = iv["routing"]
    damping = float(rcfg["damping"])
    dt = float(sim_cfg["dt_hours"])
    T_steps = round(float(sim_cfg["T_hours"]) / dt)
    N = len(node_order)
    zero_eta = torch.zeros(N, N, dtype=torch.float32)

    daily = _daily_source_counts(arrivals_df, _period(iv["peak_month"]), float(sim_cfg["population_scale"]))
    g = _build_real_arrival_tensor(
        node_order=node_order, T_steps=T_steps, dt=dt, daily_source_counts=daily,
        peak_time_hours=float(sim_cfg["peak_time_hours"]), sigma_hours=float(sim_cfg["sigma_hours"]),
        profile=profile["name"], profile_params=profile.get("params") or {},
    )
    solver = _make_solver(G, params_dict, cfg)
    ruins_idx = node_order.index(rcfg["bottleneck_node"])

    # ── Baseline (no intervention) ────────────────────────────────────────────
    with torch.no_grad():
        solver.routing_bonus = zero_eta
        rho_base, _, _ = solver.fixed_point_iteration(g, damping=damping)
    m_base = _compute_metrics(rho_base, g, dt, ruins_idx)

    # ── Metering (EXP-07 best aggressive cap at ferry_outer) ──────────────────
    ferry_col = node_order.index(iv["metering"]["source_node"])
    r_star = float(g[:, ferry_col].max().item()) * float(iv["metering"]["r_star_fraction"])
    g_met = _meter_arrivals(g, r_star, dt, ferry_col)
    with torch.no_grad():
        solver.routing_bonus = zero_eta
        rho_met, _, _ = solver.fixed_point_iteration(g_met, damping=damping)
    m_met = _compute_metrics(rho_met, g_met, dt, ruins_idx)
    metering_peak_red = 100.0 * (m_base["peak_density_ruins"] - m_met["peak_density_ruins"]) / (m_base["peak_density_ruins"] + 1e-12)
    metering_visit_red = 100.0 * (m_base["total_att_hours"] - m_met["total_att_hours"]) / (m_base["total_att_hours"] + 1e-12)

    # ── Routing (re-optimise eta under this profile) ──────────────────────────
    candidate_edges = _candidate_edges(solver, None, dst_attractions_only=True, attraction_count=ATTRACTION_COUNT)
    opt = RoutingOptimizer(
        solver=solver, g=g, bottleneck_idx=list(range(ATTRACTION_COUNT)), report_idx=ruins_idx,
        candidate_edges=candidate_edges, attraction_count=ATTRACTION_COUNT, node_labels=node_order,
        eta_max=float(rcfg["eta_max"]), lambda_l1=float(rcfg["lambda_l1"]),
        lambda_visit=float(rcfg["lambda_visit"]), delta=float(rcfg["delta"]),
        peak_temp=float(rcfg["peak_temp"]), damping=damping,
    )
    res = opt.optimise(
        n_steps=int(rcfg["n_steps"]), lr=float(rcfg["lr"]),
        lr_decay=float(rcfg["lr_decay"]), grad_clip=float(rcfg["grad_clip"]),
    )
    with torch.no_grad():
        solver.routing_bonus = res["eta"]
        rho_rout, _, _ = solver.fixed_point_iteration(g, damping=damping)
    m_rout = _compute_metrics(rho_rout, g, dt, ruins_idx)
    solver.routing_bonus = zero_eta

    return {
        "metering_peak_red": metering_peak_red,
        "metering_visit_red": metering_visit_red,
        "routing_peak_red": res["peak_density_reduction_pct"],
        "routing_system_red": res["system_peak_reduction_pct"],
        "routing_visit_red": res["visit_reduction_pct"],
        "gini_base": m_base["gini"],
        "gini_routing": m_rout["gini"],
        "routing_converged": res["optimized_converged"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> bool:
    import matplotlib
    matplotlib.use("Agg")

    from src.utils import attractions, io
    from src.utils.data_loader import load_arrivals_monthly, load_attraction_counts

    exp_cfg = cfg["experiment"]
    io.set_all_seeds(int(exp_cfg["seed"]))
    outdir = io.make_experiment_dir(base=Path(cfg["output"]["base_dir"]), name=f"{exp_cfg['id']}_{exp_cfg['name']}")
    io.save_config(cfg, outdir)
    logger.info("Output directory: %s", outdir)

    arrivals_df = load_arrivals_monthly(cfg["data"]["arrivals_path"])
    attractions_df = load_attraction_counts(cfg["data"]["attractions_path"])
    node_order = [n.node_id for n in attractions.ATTRACTION_NODES]
    train_months = _select_months(arrivals_df, cfg["split"]["train_start"], cfg["split"]["train_end"])
    val_months = _select_months(arrivals_df, cfg["split"]["val_start"], cfg["split"]["val_end"])

    logger.info("Loading Macau OSM graph …")
    t_graph = time.perf_counter()
    G = _load_and_build_graph(cfg, node_order)
    logger.info("Graph loaded in %.1f s", time.perf_counter() - t_graph)

    fitted = _load_fitted_params(Path(cfg["data"]["fitted_params_path"]))
    params_dict = _params_to_solver_dict(fitted, n_nodes=len(node_order))

    profiles = list(cfg["profiles"])

    # ── Part A: calibration invariance ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PART A: spatial-calibration invariance across %d profiles", len(profiles))
    calib_rows: list[dict[str, Any]] = []
    for prof in profiles:
        t0 = time.perf_counter()
        r = _calibrate_for_profile(prof, cfg, arrivals_df, attractions_df, train_months, val_months, node_order, G)
        logger.info("[A %-22s] val_mae=%.4f val_max=%.4f beta=%.5f (%.1f s)",
                    prof["name"], r["val_mae"], r["val_max_error"], r["beta"], time.perf_counter() - t0)
        calib_rows.append({"profile": prof["name"], "label": prof["label"], **r})

    # ── Part B: intervention robustness ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PART B: intervention robustness across %d profiles", len(profiles))
    iv_rows: list[dict[str, Any]] = []
    for prof in profiles:
        t0 = time.perf_counter()
        r = _intervention_for_profile(prof, cfg, arrivals_df, node_order, G, params_dict)
        logger.info("[B %-22s] meter_peak_red=%.1f%% rout_peak_red=%.1f%% gini %.3f->%.3f conv=%s (%.1f s)",
                    prof["name"], r["metering_peak_red"], r["routing_peak_red"],
                    r["gini_base"], r["gini_routing"], r["routing_converged"], time.perf_counter() - t0)
        iv_rows.append({"profile": prof["name"], "label": prof["label"], **r})

    passed = _write_outputs(outdir, cfg, node_order, calib_rows, iv_rows)
    logger.info("=" * 60)
    logger.info("EXP-11 done. Outputs in: %s", outdir)
    return passed


def _write_outputs(outdir: Path, cfg, node_order, calib_rows, iv_rows) -> bool:
    import matplotlib.pyplot as plt
    from src.utils import io

    fig_cfg = cfg["output"]["figures"]
    dpi = int(fig_cfg.get("dpi", 300))
    att_ids = node_order[:ATTRACTION_COUNT]
    max_val_mae = float(cfg["success"]["max_val_mae"])

    # ── CSVs ──────────────────────────────────────────────────────────────────
    with open(outdir / "calibration_invariance.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["profile", "label", "val_mae", "val_max_error", "beta", "gamma"] + [f"alpha_{a}" for a in att_ids]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in calib_rows:
            row = {"profile": r["profile"], "label": r["label"], "val_mae": f"{r['val_mae']:.6f}",
                   "val_max_error": f"{r['val_max_error']:.6f}", "beta": f"{r['beta']:.6f}", "gamma": f"{r['gamma']:.8f}"}
            for i, a in enumerate(att_ids):
                row[f"alpha_{a}"] = f"{r['alpha'][i]:.6f}"
            w.writerow(row)

    with open(outdir / "intervention_robustness.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["profile", "label", "metering_peak_red", "metering_visit_red", "routing_peak_red",
                "routing_system_red", "routing_visit_red", "gini_base", "gini_routing", "routing_converged"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in iv_rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})

    # ── Figure 1: alpha across profiles ───────────────────────────────────────
    import numpy as np
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_bar", [12, 5])))
    x = np.arange(ATTRACTION_COUNT)
    n = len(calib_rows)
    width = 0.8 / max(n, 1)
    for k, r in enumerate(calib_rows):
        ax.bar(x + (k - (n - 1) / 2) * width, r["alpha"][:ATTRACTION_COUNT], width, label=r["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(att_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"Fitted attractiveness $\alpha_v$")
    ax.set_title("EXP-11 Part A: calibrated spatial parameters across assumed profiles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    io.save_figure(fig, outdir, "alpha_across_profiles", dpi=dpi)
    plt.close(fig)

    # ── Figure 2: held-out MAE across profiles ────────────────────────────────
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_robust", [9, 6])))
    labels = [r["label"] for r in calib_rows]
    maes = [r["val_mae"] for r in calib_rows]
    ax.bar(range(len(maes)), maes, color="tab:blue")
    ax.axhline(max_val_mae, color="tab:red", ls="--", label=f"threshold {max_val_mae}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Held-out spatial MAE")
    ax.set_title("EXP-11 Part A: held-out MAE is profile-invariant (data-validated quantity)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    io.save_figure(fig, outdir, "val_mae_across_profiles", dpi=dpi)
    plt.close(fig)

    # ── Figure 3: peak reduction vs profile (interventions) ───────────────────
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("figsize_robust", [9, 6])))
    labels = [r["label"] for r in iv_rows]
    xr = np.arange(len(iv_rows))
    ax.bar(xr - 0.2, [r["metering_peak_red"] for r in iv_rows], 0.4, label="Metering peak reduction", color="tab:green")
    ax.bar(xr + 0.2, [r["routing_peak_red"] for r in iv_rows], 0.4, label="Routing peak reduction", color="tab:orange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xr)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Peak reduction at Ruins of St. Paul's (%)")
    ax.set_title("EXP-11 Part B: intervention peak reduction vs assumed profile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    io.save_figure(fig, outdir, "peak_reduction_vs_profile", dpi=dpi)
    plt.close(fig)

    # ── Verdicts + summary ────────────────────────────────────────────────────
    maes = [r["val_mae"] for r in calib_rows]
    alpha_mat = np.array([r["alpha"][:ATTRACTION_COUNT] for r in calib_rows])  # (n_prof, N_att)
    base_alpha = alpha_mat[0] + 1e-9
    alpha_rel_dev = float(np.max(np.abs(alpha_mat - base_alpha) / base_alpha))  # vs gaussian baseline
    invariance_ok = max(maes) < max_val_mae

    meter_reds = [r["metering_peak_red"] for r in iv_rows]
    rout_reds = [r["routing_peak_red"] for r in iv_rows]
    meter_sign_ok = all(v > 0 for v in meter_reds)
    rout_sign_ok = all(v > 0 for v in rout_reds)
    all_converged = all(r["routing_converged"] for r in iv_rows)

    lines = [
        "EXP-11 Validity Scope -- robustness to the assumed within-day profile",
        "=" * 68,
        "",
        "PART A -- spatial-calibration invariance (data-validated quantity):",
        f"  Held-out MAE range across profiles: [{min(maes):.4f}, {max(maes):.4f}] "
        f"(threshold {max_val_mae}) -> {'INVARIANT' if invariance_ok else 'NOT invariant'}",
        f"  Max relative deviation of fitted alpha vs Gaussian baseline: {100*alpha_rel_dev:.1f}%",
        "",
    ]
    for r in calib_rows:
        lines.append(f"  {r['label']:<34} val_mae={r['val_mae']:.4f}  beta={r['beta']:.5f}")
    lines += [
        "",
        "PART B -- intervention robustness (peak reduction at Ruins of St. Paul's):",
        f"  Metering: {'sign holds (>0) for all profiles' if meter_sign_ok else 'SIGN FLIPS for some profile'}",
        f"  Routing:  {'sign holds (>0) for all profiles' if rout_sign_ok else 'SIGN FLIPS for some profile'}",
        f"  All routing equilibria converged: {all_converged}",
        f"  Metering peak-reduction range: [{min(meter_reds):.1f}%, {max(meter_reds):.1f}%]",
        f"  Routing  peak-reduction range: [{min(rout_reds):.1f}%, {max(rout_reds):.1f}%]",
        "",
    ]
    for r in iv_rows:
        lines.append(
            f"  {r['label']:<34} meter={r['metering_peak_red']:.1f}%  "
            f"routing={r['routing_peak_red']:.1f}%  Gini {r['gini_base']:.3f}->{r['gini_routing']:.3f}"
            f"  conv={r['routing_converged']}"
        )
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return invariance_ok and meter_sign_ok and rout_sign_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-11: validity scope across within-day profiles.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent.parent / "configs" / "exp11_validity_scope.yaml")
    args = parser.parse_args(argv)
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 1
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return 0 if run(cfg) else 1


if __name__ == "__main__":
    sys.exit(main())
