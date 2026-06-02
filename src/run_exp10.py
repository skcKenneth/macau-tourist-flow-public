"""EXP-10: Fair baselines + ablation -- does the MFG earn its complexity?

Goal C of the research-hardening program. Every model below is calibrated and
evaluated on the **same real-DSEC held-out split** as the MFG (EXP-05), predicting
the monthly spatial attraction distribution, so the comparison is apples-to-apples
(unlike the original EXP-02 random walk, which used synthetic arrivals + a proxy
target).

Models compared (held-out spatial MAE):
- Gravity model (distance-decay spatial interaction)        -- src/models/baselines.py
- Multinomial logit (static discrete choice, NO congestion) -- src/models/baselines.py
- Random walk (non-strategic), re-run on the EXP-05 split   -- src/models/baseline_random.py
- MFG with beta=0 (no congestion coupling)  -- ablation
- Full MFG (EXP-05 fit)                      -- headline

Writeup: docs/09_baselines.md.
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
    _observed_distribution_for_year,
    _period,
    _prepare_month_data,
    _select_months,
)
from src.run_exp07 import _load_fitted_params, _make_solver, _params_to_solver_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_IDS = ["ferry_outer", "border_gate", "hotel_belt"]


# ---------------------------------------------------------------------------
# Data assembly for the static (gravity / MNL) models
# ---------------------------------------------------------------------------


def _static_items(arrivals_df, attractions_df, months, node_order, scale) -> list[dict]:
    """Per-month {source_weights (n_src,), obs (n_att,)} for the static models."""
    items = []
    for m in months:
        daily = _daily_source_counts(arrivals_df, m, scale)
        w = torch.tensor([float(daily.get(s, 0.0)) for s in SOURCE_IDS], dtype=torch.float32)
        obs = _observed_distribution_for_year(attractions_df, node_order, m.year)
        items.append({"source_weights": w, "obs": obs})
    return items


def _dist_source_attraction(solver, node_order) -> torch.Tensor:
    """(n_sources, n_attractions) walking-distance matrix in metres."""
    src_idx = [node_order.index(s) for s in SOURCE_IDS]
    return solver.D[src_idx, :ATTRACTION_COUNT].clone()


# ---------------------------------------------------------------------------
# MFG calibration (full, and the beta=0 ablation)
# ---------------------------------------------------------------------------


def _calibrate_mfg(
    cfg, arrivals_df, attractions_df, train_months, val_months, node_order, G,
    freeze_beta_zero: bool = False,
) -> float:
    """Calibrate the MFG (optionally with beta frozen at ~0) and return val MAE."""
    from src.calibration.estimator import MFGParameters
    from src.models.mfg_solver import MFGSolver

    sim_cfg = cfg["simulation"]
    cal_cfg = cfg["calibration"]
    solver_cfg = cfg["solver"]
    dt = float(sim_cfg["dt_hours"]); T = float(sim_cfg["T_hours"]); damping = float(solver_cfg["damping"])

    train_data = _prepare_month_data(arrivals_df, attractions_df, train_months, node_order, sim_cfg)
    val_data = _prepare_month_data(arrivals_df, attractions_df, val_months, node_order, sim_cfg)

    alpha_init = _init_alpha_from_observed(train_data[0]["obs"], len(node_order))
    params = MFGParameters(
        len(node_order), alpha_init=alpha_init,
        beta_init=float(cal_cfg["init_beta"]), gamma_init=float(cal_cfg["init_gamma"]),
    )
    if freeze_beta_zero:
        with torch.no_grad():
            params.log_beta.copy_(torch.tensor(1e-8).log())  # beta ~ 1e-8 ~ 0
        params.log_beta.requires_grad_(False)

    solver = MFGSolver(
        G=G, params=params.as_dict(), dt=dt, T=T,
        epsilon=float(solver_cfg["epsilon"]), tol=float(solver_cfg["tol"]),
        max_iter=int(solver_cfg["max_iter"]), node_order=list(range(len(node_order))),
    )
    trainable = [p for p in params.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(cal_cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cal_cfg["lr_decay"]))

    for _ in range(int(cal_cfg["n_epochs"])):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for item in train_data:
            with torch.no_grad():
                solver.params = {"alpha": params.alpha.detach(), "beta": params.beta.detach(), "gamma": params.gamma.detach()}
                rho_fp, _, _ = solver.fixed_point_iteration(item["g"], damping=damping)
            solver.params = {"alpha": params.alpha, "beta": params.beta, "gamma": params.gamma}
            u = solver.solve_hjb_backward(rho_fp)
            rho_pred = solver.solve_fp_forward(u, item["g"])
            loss = loss + F.mse_loss(_attraction_distribution(rho_pred), item["obs"])
        loss = loss / max(len(train_data), 1) + float(cal_cfg["lambda_reg"]) * (params.alpha ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(cal_cfg["grad_clip"]))
        optimizer.step()
        scheduler.step()

    val_rows = _evaluate_months(solver, params.as_dict(), val_data, damping)
    return sum(r["mae"] for r in val_rows) / max(len(val_rows), 1)


# ---------------------------------------------------------------------------
# Random walk on the EXP-05 protocol
# ---------------------------------------------------------------------------


def _random_walk_val_mae(cfg, arrivals_df, attractions_df, val_months, node_order, G) -> float:
    from src.models.baseline_random import RandomWalkBaseline

    sim_cfg = cfg["simulation"]; rw_cfg = cfg["random_walk"]
    dt = float(sim_cfg["dt_hours"]); T_steps = round(float(sim_cfg["T_hours"]) / dt)
    maes = []
    for m in val_months:
        daily = _daily_source_counts(arrivals_df, m, float(sim_cfg["population_scale"]))
        # Scale to a fixed tourist count for the random walk (shape is what matters).
        total = sum(daily.values()) + 1e-9
        daily_scaled = {k: v / total * float(rw_cfg["n_tourists"]) for k, v in daily.items()}
        g = _build_real_arrival_tensor(
            node_order, T_steps, dt, daily_scaled,
            float(sim_cfg["peak_time_hours"]), float(sim_cfg["sigma_hours"]),
        )
        rw = RandomWalkBaseline(
            G=G, arrival_rates=g, dt=dt, node_order=list(range(len(node_order))),
            exit_rate=float(rw_cfg["exit_rate_per_step"]),
        )
        rho = rw.simulate(seed=int(cfg["experiment"]["seed"]))
        pred = _attraction_distribution(rho)
        obs = _observed_distribution_for_year(attractions_df, node_order, m.year)
        maes.append(float((pred - obs).abs().mean().item()))
    return sum(maes) / max(len(maes), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.models.baselines import (
        GravityModel, MultinomialLogitModel, evaluate_static_model, fit_static_model,
    )
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
    scale = float(cfg["simulation"]["population_scale"])

    logger.info("Loading graph …")
    G = _load_and_build_graph(cfg, node_order)
    fitted = _load_fitted_params(Path(cfg["data"]["fitted_params_path"]))
    params_dict = _params_to_solver_dict(fitted, n_nodes=len(node_order))
    solver = _make_solver(G, params_dict, cfg)
    D_sa = _dist_source_attraction(solver, node_order)

    results: dict[str, float] = {}

    # ── Static baselines (gravity, MNL) ───────────────────────────────────────
    train_items = _static_items(arrivals_df, attractions_df, train_months, node_order, scale)
    val_items = _static_items(arrivals_df, attractions_df, val_months, node_order, scale)
    sf = cfg["static_fit"]
    grav = GravityModel(ATTRACTION_COUNT, D_sa)
    fit_static_model(grav, train_items, int(sf["n_epochs"]), float(sf["lr"]), float(sf["lr_decay"]))
    results["Gravity"] = evaluate_static_model(grav, val_items)
    logger.info("Gravity val MAE = %.4f", results["Gravity"])

    mnl = MultinomialLogitModel(ATTRACTION_COUNT, D_sa)
    fit_static_model(mnl, train_items, int(sf["n_epochs"]), float(sf["lr"]), float(sf["lr_decay"]))
    results["Multinomial logit (no congestion)"] = evaluate_static_model(mnl, val_items)
    logger.info("MNL val MAE = %.4f", results["Multinomial logit (no congestion)"])

    # ── Random walk on the EXP-05 protocol ────────────────────────────────────
    results["Random walk (non-strategic)"] = _random_walk_val_mae(
        cfg, arrivals_df, attractions_df, val_months, node_order, G)
    logger.info("Random walk val MAE = %.4f", results["Random walk (non-strategic)"])

    # ── MFG beta=0 ablation ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    results["MFG (beta=0, no congestion)"] = _calibrate_mfg(
        cfg, arrivals_df, attractions_df, train_months, val_months, node_order, G,
        freeze_beta_zero=True)
    logger.info("MFG beta=0 val MAE = %.4f (%.1f s)", results["MFG (beta=0, no congestion)"], time.perf_counter() - t0)

    # ── Full MFG (EXP-05 fit) ─────────────────────────────────────────────────
    val_data = _prepare_month_data(arrivals_df, attractions_df, val_months, node_order, cfg["simulation"])
    full_rows = _evaluate_months(solver, fitted, val_data, float(cfg["solver"]["damping"]))
    results["Full MFG (EXP-05)"] = sum(r["mae"] for r in full_rows) / max(len(full_rows), 1)
    logger.info("Full MFG val MAE = %.4f", results["Full MFG (EXP-05)"])

    _write_outputs(outdir, cfg, results)
    logger.info("EXP-10 done. Outputs in: %s", outdir)
    return True


def _write_outputs(outdir: Path, cfg, results: dict[str, float]) -> None:
    import matplotlib.pyplot as plt
    from src.utils import io

    ordered = sorted(results.items(), key=lambda kv: kv[1], reverse=True)  # worst first
    with open(outdir / "model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "held_out_mae"])
        w.writeheader()
        for name, mae in ordered:
            w.writerow({"model": name, "held_out_mae": f"{mae:.6f}"})

    fig, ax = plt.subplots(figsize=tuple(cfg["output"]["figures"].get("figsize_bar", [9, 6])))
    names = [k for k, _ in ordered]
    maes = [v for _, v in ordered]
    colors = ["tab:blue" if "Full MFG" not in n else "tab:red" for n in names]
    ax.barh(range(len(names)), maes, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out spatial MAE (lower is better)")
    ax.set_title("EXP-10: model comparison on the EXP-05 held-out split")
    fig.tight_layout()
    io.save_figure(fig, outdir, "model_comparison", dpi=int(cfg["output"]["figures"].get("dpi", 300)))
    plt.close(fig)

    full = results.get("Full MFG (EXP-05)", float("nan"))
    beta0 = results.get("MFG (beta=0, no congestion)", float("nan"))
    lines = [
        "EXP-10 Baselines + Ablation -- held-out spatial MAE (lower is better)",
        "=" * 68, "",
    ]
    for name, mae in sorted(results.items(), key=lambda kv: kv[1]):
        lines.append(f"  {name:<34} MAE = {mae:.4f}")
    lines += [
        "",
        f"Congestion-coupling ablation: full MFG {full:.4f} vs MFG(beta=0) {beta0:.4f} "
        f"-> congestion buys {beta0 - full:+.4f} MAE.",
        "",
        "Note: all models calibrated + evaluated on the identical EXP-05 real-DSEC split.",
    ]
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-10: fair baselines + ablation.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent.parent / "configs" / "exp10_baselines.yaml")
    args = parser.parse_args(argv)
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 1
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return 0 if run(cfg) else 1


if __name__ == "__main__":
    sys.exit(main())
