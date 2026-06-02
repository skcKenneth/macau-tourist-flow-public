"""EXP-05: Real-DSEC calibration with MGTO attraction proxy.

Uses real DSEC monthly visitor arrivals (2024-01 through 2026-04) and the
current MGTO/proxy attraction counts to calibrate the 13-node Macau MFG model.

This first real-data run is intentionally labelled "real DSEC + MGTO proxy":
the arrival side is official DSEC data, while attraction-side observations use
``data/processed/attractions.parquet`` and may have confidence="estimate" until
official MGTO per-attraction counts are entered.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from calendar import monthrange
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


ATTRACTION_COUNT = 10


def _period(value: str | pd.Period) -> pd.Period:
    """Return a monthly pandas Period."""
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(str(value), freq="M")


def _select_months(
    arrivals: pd.DataFrame,
    start: str | pd.Period,
    end: str | pd.Period,
) -> list[pd.Period]:
    """Select sorted monthly periods available in [start, end]."""
    start_p = _period(start)
    end_p = _period(end)
    months = sorted(arrivals["year_month"].unique())
    return [m for m in months if start_p <= _period(m) <= end_p]


def _days_in_month(month: str | pd.Period) -> int:
    """Number of calendar days in a monthly period."""
    p = _period(month)
    return monthrange(p.year, p.month)[1]


def _map_dsec_transit_counts(month_df: pd.DataFrame) -> dict[str, int]:
    """Map DSEC transit-point rows to the three model source nodes.

    The DSEC file contains aggregate rows (``by_land``, ``by_sea``) as well as
    named facilities. To avoid double-counting, this maps:
    - border_gate = DSEC border_gate
    - ferry_outer = all sea arrivals = DSEC by_sea
    - hotel_belt = airport + other land arrivals
    """
    counts = {
        str(row.transit_point): int(row.count)
        for row in month_df.itertuples(index=False)
    }
    border_gate = counts.get("border_gate", 0)
    by_land = counts.get("by_land", 0)
    by_sea = counts.get("by_sea", 0)
    by_air = counts.get("by_air", 0)
    other_land = max(by_land - border_gate, 0)
    return {
        "border_gate": border_gate,
        "ferry_outer": by_sea,
        "hotel_belt": by_air + other_land,
    }


def _daily_source_counts(
    arrivals: pd.DataFrame,
    month: str | pd.Period,
    population_scale: float = 1.0,
) -> dict[str, float]:
    """Return average operating-day arrivals for the model source nodes."""
    p = _period(month)
    month_df = arrivals[arrivals["year_month"].apply(_period) == p]
    if month_df.empty:
        raise ValueError(f"No DSEC arrivals found for month {p}")
    mapped = _map_dsec_transit_counts(month_df)
    days = _days_in_month(p)
    return {k: (v / days) * population_scale for k, v in mapped.items()}


def _build_real_arrival_tensor(
    node_order: list[str],
    T_steps: int,
    dt: float,
    daily_source_counts: dict[str, float],
    peak_time_hours: float,
    sigma_hours: float,
    profile: str = "gaussian",
    profile_params: dict | None = None,
) -> torch.Tensor:
    """Build a daily arrival tensor from DSEC-derived source counts.

    The daily *volume* per source node comes from data; the within-day *shape*
    is an ASSUMED profile (see ``src.utils.arrival_profiles`` and
    docs/08_validity_scope.md). The default ``profile="gaussian"`` reproduces the
    project's original single-peak shape exactly, so EXP-05/07/08 are unchanged.

    Args:
        node_order: Ordered node ids (tensor column order).
        T_steps: Number of discrete time steps in the operating day.
        dt: Time step in hours.
        daily_source_counts: Daily arrival count per source node id.
        peak_time_hours: Peak hour for the default Gaussian profile.
        sigma_hours: Sharpness for the default Gaussian profile.
        profile: Name of the within-day profile (see ``arrival_profiles.PROFILES``).
        profile_params: Optional overrides for the chosen profile's parameters.

    Returns:
        Float32 tensor (T_steps, N_nodes); each source column is the daily count
        times the normalised within-day profile.
    """
    from src.utils import arrival_profiles

    params = dict(profile_params or {})
    if profile == "gaussian":
        params.setdefault("peak_time_hours", peak_time_hours)
        params.setdefault("sigma_hours", sigma_hours)
    g_norm = arrival_profiles.weights(profile, T_steps, dt, **params)

    arrivals = torch.zeros(T_steps, len(node_order), dtype=torch.float32)
    node_to_idx = {nid: i for i, nid in enumerate(node_order)}
    for node_id, count in daily_source_counts.items():
        if node_id not in node_to_idx:
            raise ValueError(f"Source node {node_id!r} not in node_order")
        arrivals[:, node_to_idx[node_id]] = float(count) * g_norm
    return arrivals


def _observed_distribution_for_year(
    attractions: pd.DataFrame,
    node_order: list[str],
    year: int,
) -> torch.Tensor:
    """Build normalized attraction-only observed distribution for ``year``."""
    attraction_ids = node_order[:ATTRACTION_COUNT]
    available_years = sorted(int(y) for y in attractions["year"].unique())
    selected_years = [y for y in available_years if y <= year]
    selected_year = selected_years[-1] if selected_years else available_years[0]
    sub = attractions[attractions["year"].astype(int) == selected_year]
    by_node = {str(r.node_id): int(r.annual_visitors) for r in sub.itertuples()}
    missing = [nid for nid in attraction_ids if nid not in by_node]
    if missing:
        raise ValueError(f"Attraction counts missing node_ids: {missing}")
    obs = torch.tensor([by_node[nid] for nid in attraction_ids], dtype=torch.float32)
    return obs / obs.sum()


def _attraction_distribution(rho: torch.Tensor) -> torch.Tensor:
    """Convert a density trajectory to normalized cumulative attraction share."""
    att = rho[:, :ATTRACTION_COUNT].sum(dim=0)
    total = att.sum()
    if float(total.item()) <= 1e-12:
        return torch.full((ATTRACTION_COUNT,), 1.0 / ATTRACTION_COUNT)
    return att / total


def _mae(pred: torch.Tensor, obs: torch.Tensor) -> float:
    return float((pred - obs).abs().mean().item())


def _init_alpha_from_observed(obs: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Initialize alpha from observed attraction proportions."""
    alpha = torch.full((n_nodes,), 1e-6, dtype=torch.float32)
    scaled = obs / obs.mean()
    alpha[:ATTRACTION_COUNT] = scaled.clamp(min=1e-4)
    return alpha


def _load_and_build_graph(cfg: dict[str, Any], node_order: list[str]):
    """Load OSM graph and build the 13-node simulation subgraph."""
    from src.run_exp02 import _build_subgraph
    from src.utils import attractions, graph_loader

    graph_cfg = cfg["graph"]
    snap_cfg = cfg["snapping"]
    G_osm = graph_loader.load_macau_graph(
        bbox=graph_cfg.get("bbox"),
        network_type="walk",
        simplify=True,
        retain_all=False,
        consolidate_tolerance_m=graph_cfg.get("consolidate_tolerance_m", 15.0),
        walk_speed_kmh=graph_cfg.get("walk_speed_kmh", 5.0),
        cache_path=Path(cfg["data"]["graph_cache_path"]),
    )
    snap_map = _snap_to_graph_haversine(
        G_osm,
        max_dist_attraction_m=snap_cfg.get("max_dist_attraction_m", 100.0),
        max_dist_transit_m=snap_cfg.get("max_dist_transit_m", 300.0),
    )
    return _build_subgraph(G_osm, snap_map, node_order)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in metres."""
    radius_m = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * radius_m * asin(sqrt(a))


def _snap_to_graph_haversine(
    G,
    max_dist_attraction_m: float = 100.0,
    max_dist_transit_m: float = 300.0,
) -> dict[str, int]:
    """Snap attraction registry nodes to graph nodes without pyproj/osmnx CRS."""
    from src.utils.attractions import ATTRACTION_NODES, NodeType

    graph_nodes = [
        (osm_id, float(data["y"]), float(data["x"]))
        for osm_id, data in G.nodes(data=True)
        if "x" in data and "y" in data
    ]
    if not graph_nodes:
        raise ValueError("OSM graph has no x/y node coordinates for snapping")

    snap_map: dict[str, int] = {}
    for node in ATTRACTION_NODES:
        best_id = None
        best_dist = float("inf")
        for osm_id, lat, lon in graph_nodes:
            dist = _haversine_m(node.lat, node.lon, lat, lon)
            if dist < best_dist:
                best_id = osm_id
                best_dist = dist
        limit = (
            max_dist_transit_m
            if node.node_type == NodeType.TRANSIT
            else max_dist_attraction_m
        )
        if best_dist > limit:
            logger.warning(
                "%s (%s) snapped %.1f m away (limit %.1f m)",
                node.node_id,
                node.name_en,
                best_dist,
                limit,
            )
        snap_map[node.node_id] = best_id
    return snap_map


def _prepare_month_data(
    arrivals_df: pd.DataFrame,
    attractions_df: pd.DataFrame,
    months: list[pd.Period],
    node_order: list[str],
    sim_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Prepare arrival tensors and observed distributions for each month."""
    dt = float(sim_cfg["dt_hours"])
    T_steps = round(float(sim_cfg["T_hours"]) / dt)
    prepared = []
    for month in months:
        daily_counts = _daily_source_counts(
            arrivals_df,
            month,
            population_scale=float(sim_cfg.get("population_scale", 1.0)),
        )
        g = _build_real_arrival_tensor(
            node_order=node_order,
            T_steps=T_steps,
            dt=dt,
            daily_source_counts=daily_counts,
            peak_time_hours=float(sim_cfg["peak_time_hours"]),
            sigma_hours=float(sim_cfg["sigma_hours"]),
            profile=str(sim_cfg.get("profile", "gaussian")),
            profile_params=sim_cfg.get("profile_params"),
        )
        obs = _observed_distribution_for_year(attractions_df, node_order, month.year)
        prepared.append({"month": month, "g": g, "obs": obs, "daily_counts": daily_counts})
    return prepared


def _evaluate_months(solver, params: dict[str, Any], month_data: list[dict[str, Any]], damping: float) -> list[dict[str, Any]]:
    """Run fixed-point solver for each month and return prediction metrics."""
    solver.params = solver._parse_params(params)
    rows = []
    with torch.no_grad():
        for item in month_data:
            rho, _, info = solver.fixed_point_iteration(item["g"], damping=damping)
            pred = _attraction_distribution(rho)
            obs = item["obs"]
            per_node = (pred - obs).abs()
            rows.append({
                "month": str(item["month"]),
                "mae": _mae(pred, obs),
                "max_error": float(per_node.max().item()),
                "pred": pred.detach().clone(),
                "obs": obs.detach().clone(),
                "n_iter": info["n_iter"],
                "converged": info["converged"],
                "final_residual": info["final_residual"],
            })
    return rows


def _fit_calibration(
    G,
    node_order: list[str],
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit MFG parameters on ``train_data`` and evaluate on ``val_data``.

    Shared by the main EXP-05 run and the month-resampling bootstrap so both
    use identical optimisation settings.

    Args:
        G: Simulation graph.
        node_order: Node id list (tensor column order).
        train_data: Prepared training months (``g``, ``obs``, ...).
        val_data: Prepared validation months.
        cfg: Full EXP-05 config (uses ``simulation``, ``solver``, ``calibration``).
        seed: Optional RNG seed (bootstrap replicates pass distinct seeds).

    Returns:
        Dict with ``final_params``, ``train_rows``, ``val_rows``, ``loss_history``,
        and ``solver``.
    """
    from src.calibration.estimator import MFGParameters
    from src.models.mfg_solver import MFGSolver
    from src.utils import io

    if seed is not None:
        io.set_all_seeds(int(seed))

    sim_cfg = cfg["simulation"]
    solver_cfg = cfg["solver"]
    cal_cfg = cfg["calibration"]
    dt = float(sim_cfg["dt_hours"])
    T_hours = float(sim_cfg["T_hours"])
    damping = float(solver_cfg["damping"])

    first_obs = train_data[0]["obs"] if train_data else val_data[0]["obs"]
    alpha_init = _init_alpha_from_observed(first_obs, len(node_order))
    params = MFGParameters(
        len(node_order),
        alpha_init=alpha_init,
        beta_init=float(cal_cfg["init_beta"]),
        gamma_init=float(cal_cfg["init_gamma"]),
    )
    solver = MFGSolver(
        G=G,
        params=params.as_dict(),
        dt=dt,
        T=T_hours,
        epsilon=float(solver_cfg["epsilon"]),
        tol=float(solver_cfg["tol"]),
        max_iter=int(solver_cfg["max_iter"]),
        node_order=list(range(len(node_order))),
    )
    optimizer = torch.optim.Adam(params.parameters(), lr=float(cal_cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(cal_cfg["lr_decay"])
    )
    loss_history: list[float] = []

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

            solver.params = {
                "alpha": params.alpha,
                "beta": params.beta,
                "gamma": params.gamma,
            }
            u = solver.solve_hjb_backward(rho_fp)
            rho_pred = solver.solve_fp_forward(u, item["g"])
            pred_dist = _attraction_distribution(rho_pred)
            epoch_loss = epoch_loss + torch.nn.functional.mse_loss(pred_dist, item["obs"])

        epoch_loss = epoch_loss / max(len(train_data), 1)
        epoch_loss = epoch_loss + float(cal_cfg["lambda_reg"]) * (params.alpha ** 2).mean()
        epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), float(cal_cfg["grad_clip"]))
        optimizer.step()
        scheduler.step()
        loss_history.append(float(epoch_loss.item()))
        if (epoch + 1) % int(cal_cfg.get("log_every", 50)) == 0:
            logger.info(
                "Epoch %d/%d loss=%.4e beta=%.5f gamma=%.6f",
                epoch + 1,
                int(cal_cfg["n_epochs"]),
                loss_history[-1],
                float(params.beta.detach()),
                float(params.gamma.detach()),
            )

    final_params = params.as_dict()
    train_rows = _evaluate_months(solver, final_params, train_data, damping)
    val_rows = _evaluate_months(solver, final_params, val_data, damping)
    val_mae = sum(r["mae"] for r in val_rows) / max(len(val_rows), 1)
    return {
        "final_params": final_params,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "loss_history": loss_history,
        "val_mae": val_mae,
        "solver": solver,
    }


def run(cfg: dict[str, Any]) -> bool:
    """Execute EXP-05 and return True if validation criteria are met."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils import attractions, io
    from src.utils.data_loader import load_arrivals_monthly, load_attraction_counts

    exp_cfg = cfg["experiment"]
    sim_cfg = cfg["simulation"]
    cal_cfg = cfg["calibration"]
    out_cfg = cfg["output"]
    hyp_cfg = cfg["hypothesis"]

    outdir = io.make_experiment_dir(
        base=Path(out_cfg["base_dir"]),
        name=f"{exp_cfg['id']}_{exp_cfg['name']}",
    )
    io.set_all_seeds(int(exp_cfg["seed"]))
    git_hash = io.get_git_hash()
    io.save_config(cfg, outdir)

    arrivals_df = load_arrivals_monthly(cfg["data"]["arrivals_path"])
    attractions_df = load_attraction_counts(cfg["data"]["attractions_path"])
    node_order = [n.node_id for n in attractions.ATTRACTION_NODES]
    train_months = _select_months(arrivals_df, cfg["split"]["train_start"], cfg["split"]["train_end"])
    val_months = _select_months(arrivals_df, cfg["split"]["val_start"], cfg["split"]["val_end"])
    logger.info("Train months: %s", [str(m) for m in train_months])
    logger.info("Validation months: %s", [str(m) for m in val_months])

    G = _load_and_build_graph(cfg, node_order)
    train_data = _prepare_month_data(arrivals_df, attractions_df, train_months, node_order, sim_cfg)
    val_data = _prepare_month_data(arrivals_df, attractions_df, val_months, node_order, sim_cfg)

    t0 = time.perf_counter()
    fit = _fit_calibration(G, node_order, train_data, val_data, cfg, seed=int(exp_cfg["seed"]))
    elapsed = time.perf_counter() - t0
    final_params = fit["final_params"]
    train_rows = fit["train_rows"]
    val_rows = fit["val_rows"]
    loss_history = fit["loss_history"]

    bootstrap_summary: dict[str, Any] | None = None
    boot_cfg = cfg.get("bootstrap")
    if boot_cfg and boot_cfg.get("enabled", False):
        from src.calibration.bootstrap import run_bootstrap_calibration

        boot_cal_cfg = {**cal_cfg, **boot_cfg.get("calibration_override", {})}
        boot_full_cfg = {**cfg, "calibration": boot_cal_cfg}
        base_seed = int(boot_cfg.get("seed", exp_cfg["seed"]))
        _rep_counter = [0]

        def _fit_replicate(resampled_train: list[dict[str, Any]]) -> dict[str, Any]:
            _rep_counter[0] += 1
            return _fit_calibration(
                G,
                node_order,
                resampled_train,
                val_data,
                boot_full_cfg,
                seed=base_seed + _rep_counter[0],
            )

        logger.info(
            "Bootstrap: %d replicates (n_epochs=%d per replicate) …",
            int(boot_cfg["n_bootstrap"]),
            int(boot_cal_cfg["n_epochs"]),
        )
        t_boot = time.perf_counter()
        bootstrap_summary = run_bootstrap_calibration(
            train_data,
            val_data,
            fit_fn=_fit_replicate,
            n_bootstrap=int(boot_cfg["n_bootstrap"]),
            seed=int(boot_cfg.get("resample_seed", base_seed)),
            ci_percent=tuple(boot_cfg.get("ci_percent", [5.0, 95.0])),
            attraction_count=ATTRACTION_COUNT,
            log_every=int(boot_cfg.get("log_every", 5)),
        )
        bootstrap_summary.pop("replicates", None)  # omit large list from YAML dump
        logger.info(
            "Bootstrap done in %.1f s: val_mae [%.4f, %.4f] beta [%.5f, %.5f]",
            time.perf_counter() - t_boot,
            bootstrap_summary["val_mae_p_low"],
            bootstrap_summary["val_mae_p_high"],
            bootstrap_summary["beta_p_low"],
            bootstrap_summary["beta_p_high"],
        )
        import numpy as np

        def _yaml_safe(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        boot_yaml = {
            k: _yaml_safe(v)
            for k, v in bootstrap_summary.items()
            if k != "replicates"
        }
        with open(outdir / "bootstrap_ci.yaml", "w", encoding="utf-8") as f:
            yaml.dump(boot_yaml, f, sort_keys=False)

    val_mae = sum(r["mae"] for r in val_rows) / max(len(val_rows), 1)
    val_max_error = max((r["max_error"] for r in val_rows), default=0.0)
    passed = (
        val_mae < float(hyp_cfg["validation_mae_threshold"])
        and val_max_error < float(hyp_cfg["max_attraction_error_threshold"])
    )

    _write_outputs(
        outdir=outdir,
        node_order=node_order,
        train_rows=train_rows,
        val_rows=val_rows,
        final_params=final_params,
        loss_history=loss_history,
        cfg=cfg,
        git_hash=git_hash,
        elapsed=elapsed,
        passed=passed,
        bootstrap_summary=bootstrap_summary,
    )
    return passed


def _write_outputs(
    outdir: Path,
    node_order: list[str],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    final_params: dict[str, Any],
    loss_history: list[float],
    cfg: dict[str, Any],
    git_hash: str,
    elapsed: float,
    passed: bool,
    bootstrap_summary: dict[str, Any] | None = None,
) -> None:
    """Write EXP-05 CSVs, figures, YAML params, and summary."""
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils import io

    dpi = int(cfg["output"].get("figures", {}).get("dpi", 300))
    attraction_ids = node_order[:ATTRACTION_COUNT]
    all_rows = [("train", r) for r in train_rows] + [("validation", r) for r in val_rows]

    with open(outdir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "month", "mae", "max_error", "n_iter",
                "converged", "final_residual",
            ],
        )
        writer.writeheader()
        for split, r in all_rows:
            writer.writerow({
                "split": split,
                "month": r["month"],
                "mae": f"{r['mae']:.6f}",
                "max_error": f"{r['max_error']:.6f}",
                "n_iter": r["n_iter"],
                "converged": r["converged"],
                "final_residual": f"{r['final_residual']:.6e}",
            })

    with open(outdir / "per_attraction_errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "month", "node_id", "observed", "predicted", "abs_error"],
        )
        writer.writeheader()
        for split, r in all_rows:
            for i, node_id in enumerate(attraction_ids):
                obs = float(r["obs"][i].item())
                pred = float(r["pred"][i].item())
                writer.writerow({
                    "split": split,
                    "month": r["month"],
                    "node_id": node_id,
                    "observed": f"{obs:.6f}",
                    "predicted": f"{pred:.6f}",
                    "abs_error": f"{abs(pred - obs):.6f}",
                })

    with open(outdir / "fitted_params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(final_params, f, sort_keys=False)

    val_mean_pred = torch.stack([r["pred"] for r in val_rows]).mean(dim=0)
    val_obs = val_rows[0]["obs"] if val_rows else torch.zeros(ATTRACTION_COUNT)
    x = list(range(ATTRACTION_COUNT))
    fig, ax = plt.subplots(figsize=cfg["output"]["figures"].get("figsize_bar", [12, 5]))
    ax.bar([i - 0.2 for i in x], val_obs.numpy(), width=0.4, label="Observed proxy")
    pred_vals = val_mean_pred[:ATTRACTION_COUNT].numpy()
    ax.bar([i + 0.2 for i in x], pred_vals, width=0.4, label="Predicted")
    if bootstrap_summary is not None:
        lo = bootstrap_summary["val_pred_p_low"]
        hi = bootstrap_summary["val_pred_p_high"]
        yerr = np.vstack([pred_vals - lo, hi - pred_vals])
        ax.errorbar(
            [i + 0.2 for i in x],
            pred_vals,
            yerr=yerr,
            fmt="none",
            ecolor="tab:orange",
            capsize=3,
            label=f"Bootstrap {bootstrap_summary['ci_percent'][0]:.0f}-"
            f"{bootstrap_summary['ci_percent'][1]:.0f}% CI",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(attraction_ids, rotation=45, ha="right")
    ax.set_ylabel("Normalized attraction share")
    ax.set_title("EXP-05 predicted vs observed attraction distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    io.save_figure(fig, outdir, "predicted_vs_observed", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=cfg["output"]["figures"].get("figsize_monthly", [10, 4]))
    ax.plot([r["month"] for r in train_rows], [r["mae"] for r in train_rows], marker="o", label="train")
    ax.plot([r["month"] for r in val_rows], [r["mae"] for r in val_rows], marker="o", label="validation")
    ax.axhline(float(cfg["hypothesis"]["validation_mae_threshold"]), color="red", ls="--", label="threshold")
    if bootstrap_summary is not None:
        ax.axhspan(
            bootstrap_summary["val_mae_p_low"],
            bootstrap_summary["val_mae_p_high"],
            color="tab:purple",
            alpha=0.12,
            label="Bootstrap val MAE CI",
        )
        ax.axhline(bootstrap_summary["val_mae_mean"], color="tab:purple", ls=":", lw=1.0)
    ax.set_ylabel("MAE")
    ax.set_title("EXP-05 monthly calibration MAE")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=8)
    fig.tight_layout()
    io.save_figure(fig, outdir, "monthly_mae", dpi=dpi)
    plt.close(fig)

    train_mae = sum(r["mae"] for r in train_rows) / max(len(train_rows), 1)
    val_mae = sum(r["mae"] for r in val_rows) / max(len(val_rows), 1)
    val_max = max((r["max_error"] for r in val_rows), default=0.0)
    lines = [
        f"EXP-05 Real-DSEC Calibration -- [{'PASS' if passed else 'FAIL'}]",
        "=" * 68,
        f"Train months: {train_rows[0]['month']} to {train_rows[-1]['month']} ({len(train_rows)} months)",
        f"Validation months: {val_rows[0]['month']} to {val_rows[-1]['month']} ({len(val_rows)} months)",
        "Data: real DSEC arrivals + MGTO/proxy attraction distribution",
        f"Train MAE: {train_mae:.4f}",
        f"Validation MAE: {val_mae:.4f}",
        f"Validation max attraction error: {val_max:.4f}",
        f"Thresholds: MAE < {cfg['hypothesis']['validation_mae_threshold']}, max error < {cfg['hypothesis']['max_attraction_error_threshold']}",
        f"Fitted beta: {final_params['beta']:.6f}",
        f"Fitted gamma: {final_params['gamma']:.6f}",
    ]
    if bootstrap_summary is not None:
        lo, hi = bootstrap_summary["ci_percent"]
        lines += [
            f"Bootstrap ({bootstrap_summary['n_bootstrap']} month-resamples, {lo:.0f}-{hi:.0f}% CI):",
            f"  val_mae: [{bootstrap_summary['val_mae_p_low']:.4f}, {bootstrap_summary['val_mae_p_high']:.4f}]",
            f"  beta:    [{bootstrap_summary['beta_p_low']:.5f}, {bootstrap_summary['beta_p_high']:.5f}]",
            f"  gamma:   [{bootstrap_summary['gamma_p_low']:.6f}, {bootstrap_summary['gamma_p_high']:.6f}]",
        ]
    lines += [
        f"Final loss: {loss_history[-1]:.6e}" if loss_history else "Final loss: N/A",
        f"Wall time: {elapsed:.1f} s",
        f"Git hash: {git_hash}",
        "Caveat: attraction-side observations are proxy estimates unless MGTO manual counts are filled in.",
    ]
    summary = "\n".join(lines)
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP-05: Real-data calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp05_real_calibration.yaml"),
        help="Path to EXP-05 YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ok = run(cfg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
