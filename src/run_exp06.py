"""EXP-06: One-at-a-time (OAT) Sensitivity Analysis.

Hypothesis:
    Equilibrium peak density at the bottleneck node (node 1, ruins-equivalent)
    is most sensitive to congestion coefficient beta, followed by the bottleneck's
    own attractiveness alpha_1, with walking cost gamma and secondary-node
    attractiveness having smaller influence.

Method:
    Use calibrated baseline parameters from EXP-04 R1_base:
        alpha=[0.0, 2.095, 1.022, 0.527]  beta=0.0133  gamma=0.0

    For each of 5 parameters (alpha_1, alpha_2, beta, gamma, n_tourists):
        Vary the parameter by factors [0.8, 0.9, 1.0, 1.1, 1.2] (i.e. ±20% in 10%
        increments). For each setting run the MFG fixed-point to convergence
        (damping=0.5, tol=5e-4) and record:
            - peak_density_node1: max over time of rho[t, node=1]
            - gini: Gini coefficient of time-averaged density distribution
            - mean_attractions: fraction of attraction nodes with > 5% of tourist-time

    Sensitivity index S_i = (metric_max − metric_min) / metric_baseline.

    Graph: 4-node star (1 transit hub + 3 attractions, edge=200 m) — same as EXP-03/04.

Success criterion:
    Produce a ranked sensitivity table + identify the 2–3 most influential parameters.
    Expected order for peak_density_node1: alpha_1 > n_tourists > beta > alpha_2 > gamma.

Usage::

    python -m src.run_exp06
    python -m src.run_exp06 --config configs/exp06_sensitivity.yaml

Outputs (in experiments/YYYYMMDD_EXP-06_sensitivity/):
    tornado_peak_density.png/.pdf   — ranked horizontal bar chart
    sensitivity_curves.png/.pdf     — metric vs perturbation, one subplot per metric
    sensitivity_table.csv           — raw results (param × perturb × metrics)
    sensitivity_indices.csv         — S_i per param × metric
    summary.txt                     — ranked table + PASS/FAIL
    config.yaml                     — parameter snapshot

See docs/05_experiment_plan.md §EXP-06 for full specification.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import networkx as nx
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph and arrival helpers (same pattern as run_exp03/04)
# ---------------------------------------------------------------------------


def _build_star_graph(n_att: int, edge_length_m: float) -> nx.DiGraph:
    """Fully-connected directed star graph (K_{n_att+1})."""
    N = n_att + 1
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


def _build_arrival_tensor(
    N: int, T_steps: int, dt: float, n_tourists: float,
    peak_time_hours: float, sigma_hours: float,
) -> torch.Tensor:
    """Gaussian arrival profile at transit hub (node 0 only)."""
    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - peak_time_hours) / sigma_hours) ** 2)
    g_norm = g_raw / (g_raw.sum() * dt + 1e-12)
    arrivals = torch.zeros(T_steps, N, dtype=torch.float32)
    arrivals[:, 0] = (n_tourists * g_norm).float()
    return arrivals


# ---------------------------------------------------------------------------
# Parameter perturbation helper
# ---------------------------------------------------------------------------


def _build_perturbed_params(
    baseline: dict,
    param_id: str,
    perturb: float,
    n_nodes: int,
    baseline_override: float | None = None,
) -> dict:
    """Apply a scalar multiplier to one parameter, keeping others at baseline.

    Args:
        baseline: Dict with keys ``alpha`` (list), ``beta`` (float), ``gamma`` (float).
        param_id: Which parameter to perturb. One of
            ``"alpha_1"``, ``"alpha_2"``, ``"beta"``, ``"gamma"``, ``"n_tourists"``.
            Note: ``"n_tourists"`` is handled externally (changes arrival tensor).
        perturb: Multiplicative factor, e.g. 1.2 = +20%.
        n_nodes: Number of nodes (used for alpha list length).
        baseline_override: If provided and the baseline value is zero (e.g. gamma=0),
            use this value as the reference for the ±perturb multiplier.

    Returns:
        New params dict with the perturbed value; other entries are shallow-copied.
    """
    import copy
    result = {
        "alpha": list(baseline["alpha"]),  # copy
        "beta": float(baseline["beta"]),
        "gamma": float(baseline["gamma"]),
    }

    if param_id == "alpha_1":
        ref = float(baseline["alpha"][1])
        result["alpha"][1] = ref * perturb

    elif param_id == "alpha_2":
        ref = float(baseline["alpha"][2])
        result["alpha"][2] = ref * perturb

    elif param_id == "beta":
        result["beta"] = float(baseline["beta"]) * perturb

    elif param_id == "gamma":
        ref = float(baseline["gamma"])
        if abs(ref) < 1e-12 and baseline_override is not None:
            ref = float(baseline_override)
        result["gamma"] = ref * perturb

    elif param_id == "n_tourists":
        pass  # handled by caller via n_tourists scaling

    else:
        raise ValueError(f"Unknown param_id: '{param_id}'")

    return result


# ---------------------------------------------------------------------------
# Single-configuration MFG run + metric extraction
# ---------------------------------------------------------------------------


def _run_single(
    solver,
    g: torch.Tensor,
    params_dict: dict,
    solver_cfg: dict,
) -> dict[str, float]:
    """Run MFG fixed-point with given params; return sensitivity metrics.

    Args:
        solver: MFGSolver instance (G and hyperparams already set; params will
            be overwritten).
        g: Arrival tensor (T_steps, N_nodes).
        params_dict: Dict with ``alpha`` (list/tensor), ``beta``, ``gamma``.
        solver_cfg: Dict with at least ``damping`` key.

    Returns:
        Dict with keys:
        - ``peak_density_node1``: max over time of rho[:, 1]
        - ``gini``: Gini coefficient of time-averaged density distribution
        - ``mean_attractions``: fraction of attraction nodes with > 5% of total
          tourist-time (proxy for "how many attractions are meaningfully visited")
        - ``n_fp_iter``: number of fixed-point iterations to convergence
        - ``fp_converged``: bool
    """
    from src.evaluation.metrics import gini_coefficient

    damping = float(solver_cfg["damping"])
    N = g.shape[1]

    # Set solver params
    alpha_t = torch.tensor(params_dict["alpha"], dtype=torch.float32)
    solver.params = {
        "alpha": alpha_t,
        "beta": torch.tensor(float(params_dict["beta"]), dtype=torch.float32),
        "gamma": torch.tensor(float(params_dict["gamma"]), dtype=torch.float32),
    }

    with torch.no_grad():
        rho, _u, info = solver.fixed_point_iteration(g, damping=damping)

    # peak_density_node1: max density at bottleneck node (node 1) over all time steps
    peak_density_node1 = float(rho[:, 1].max().item())

    # gini: evaluated at the final time step (equilibrium snapshot)
    t_final = rho.shape[0] - 1
    gini = gini_coefficient(rho, t_idx=t_final)

    # mean_attractions: fraction of attraction nodes (1..N-1) whose time-averaged
    # density is > 5% of the sum across all attraction nodes.
    # This is a proxy for "how many attractions are meaningfully visited."
    rho_mean_att = rho[:, 1:].mean(dim=0)          # (N-1,) time-averaged density per attraction
    total_att = float(rho_mean_att.sum().item())
    if total_att > 1e-12:
        att_fractions = rho_mean_att / total_att    # normalized fractions
        mean_attractions = float((att_fractions > 0.05).float().sum().item())
    else:
        mean_attractions = 0.0

    return {
        "peak_density_node1": peak_density_node1,
        "gini": gini,
        "mean_attractions": mean_attractions,
        "n_fp_iter": int(info["n_iter"]),
        "fp_converged": bool(info["converged"]),
    }


# ---------------------------------------------------------------------------
# Sensitivity index computation
# ---------------------------------------------------------------------------


def _compute_sensitivity_indices(
    records: list[dict],
    param_ids: list[str],
    metric_names: list[str],
    perturbations: list[float],
) -> dict[str, dict[str, float]]:
    """Compute OAT sensitivity indices S_i = (max − min) / baseline for each param×metric.

    Args:
        records: List of dicts from the sweep with keys:
            ``param_id``, ``perturb``, and metric values.
        param_ids: Ordered list of parameter IDs.
        metric_names: Ordered list of metric names.
        perturbations: The perturbation factors used (1.0 must be in the list).

    Returns:
        Nested dict: ``indices[param_id][metric_name] = S_i``.
    """
    indices: dict[str, dict[str, float]] = {}
    for pid in param_ids:
        indices[pid] = {}
        rows = [r for r in records if r["param_id"] == pid]
        # Baseline row (perturb=1.0)
        baseline_row = next((r for r in rows if abs(r["perturb"] - 1.0) < 1e-9), None)
        for mname in metric_names:
            vals = [r[mname] for r in rows]
            val_max = max(vals)
            val_min = min(vals)
            val_baseline = baseline_row[mname] if baseline_row is not None else 1.0
            denom = abs(val_baseline) if abs(val_baseline) > 1e-12 else 1.0
            indices[pid][mname] = (val_max - val_min) / denom
    return indices


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run(cfg: dict) -> bool:
    """Execute EXP-06 and return True (success criterion: table produced + ≥2 params with S_i > 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.models.mfg_solver import MFGSolver
    from src.utils import io

    exp_cfg = cfg["experiment"]
    graph_cfg = cfg["graph"]
    solver_cfg = cfg["solver"]
    arr_cfg = cfg["arrivals"]
    baseline_cfg = cfg["baseline"]
    sweep_cfg = cfg["sweep"]
    out_cfg = cfg["output"]

    # ── Experiment directory ──────────────────────────────────────────────────
    outdir = io.make_experiment_dir(
        base=Path(out_cfg["base_dir"]),
        name=f"{exp_cfg['id']}_{exp_cfg['name']}",
    )
    logger.info("Output directory: %s", outdir)

    seed = exp_cfg["seed"]
    io.set_all_seeds(seed)
    git_hash = io.get_git_hash()
    logger.info("Git hash: %s  Seed: %d", git_hash, seed)
    cfg_to_save = dict(cfg)
    cfg_to_save["_meta"] = {"git_hash": git_hash}
    io.save_config(cfg_to_save, outdir)

    # ── Graph and solver setup ────────────────────────────────────────────────
    n_att = int(graph_cfg["n_attraction_nodes"])
    edge_length_m = float(graph_cfg["edge_length_m"])
    N = n_att + 1
    node_order = list(range(N))

    dt = float(solver_cfg["dt_hours"])
    T_hours = float(solver_cfg["T_hours"])
    T_steps = int(round(T_hours / dt))
    epsilon = float(solver_cfg["epsilon"])
    tol = float(solver_cfg["tol"])
    max_iter = int(solver_cfg["max_iter"])
    damping = float(solver_cfg["damping"])

    n_tourists_base = float(arr_cfg["n_tourists"])
    peak_time_hours = float(arr_cfg["peak_time_hours"])
    sigma_hours = float(arr_cfg["sigma_hours"])

    baseline = {
        "alpha": list(baseline_cfg["alpha"]),
        "beta": float(baseline_cfg["beta"]),
        "gamma": float(baseline_cfg["gamma"]),
    }

    perturbations = list(sweep_cfg["perturbations"])
    param_entries = list(sweep_cfg["parameters"])
    param_ids = [p["id"] for p in param_entries]
    baseline_overrides = {
        p["id"]: p["baseline_override"]
        for p in param_entries if "baseline_override" in p
    }

    metric_names = list(cfg.get("metrics", ["peak_density_node1", "gini", "mean_attractions"]))

    fig_cfg = out_cfg.get("figures", {})
    dpi = int(fig_cfg.get("dpi", 300))
    figsize_tornado = fig_cfg.get("figsize_tornado", [10, 6])
    figsize_curves = fig_cfg.get("figsize_curves", [14, 10])

    G = _build_star_graph(n_att, edge_length_m)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("EXP-06 OAT sensitivity sweep: %d params × %d perturbations",
                len(param_ids), len(perturbations))
    logger.info("Baseline: alpha=%s  beta=%.4f  gamma=%.6f",
                baseline["alpha"], baseline["beta"], baseline["gamma"])
    logger.info("=" * 65)

    records: list[dict] = []

    for param_entry in param_entries:
        pid = param_entry["id"]
        desc = param_entry.get("description", pid)
        bov = baseline_overrides.get(pid, None)
        logger.info("  Parameter: %s (%s)", pid, desc)

        for perturb in perturbations:
            # For n_tourists, scale arrival tensor; for others, perturb params dict
            if pid == "n_tourists":
                n_tourists_eff = n_tourists_base * perturb
                g = _build_arrival_tensor(N, T_steps, dt, n_tourists_eff,
                                          peak_time_hours, sigma_hours)
                params_dict = dict(baseline)
                params_dict["alpha"] = list(baseline["alpha"])
            else:
                g = _build_arrival_tensor(N, T_steps, dt, n_tourists_base,
                                          peak_time_hours, sigma_hours)
                params_dict = _build_perturbed_params(
                    baseline, pid, perturb, N, baseline_override=bov
                )

            # Build solver (reuse graph, update params each call)
            alpha_t = torch.tensor(params_dict["alpha"], dtype=torch.float32)
            solver = MFGSolver(
                G=G,
                params={
                    "alpha": alpha_t,
                    "beta": torch.tensor(float(params_dict["beta"]), dtype=torch.float32),
                    "gamma": torch.tensor(float(params_dict["gamma"]), dtype=torch.float32),
                },
                dt=dt, T=T_hours, epsilon=epsilon,
                tol=tol, max_iter=max_iter, node_order=node_order,
            )

            metrics = _run_single(solver, g, params_dict, solver_cfg)

            if not metrics["fp_converged"]:
                logger.warning(
                    "    FP did not converge: param=%s perturb=%.2f (n_iter=%d)",
                    pid, perturb, metrics["n_fp_iter"]
                )

            record = {"param_id": pid, "perturb": perturb}
            record.update({k: metrics[k] for k in metric_names})
            record["n_fp_iter"] = metrics["n_fp_iter"]
            record["fp_converged"] = metrics["fp_converged"]
            records.append(record)

            logger.info(
                "    perturb=%.2f: peak_node1=%.4f  gini=%.4f  mean_att=%.2f  fp=%d(%s)",
                perturb,
                metrics["peak_density_node1"],
                metrics["gini"],
                metrics["mean_attractions"],
                metrics["n_fp_iter"],
                "OK" if metrics["fp_converged"] else "no-conv",
            )

    # ── Sensitivity indices ───────────────────────────────────────────────────
    indices = _compute_sensitivity_indices(records, param_ids, metric_names, perturbations)

    # Rank by S_i for primary metric (peak_density_node1)
    primary_metric = "peak_density_node1"
    ranked_params = sorted(
        param_ids,
        key=lambda pid: indices[pid][primary_metric],
        reverse=True,
    )

    logger.info("=" * 65)
    logger.info("Sensitivity indices (S_i = (max-min)/baseline) for %s:", primary_metric)
    for rank, pid in enumerate(ranked_params, 1):
        logger.info("  #%d  %-15s  S_i=%.4f", rank, pid, indices[pid][primary_metric])

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_tornado(indices, ranked_params, metric_names, primary_metric,
                  figsize_tornado, dpi, outdir, io)
    _plot_sensitivity_curves(records, param_ids, metric_names, perturbations,
                             figsize_curves, dpi, outdir, io)

    # ── sensitivity_table.csv ─────────────────────────────────────────────────
    _write_sensitivity_table(records, metric_names, outdir)

    # ── sensitivity_indices.csv ───────────────────────────────────────────────
    _write_indices_csv(indices, param_ids, metric_names, outdir)

    # ── summary.txt ───────────────────────────────────────────────────────────
    n_nonzero = sum(1 for pid in param_ids if indices[pid][primary_metric] > 1e-6)
    passed = n_nonzero >= 2
    sep = "=" * 65

    lines = [
        f"EXP-06 Sensitivity Analysis -- [{'PASS' if passed else 'FAIL'}]",
        sep,
        f"Graph: {n_att}-attraction star, edge={edge_length_m:.0f} m",
        f"Solver: dt={dt:.4f} h, T={T_hours:.0f} h, eps={epsilon}, tol={tol:.0e}",
        f"Arrivals: {n_tourists_base:.0f} tourists (base), peak +{peak_time_hours:.1f} h, sigma={sigma_hours:.1f} h",
        f"Perturbations: {perturbations}",
        "",
        f"Ranked sensitivity for primary metric ({primary_metric}):",
        f"  {'Rank':<6}  {'Parameter':<16}  {'S_i':>8}  {'Description'}",
        "  " + "-" * 60,
    ]
    for rank, pid in enumerate(ranked_params, 1):
        desc = next((p.get("description", pid) for p in param_entries if p["id"] == pid), pid)
        si = indices[pid][primary_metric]
        lines.append(f"  #{rank:<5}  {pid:<16}  {si:>8.4f}  {desc}")

    lines += [
        "",
        "Full sensitivity indices (all metrics):",
        "  " + f"{'Parameter':<16} " + " ".join(f"{m:>22}" for m in metric_names),
        "  " + "-" * (16 + 23 * len(metric_names)),
    ]
    for pid in ranked_params:
        row_vals = " ".join(f"{indices[pid][m]:>22.4f}" for m in metric_names)
        lines.append(f"  {pid:<16} {row_vals}")

    lines += [
        "",
        f"Params with S_i > 0: {n_nonzero}/{len(param_ids)}",
        f"Overall: {'PASS' if passed else 'FAIL'} (need >= 2 params with S_i > 0)",
        sep,
        f"Git hash: {git_hash}",
        f"Seed: {seed}",
        "",
    ]
    summary_txt = "\n".join(lines)
    (outdir / "summary.txt").write_text(summary_txt, encoding="utf-8")
    print("\n" + summary_txt)

    return passed


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _plot_tornado(
    indices: dict, ranked_params: list[str], metric_names: list[str],
    primary_metric: str, figsize: list, dpi: int, outdir: Path, io_mod,
) -> None:
    """Horizontal bar chart of sensitivity indices (tornado plot)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(ranked_params))
    si_values = [indices[pid][primary_metric] for pid in ranked_params]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranked_params)))

    bars = ax.barh(y_pos, si_values, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked_params, fontsize=11)
    ax.set_xlabel(f"Sensitivity index  S_i = (max − min) / baseline", fontsize=10)
    ax.set_title(
        f"EXP-06: OAT Sensitivity Analysis\n"
        f"Primary metric: {primary_metric}  |  baseline = EXP-04 R1_base",
        fontsize=11,
    )
    ax.invert_yaxis()  # highest sensitivity at top
    ax.grid(axis="x", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, si_values):
        ax.text(
            bar.get_width() + max(si_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=9,
        )

    fig.tight_layout()
    io_mod.save_figure(fig, outdir, "tornado_peak_density", dpi=dpi)
    plt.close(fig)
    logger.info("Tornado plot saved.")


def _plot_sensitivity_curves(
    records: list[dict], param_ids: list[str], metric_names: list[str],
    perturbations: list[float], figsize: list, dpi: int, outdir: Path, io_mod,
) -> None:
    """Grid of subplots: metric vs perturbation for each parameter."""
    import matplotlib.pyplot as plt
    import numpy as np

    n_metrics = len(metric_names)
    n_params = len(param_ids)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=False)
    if n_metrics == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    for mi, mname in enumerate(metric_names):
        ax = axes[mi]
        for ki, pid in enumerate(param_ids):
            rows = sorted(
                [r for r in records if r["param_id"] == pid],
                key=lambda r: r["perturb"],
            )
            xs = [r["perturb"] for r in rows]
            ys = [r[mname] for r in rows]
            ax.plot(xs, ys, "o-", color=cmap(ki / 10.0), label=pid, lw=1.5, markersize=5)

        ax.axvline(1.0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
        ax.set_ylabel(mname.replace("_", " "), fontsize=9)
        ax.set_xlabel("Perturbation factor", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)

    fig.suptitle(
        "EXP-06: Sensitivity Curves (OAT, ±20%)\nBaseline = EXP-04 R1_base recovered params",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    io_mod.save_figure(fig, outdir, "sensitivity_curves", dpi=dpi)
    plt.close(fig)
    logger.info("Sensitivity curves saved.")


# ---------------------------------------------------------------------------
# CSV output helpers
# ---------------------------------------------------------------------------


def _write_sensitivity_table(
    records: list[dict], metric_names: list[str], outdir: Path
) -> None:
    csv_path = outdir / "sensitivity_table.csv"
    fieldnames = ["param_id", "perturb"] + metric_names + ["n_fp_iter", "fp_converged"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in fieldnames if k in r}
            for m in metric_names:
                row[m] = f"{r[m]:.6f}" if isinstance(r.get(m), float) else r.get(m, "")
            writer.writerow(row)
    logger.info("sensitivity_table.csv written: %s", csv_path)


def _write_indices_csv(
    indices: dict, param_ids: list[str], metric_names: list[str], outdir: Path
) -> None:
    csv_path = outdir / "sensitivity_indices.csv"
    fieldnames = ["param_id"] + metric_names
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid in param_ids:
            row = {"param_id": pid}
            for m in metric_names:
                row[m] = f"{indices[pid][m]:.6f}"
            writer.writerow(row)
    logger.info("sensitivity_indices.csv written: %s", csv_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-06: OAT Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp06_sensitivity.yaml"),
        help="Path to YAML config (default: configs/exp06_sensitivity.yaml)",
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
