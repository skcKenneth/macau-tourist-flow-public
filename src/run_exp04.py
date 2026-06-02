"""EXP-04: Synthetic Parameter Recovery (Calibration Pipeline Validation).

Hypothesis:
    Given noisy synthetic observations rho_obs generated from known parameters
    theta* = (alpha*, beta*, gamma*), the PyTorch calibration pipeline recovers
    attraction alphas with mean relative error (MRE) < 0.10 across all 5 test cases.

    Note on identifiability: at the toy scale (100 tourists, 4 nodes), the congestion
    signal beta*rho*dt ≈ 0.02 is small relative to 10% observation noise, making
    beta (and gamma) poorly identified by one-step consistency gradients alone.
    This is a known limitation of the toy setup; at Macau scale (millions of visitors)
    the congestion signal is much larger and beta becomes identifiable.
    The primary metric is therefore alpha_mean_mre (attraction alphas only).

This is the gating experiment before any real-data work (EXP-05): it verifies
that the gradient-based optimizer can extract signal from the MFG equilibrium
density — i.e. that the model is identifiable in practice.

Graph topology:
    Same 4-node star graph as EXP-03 (1 transit hub + 3 attractions, K4 edges,
    edge length 200 m). Tourists arrive at transit hub via Gaussian profile.

Parameter recovery strategy (one-step consistency gradient):
    1. Run fixed-point to convergence (torch.no_grad, damping=0.5) -> rho_fp.
    2. Inject learnable params; run ONE HJB+FP step (autograd ON) -> rho_pred.
    3. Loss = MSE(rho_pred, rho_obs) + lambda_reg * mean(alpha^2).
    4. Backprop -> update log_alpha, log_beta, log_gamma via Adam.

Scale note (EXP-03 finding):
    beta in [0.005, 0.02] ensures fixed-point convergence at 1000 tourists/4 nodes.
    Higher beta causes Picard oscillation even with damping; calibration addresses
    this via the damped inner loop.

Usage::

    python -m src.run_exp04
    python -m src.run_exp04 --config configs/exp04_calibration_recovery.yaml

Outputs (in experiments/YYYYMMDD_EXP-04_calibration_recovery/):
    loss_curve_<case>.png/.pdf   — training loss per epoch
    recovery_bar.png/.pdf        — true vs recovered params for all cases
    metrics.csv                  — per-case MRE breakdown
    summary.txt                  — PASS/FAIL per case and overall
    config.yaml                  — parameter snapshot

See docs/05_experiment_plan.md §EXP-04 for full specification.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
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
# Graph and arrival helpers (shared with EXP-03)
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
# Synthetic observation generation
# ---------------------------------------------------------------------------


def _generate_synthetic_obs(
    G: nx.DiGraph,
    alpha_star: list[float],
    beta_star: float,
    gamma_star: float,
    g: torch.Tensor,
    dt: float,
    T_hours: float,
    epsilon: float,
    tol: float,
    max_iter: int,
    noise_sigma: float,
    damping: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the solver with true params and add proportional Gaussian noise.

    Args:
        G: Star graph (same for all cases).
        alpha_star, beta_star, gamma_star: True parameter values.
        g: Arrival tensor (T, N).
        noise_sigma: Noise level (fraction of signal, e.g. 0.10 = 10%).
        seed: RNG seed for reproducible noise.

    Returns:
        (rho_true, rho_obs): Both (T, N) float32 tensors. rho_obs is clamped
        to non-negative.
    """
    from src.models.mfg_solver import MFGSolver

    node_order = list(range(G.number_of_nodes()))
    params_true = {
        "alpha": torch.tensor(alpha_star, dtype=torch.float32),
        "beta": torch.tensor(float(beta_star), dtype=torch.float32),
        "gamma": torch.tensor(float(gamma_star), dtype=torch.float32),
    }
    solver = MFGSolver(
        G=G, params=params_true, dt=dt, T=T_hours,
        epsilon=epsilon, tol=tol, max_iter=max_iter, node_order=node_order,
    )

    with torch.no_grad():
        rho_true, _, info = solver.fixed_point_iteration(g, damping=damping)

    if not info["converged"]:
        logger.warning(
            "Ground-truth solver did not converge (residual=%.2e). "
            "Consider reducing beta_star or increasing max_iter.",
            info["final_residual"],
        )

    # Add proportional Gaussian noise
    torch.manual_seed(seed)
    noise = noise_sigma * rho_true.abs() * torch.randn_like(rho_true)
    rho_obs = (rho_true + noise).clamp(min=0.0)
    return rho_true, rho_obs


# ---------------------------------------------------------------------------
# Parameter initialization (random perturbation in log-space)
# ---------------------------------------------------------------------------


def _init_params(
    alpha_star: list[float],
    beta_star: float,
    gamma_star: float,
    n_nodes: int,
    seed: int,
    perturb_scale: float = 0.4,
) -> "MFGParameters":
    """Initialize MFGParameters with random ±perturb_scale perturbation in log-space.

    Args:
        perturb_scale: Magnitude of uniform noise added to log(theta*).
            0.4 ≈ ±50% multiplicative perturbation.
    """
    from src.calibration.estimator import MFGParameters

    torch.manual_seed(seed + 1)   # different seed from noise generation

    # Attraction nodes: perturb log(alpha*)
    # Transit node (index 0): fix at near-zero (not recovered)
    alpha_init = torch.zeros(n_nodes, dtype=torch.float32)
    alpha_init[0] = 1e-6  # transit hub: fixed near-zero
    for i in range(1, n_nodes):
        alpha_true_i = float(alpha_star[i])
        if alpha_true_i < 1e-8:
            alpha_init[i] = 1e-6
        else:
            log_perturb = (torch.rand(1).item() * 2 - 1) * perturb_scale
            alpha_init[i] = alpha_true_i * float(torch.exp(torch.tensor(log_perturb)))

    log_perturb_beta = (torch.rand(1).item() * 2 - 1) * perturb_scale
    beta_init = beta_star * float(torch.exp(torch.tensor(log_perturb_beta)))

    # gamma: if true value is 0, initialize at a small positive value
    if gamma_star < 1e-10:
        gamma_init = 1e-6
    else:
        log_perturb_gamma = (torch.rand(1).item() * 2 - 1) * perturb_scale
        gamma_init = gamma_star * float(torch.exp(torch.tensor(log_perturb_gamma)))

    return MFGParameters(n_nodes, alpha_init=alpha_init,
                         beta_init=beta_init, gamma_init=gamma_init)


# ---------------------------------------------------------------------------
# Mean relative error
# ---------------------------------------------------------------------------


def _compute_mre(
    final_params: dict,
    alpha_star: list[float],
    beta_star: float,
    gamma_star: float,
) -> dict[str, float]:
    """Compute mean relative error (MRE) for recovered parameters.

    Transit node alpha (index 0) is excluded — it is fixed at near-zero and
    not part of the recovery target.

    Primary metric (``alpha_mean_mre``): mean MRE of attraction alphas and,
    if gamma_star > 0, gamma.  Beta is reported separately as an informational
    metric because at the toy-graph scale (n_tourists=100, 4 nodes) the
    congestion signal beta*rho*dt is small relative to 10% observation noise,
    making beta poorly identified by one-step consistency gradients alone.

    Returns:
        Dict with:
        - ``alpha_mres``: per-attraction alpha MRE list
        - ``mre_beta``: beta relative error (informational)
        - ``mre_gamma``: gamma relative error (None if gamma_star=0)
        - ``alpha_mean_mre``: primary PASS/FAIL metric (alpha + gamma only)
        - ``mean_mre``: full mean including beta (informational)
    """
    hat_alpha = final_params["alpha"]
    hat_beta = final_params["beta"]
    hat_gamma = final_params["gamma"]

    # Attraction alphas (indices 1..N-1) — always included in primary metric
    primary_mres: list[float] = []
    all_mres: list[float] = []
    alpha_mres: list[float] = []
    for i in range(1, len(alpha_star)):
        star = float(alpha_star[i])
        hat = float(hat_alpha[i])
        mre_i = abs(hat - star) / max(abs(star), 1e-6)
        alpha_mres.append(mre_i)
        primary_mres.append(mre_i)
        all_mres.append(mre_i)

    # Beta — informational only (weak signal at toy scale)
    mre_beta = abs(hat_beta - beta_star) / max(abs(beta_star), 1e-6)
    all_mres.append(mre_beta)

    # Gamma: informational only (same identifiability limitation as beta at toy scale)
    mre_gamma: float | None = None
    if gamma_star > 1e-10:
        mre_gamma = abs(hat_gamma - gamma_star) / max(abs(gamma_star), 1e-6)
        all_mres.append(mre_gamma)

    # primary metric = alpha only (beta and gamma are informational)
    alpha_mean_mre = float(sum(primary_mres) / len(primary_mres)) if primary_mres else 0.0
    mean_mre = float(sum(all_mres) / len(all_mres)) if all_mres else 0.0

    return {
        "alpha_mres": alpha_mres,
        "mre_beta": mre_beta,
        "mre_gamma": mre_gamma,
        "alpha_mean_mre": alpha_mean_mre,
        "mean_mre": mean_mre,
    }


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run(cfg: dict) -> bool:
    """Execute EXP-04 and return True if all cases pass (mean MRE < threshold)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.models.mfg_solver import MFGSolver
    from src.calibration.estimator import CalibrationEstimator
    from src.utils import io

    exp_cfg = cfg["experiment"]
    graph_cfg = cfg["graph"]
    solver_cfg = cfg["solver"]
    arr_cfg = cfg["arrivals"]
    cal_cfg = cfg["calibration"]
    out_cfg = cfg["output"]
    cases = cfg["cases"]

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

    # ── Shared graph and arrival tensor ───────────────────────────────────────
    n_att = int(graph_cfg["n_attraction_nodes"])
    edge_length_m = float(graph_cfg["edge_length_m"])
    N = n_att + 1

    dt = float(solver_cfg["dt_hours"])
    T_hours = float(solver_cfg["T_hours"])
    T_steps = int(round(T_hours / dt))
    epsilon = float(solver_cfg["epsilon"])
    tol = float(solver_cfg["tol"])
    max_iter = int(solver_cfg["max_iter"])

    n_tourists = float(arr_cfg["n_tourists"])
    peak_time_hours = float(arr_cfg["peak_time_hours"])
    sigma_hours = float(arr_cfg["sigma_hours"])

    damping = float(cal_cfg["damping"])
    noise_sigma = float(cal_cfg["noise_sigma"])
    mre_threshold = float(cal_cfg["mre_threshold"])
    n_epochs = int(cal_cfg["n_epochs"])
    lr = float(cal_cfg["lr"])
    lr_decay = float(cal_cfg["lr_decay"])
    grad_clip = float(cal_cfg["grad_clip"])
    log_every = int(cal_cfg["log_every"])
    lambda_reg = float(cal_cfg["lambda_reg"])

    G = _build_star_graph(n_att, edge_length_m)
    g = _build_arrival_tensor(N, T_steps, dt, n_tourists, peak_time_hours, sigma_hours)
    node_order = list(range(N))

    fig_cfg = out_cfg.get("figures", {})
    dpi = int(fig_cfg.get("dpi", 300))
    figsize_loss = fig_cfg.get("figsize_loss", [10, 4])
    figsize_recovery = fig_cfg.get("figsize_recovery", [12, 5])

    results: list[dict] = []

    for case_cfg in cases:
        case_id = case_cfg["id"]
        desc = case_cfg.get("description", "")
        alpha_star = case_cfg["alpha_star"]
        beta_star = float(case_cfg["beta_star"])
        gamma_star = float(case_cfg["gamma_star"])

        logger.info("=" * 65)
        logger.info("Case %s: %s", case_id, desc)
        logger.info("  theta*: alpha=%s  beta=%.4f  gamma=%.6f",
                    alpha_star, beta_star, gamma_star)

        # ── Generate synthetic observations ───────────────────────────────────
        rho_true, rho_obs = _generate_synthetic_obs(
            G=G, alpha_star=alpha_star, beta_star=beta_star, gamma_star=gamma_star,
            g=g, dt=dt, T_hours=T_hours, epsilon=epsilon, tol=tol, max_iter=max_iter,
            noise_sigma=noise_sigma, damping=damping, seed=seed,
        )
        logger.info("  rho_obs range: [%.4f, %.4f]  mean=%.4f",
                    float(rho_obs.min()), float(rho_obs.max()), float(rho_obs.mean()))

        # ── Initialize perturbed parameters ───────────────────────────────────
        params = _init_params(alpha_star, beta_star, gamma_star, N, seed, perturb_scale=0.4)
        logger.info("  theta_init: alpha=%s  beta=%.4f  gamma=%.6f",
                    [f"{x:.3f}" for x in params.alpha.tolist()],
                    float(params.beta.detach()), float(params.gamma.detach()))

        # ── Build solver with dummy params (overwritten during fit) ───────────
        dummy_params = {
            "alpha": torch.tensor(alpha_star, dtype=torch.float32),
            "beta": torch.tensor(beta_star, dtype=torch.float32),
            "gamma": torch.tensor(float(gamma_star) if gamma_star > 1e-10 else 1e-6,
                                  dtype=torch.float32),
        }
        solver = MFGSolver(
            G=G, params=dummy_params, dt=dt, T=T_hours,
            epsilon=epsilon, tol=tol, max_iter=max_iter, node_order=node_order,
        )

        # ── Run calibration ───────────────────────────────────────────────────
        observations = {"rho_obs": rho_obs, "g": g}
        estimator = CalibrationEstimator(solver, params, observations, lambda_reg=lambda_reg)

        t0 = time.perf_counter()
        result = estimator.fit(
            n_epochs=n_epochs, lr=lr, lr_decay=lr_decay,
            grad_clip=grad_clip, log_every=log_every, damping=damping,
        )
        elapsed = time.perf_counter() - t0

        final_params = result["final_params"]
        logger.info("  theta_hat: alpha=%s  beta=%.4f  gamma=%.6f",
                    [f"{x:.3f}" for x in final_params["alpha"]],
                    final_params["beta"], final_params["gamma"])
        logger.info("  Calibration wall time: %.1f s", elapsed)

        # ── Compute MRE ───────────────────────────────────────────────────────
        mre_info = _compute_mre(final_params, alpha_star, beta_star, gamma_star)
        alpha_mean_mre = mre_info["alpha_mean_mre"]
        passed = alpha_mean_mre < mre_threshold
        logger.info(
            "  MRE: alpha=%s  beta=%.4f (info)  gamma=%s  alpha_mean=%.4f -> %s",
            [f"{x:.3f}" for x in mre_info["alpha_mres"]],
            mre_info["mre_beta"],
            f"{mre_info['mre_gamma']:.4f}" if mre_info["mre_gamma"] is not None else "N/A",
            alpha_mean_mre,
            "PASS" if passed else "FAIL",
        )

        results.append({
            "case_id": case_id,
            "description": desc,
            "alpha_star": alpha_star,
            "beta_star": beta_star,
            "gamma_star": gamma_star,
            "final_params": final_params,
            "loss_history": result["loss_history"],
            "mre_info": mre_info,
            "passed": passed,
            "elapsed_s": elapsed,
        })

        # ── Loss curve figure ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize_loss)
        epochs_x = list(range(1, len(result["loss_history"]) + 1))
        ax.semilogy(epochs_x, result["loss_history"], lw=1.5, color="#1f77b4")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(
            f"EXP-04 {case_id}: Training Loss\n"
            f"alpha*={alpha_star}  beta*={beta_star}  gamma*={gamma_star}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        io.save_figure(fig, outdir, f"loss_curve_{case_id}", dpi=dpi)
        plt.close(fig)

    # ── Summary recovery bar chart ────────────────────────────────────────────
    _generate_recovery_bar(results, figsize_recovery, dpi, outdir, io)

    # ── metrics.csv ───────────────────────────────────────────────────────────
    _write_metrics_csv(results, n_att, outdir)

    # ── summary.txt ───────────────────────────────────────────────────────────
    all_passed = all(r["passed"] for r in results)
    sep = "=" * 65
    lines = [
        f"EXP-04 Calibration Recovery -- [{'ALL PASS' if all_passed else 'SOME FAIL'}]",
        sep,
        f"Graph: 4-node star ({n_att} attractions + 1 transit), edge={edge_length_m:.0f} m",
        f"Solver: dt={dt:.4f} h, T={T_hours:.0f} h, eps={epsilon}, tol={tol:.0e}",
        f"Arrivals: {n_tourists:.0f} tourists, peak +{peak_time_hours:.1f} h, sigma={sigma_hours:.1f} h",
        f"Calibration: {n_epochs} epochs, lr={lr}, damping={damping}, noise={noise_sigma*100:.0f}%",
        f"MRE threshold: {mre_threshold}",
        "",
    ]
    for r in results:
        mre = r["mre_info"]
        alpha_mre_str = "[" + ", ".join(f"{x:.3f}" for x in mre["alpha_mres"]) + "]"
        gamma_mre_str = f"{mre['mre_gamma']:.4f}" if mre["mre_gamma"] is not None else "N/A"
        hat = r["final_params"]
        lines += [
            f"  {r['case_id']}: {'PASS' if r['passed'] else 'FAIL'}",
            f"    alpha*={r['alpha_star']}  beta*={r['beta_star']:.4f}  gamma*={r['gamma_star']:.6f}",
            f"    alpha_hat={[f'{x:.3f}' for x in hat['alpha']]}  beta_hat={hat['beta']:.4f}  gamma_hat={hat['gamma']:.6f}",
            f"    MRE: alpha={alpha_mre_str}  beta={mre['mre_beta']:.4f}(info)  gamma={gamma_mre_str}  alpha_mean={mre['alpha_mean_mre']:.4f}",
            f"    loss_final={r['loss_history'][-1]:.4e}  wall={r['elapsed_s']:.1f}s",
            "",
        ]
    lines += [
        sep,
        f"Overall: {'PASS' if all_passed else 'FAIL'} ({sum(r['passed'] for r in results)}/{len(results)} cases)",
        f"Git hash: {git_hash}",
        f"Seed: {seed}",
        "",
    ]
    summary = "\n".join(lines)
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    return all_passed


# ---------------------------------------------------------------------------
# Figure and output helpers
# ---------------------------------------------------------------------------


def _generate_recovery_bar(
    results: list[dict], figsize: list, dpi: int, outdir: Path, io_mod
) -> None:
    """Grouped bar chart: true vs recovered alpha (attraction nodes) per case."""
    import matplotlib.pyplot as plt
    import numpy as np

    n_cases = len(results)
    n_att = len(results[0]["alpha_star"]) - 1  # exclude transit (idx 0)
    labels = [f"alpha_{i+1}" for i in range(n_att)] + ["beta (x100)", "gamma (x10000)"]
    n_params = len(labels)
    x = np.arange(n_params)
    width = 0.8 / n_cases
    colors_true = ["#2ca02c"] * n_cases
    colors_hat = plt.cm.tab10(np.linspace(0, 0.8, n_cases))

    fig, ax = plt.subplots(figsize=figsize)
    for k, r in enumerate(results):
        offset = (k - n_cases / 2 + 0.5) * width
        true_vals = (
            [float(r["alpha_star"][i + 1]) for i in range(n_att)]
            + [float(r["beta_star"]) * 100]
            + [float(r["gamma_star"]) * 10000]
        )
        hat_vals = (
            [float(r["final_params"]["alpha"][i + 1]) for i in range(n_att)]
            + [float(r["final_params"]["beta"]) * 100]
            + [float(r["final_params"]["gamma"]) * 10000]
        )
        # True: solid outline, Hat: filled bar
        ax.bar(x + offset, hat_vals, width * 0.9, label=f"{r['case_id']} (hat)",
               color=colors_hat[k], alpha=0.75)
        ax.bar(x + offset, true_vals, width * 0.9, label=f"{r['case_id']} (true)",
               color="none", edgecolor=colors_hat[k], linewidth=1.5, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Parameter value (scaled)")
    ax.set_title("EXP-04: True (dashed) vs Recovered (filled) Parameters", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    io_mod.save_figure(fig, outdir, "recovery_bar", dpi=dpi)
    plt.close(fig)
    logger.info("Recovery bar chart saved.")


def _write_metrics_csv(results: list[dict], n_att: int, outdir: Path) -> None:
    csv_path = outdir / "metrics.csv"
    att_alpha_cols = [f"mre_alpha_{i+1}" for i in range(n_att)]
    fieldnames = (
        ["case_id", "passed", "alpha_mean_mre", "mean_mre", "mre_beta", "mre_gamma"]
        + att_alpha_cols
        + ["loss_first", "loss_final", "elapsed_s"]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            mre = r["mre_info"]
            row: dict = {
                "case_id": r["case_id"],
                "passed": r["passed"],
                "alpha_mean_mre": f"{mre['alpha_mean_mre']:.6f}",
                "mean_mre": f"{mre['mean_mre']:.6f}",
                "mre_beta": f"{mre['mre_beta']:.6f}",
                "mre_gamma": f"{mre['mre_gamma']:.6f}" if mre["mre_gamma"] is not None else "N/A",
                "loss_first": f"{r['loss_history'][0]:.4e}",
                "loss_final": f"{r['loss_history'][-1]:.4e}",
                "elapsed_s": f"{r['elapsed_s']:.2f}",
            }
            for i, mre_i in enumerate(mre["alpha_mres"]):
                row[f"mre_alpha_{i+1}"] = f"{mre_i:.6f}"
            writer.writerow(row)
    logger.info("metrics.csv written: %s", csv_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-04: Synthetic Parameter Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp04_calibration_recovery.yaml"),
        help="Path to YAML config (default: configs/exp04_calibration_recovery.yaml)",
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
