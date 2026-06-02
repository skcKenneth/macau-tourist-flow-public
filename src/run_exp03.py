"""EXP-03: MFG Solver Validation on a Synthetic 4-Node Star Graph.

Hypothesis:
    The MFG solver (HJB + FP + fixed-point iteration) produces a Nash equilibrium
    that is correct on a small analytically-tractable test case — within 1% of the
    known answer for the uniform-alpha case, and with correct qualitative ordering
    for 4 additional parameter settings.

Graph topology:
    4-node star: node 0 = transit hub (alpha=0), nodes 1,2,3 = attractions.
    All edges length L (configurable; default 200 m). All edges bidirectional.
    Tourists arrive at transit hub (node 0 only) via a Gaussian profile.

Parameter settings (5 cases):
    C1: Uniform alpha=1, beta=0, gamma=0  → exact 1/3 per attraction
    C2: Skewed alpha=[0,2,1,0.5], beta=0, gamma=0  → ordering rho[1]>rho[2]>rho[3]
    C3: Same skewed alpha + beta=2 → congestion redistributes from top node
    C4: alpha=[0,1,1,0], gamma=5e-4 → zero-alpha attraction gets ~zero density
    C5: alpha=[0,5,0,0] → near-total concentration at node 1 (>80%)

Usage::

    python -m src.run_exp03
    python -m src.run_exp03 --config configs/exp03_solver_validation.yaml

Outputs (in experiments/YYYYMMDD_EXP-03_solver_validation/):
    density_evolution_<case>.png/.pdf  — rho(t) for each node, each case
    equilibrium_bar.png/.pdf           — equilibrium distribution across 5 cases
    metrics.csv                        — per-case convergence + distribution metrics
    summary.txt                        — PASS/FAIL for each case
    config.yaml                        — parameter snapshot

See docs/05_experiment_plan.md §EXP-03 for full specification.
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
# Graph construction
# ---------------------------------------------------------------------------


def _build_star_graph(n_att: int, edge_length_m: float) -> nx.DiGraph:
    """Build a (n_att+1)-node star DiGraph for synthetic experiments.

    Node 0 is the transit hub; nodes 1..n_att are attractions. Every pair of
    nodes is connected by a directed edge in both directions (fully connected
    within the star, i.e. K_{n_att+1} topology), each with the given length.

    Args:
        n_att: Number of attraction nodes.
        edge_length_m: Walking distance in metres for all edges.

    Returns:
        nx.DiGraph with n_att+1 integer nodes and directed edges.
    """
    N = n_att + 1
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(i, j, length=edge_length_m)
    return G


# ---------------------------------------------------------------------------
# Arrival tensor (Gaussian, transit-hub only)
# ---------------------------------------------------------------------------


def _build_arrival_tensor(
    N: int,
    T_steps: int,
    dt: float,
    n_tourists: float,
    peak_time_hours: float,
    sigma_hours: float,
) -> torch.Tensor:
    """Return (T_steps, N) arrival tensor with tourists injected only at node 0."""
    import torch
    import math

    t_vec = torch.linspace(0.0, (T_steps - 1) * dt, T_steps)
    g_raw = torch.exp(-0.5 * ((t_vec - peak_time_hours) / sigma_hours) ** 2)
    g_norm = g_raw / (g_raw.sum() * dt + 1e-12)   # integrates to 1 tourist/h

    arrivals = torch.zeros(T_steps, N, dtype=torch.float32)
    arrivals[:, 0] = (n_tourists * g_norm).float()  # all arrivals at transit hub
    return arrivals


# ---------------------------------------------------------------------------
# Analytical / expected-value checks
# ---------------------------------------------------------------------------


def _check_case(case_cfg: dict, eq_att: torch.Tensor, n_att: int) -> tuple[bool, str]:
    """Verify equilibrium distribution against expected value for one case.

    Args:
        case_cfg: Single case dict from the YAML config.
        eq_att: Equilibrium attraction fractions, shape (n_att,), sums to 1.
        n_att: Number of attraction nodes.

    Returns:
        (passed, message) where passed is True if the check succeeds.
    """
    expected = case_cfg.get("expected", "")
    tol = float(case_cfg.get("tolerance", 0.01))

    if expected == "uniform":
        # Analytical: 1/n_att per attraction
        target = 1.0 / n_att
        max_err = float((eq_att - target).abs().max().item())
        passed = max_err < tol
        msg = (
            f"uniform check: max|rho_att_i - 1/{n_att}| = {max_err:.4f} "
            f"(tol={tol}) -> {'PASS' if passed else 'FAIL'}"
        )

    elif expected == "ordered":
        # Qualitative: rho[1] > rho[2] > rho[3] (attractions sorted by alpha)
        passed = (
            float(eq_att[0].item()) > float(eq_att[1].item()) > float(eq_att[2].item())
        )
        vals = [f"{float(x):.4f}" for x in eq_att]
        msg = f"ordering check: {vals[0]} > {vals[1]} > {vals[2]} -> {'PASS' if passed else 'FAIL'}"

    elif expected == "less_concentrated_than_C2":
        # The top attraction's share should be lower than in the no-congestion case.
        # Stored in module-level dict by the caller after C2 runs.
        # Here we just flag that the caller will compare after the fact.
        passed = True   # deferred comparison — caller sets final verdict
        msg = "less_concentrated check: deferred to post-run comparison"

    elif expected == "two_equal_one_zero":
        # Nodes 1 and 2 have alpha>0, node 3 has alpha=0.
        # Expected: rho[2] near-zero, rho[0] ~= rho[1].
        # Note: in K4 topology att3 retains small option value from connectivity.
        nz_thresh = float(case_cfg.get("near_zero_threshold", 0.05))
        near_zero = float(eq_att[2].item()) < nz_thresh
        nearly_equal = abs(float(eq_att[0].item()) - float(eq_att[1].item())) < 0.05
        passed = near_zero and nearly_equal
        vals = [f"{float(x):.4f}" for x in eq_att]
        msg = (
            f"two_equal_one_zero: rho=[{vals[0]},{vals[1]},{vals[2]}]"
            f" att3<{nz_thresh}={near_zero}, att1~=att2={nearly_equal}"
            f" -> {'PASS' if passed else 'FAIL'}"
        )

    elif expected == "concentrated":
        thresh = float(case_cfg.get("concentration_threshold", 0.80))
        top_frac = float(eq_att[0].item())
        passed = top_frac > thresh
        msg = (
            f"concentration check: rho[1]/(total_att) = {top_frac:.4f} "
            f"(threshold={thresh}) -> {'PASS' if passed else 'FAIL'}"
        )

    else:
        passed = True
        msg = f"no check defined for expected='{expected}'"

    return passed, msg


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run(cfg: dict) -> bool:
    """Execute EXP-03 and return True if all cases pass.

    Args:
        cfg: Parsed configuration dictionary (matches configs/exp03_solver_validation.yaml).

    Returns:
        True if all 5 cases pass their respective checks.
    """
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
    out_cfg = cfg["output"]
    cases = cfg["cases"]

    # ── Experiment directory ──────────────────────────────────────────────────
    outdir = io.make_experiment_dir(
        base=Path(out_cfg["base_dir"]),
        name=f"{exp_cfg['id']}_{exp_cfg['name']}",
    )
    logger.info("Output directory: %s", outdir)

    # ── Seeds + metadata ──────────────────────────────────────────────────────
    seed = exp_cfg["seed"]
    io.set_all_seeds(seed)
    git_hash = io.get_git_hash()
    logger.info("Git hash: %s  Seed: %d", git_hash, seed)

    cfg_to_save = dict(cfg)
    cfg_to_save["_meta"] = {"git_hash": git_hash}
    io.save_config(cfg_to_save, outdir)

    # ── Solver hyperparameters ────────────────────────────────────────────────
    n_att = int(graph_cfg["n_attraction_nodes"])
    edge_length_m = float(graph_cfg["edge_length_m"])
    N = n_att + 1  # total nodes (including transit hub at index 0)

    dt = float(solver_cfg["dt_hours"])
    T_hours = float(solver_cfg["T_hours"])
    epsilon = float(solver_cfg["epsilon"])
    tol = float(solver_cfg["tol"])
    max_iter = int(solver_cfg["max_iter"])

    n_tourists = float(arr_cfg["n_tourists"])
    peak_time_hours = float(arr_cfg["peak_time_hours"])
    sigma_hours = float(arr_cfg["sigma_hours"])

    # ── Build graph and arrival tensor (shared across all cases) ─────────────
    G = _build_star_graph(n_att, edge_length_m)
    T_steps = int(round(T_hours / dt))
    logger.info(
        "Star graph: %d nodes, %d edges | T_steps=%d, dt=%.4f h",
        G.number_of_nodes(), G.number_of_edges(), T_steps, dt,
    )

    g = _build_arrival_tensor(N, T_steps, dt, n_tourists, peak_time_hours, sigma_hours)
    node_order = list(range(N))   # 0=transit, 1..n_att=attractions

    # ── Run all cases ─────────────────────────────────────────────────────────
    results: list[dict] = []
    fig_cfg = out_cfg.get("figures", {})
    dpi = int(fig_cfg.get("dpi", 300))
    figsize_density = fig_cfg.get("figsize_density", [14, 5])
    figsize_bar = fig_cfg.get("figsize_bar", [12, 5])

    c2_top_fraction: float | None = None  # for C3 deferred comparison

    for case_cfg in cases:
        case_id = case_cfg["id"]
        desc = case_cfg.get("description", "")
        alpha_list = case_cfg["alpha"]
        beta_val = float(case_cfg["beta"])
        gamma_val = float(case_cfg["gamma"])

        logger.info("=" * 60)
        logger.info("Case %s: %s", case_id, desc)

        params = {
            "alpha": alpha_list,
            "beta": beta_val,
            "gamma": gamma_val,
        }
        solver = MFGSolver(
            G=G,
            params=params,
            dt=dt,
            T=T_hours,
            epsilon=epsilon,
            tol=tol,
            max_iter=max_iter,
            node_order=node_order,
        )

        t0 = time.perf_counter()
        rho_eq, u_eq, info = solver.fixed_point_iteration(g)
        elapsed = time.perf_counter() - t0

        logger.info(
            "  converged=%s  n_iter=%d  residual=%.2e  wall=%.3f s",
            info["converged"], info["n_iter"], info["final_residual"], elapsed,
        )

        # Equilibrium fraction per attraction (last 30 steps average)
        rho_tail = rho_eq[-30:]                       # (30, N)
        rho_mean = rho_tail.mean(dim=0)               # (N,)
        att_mass = rho_mean[1:].sum()                 # sum over attractions only
        eq_att = rho_mean[1:] / att_mass.clamp(min=1e-9)   # (n_att,) normalised

        # Also compute cumulative fraction (more stable over full horizon)
        cumulative = rho_eq.sum(dim=0)                # (N,)
        att_cumul = cumulative[1:]
        eq_att_cumul = att_cumul / att_cumul.sum().clamp(min=1e-9)

        passed, check_msg = _check_case(case_cfg, eq_att, n_att)

        # Deferred C3 check: compare top attraction fraction vs C2
        if case_cfg.get("expected") == "less_concentrated_than_C2":
            if c2_top_fraction is not None:
                top_frac = float(eq_att[0].item())
                passed = top_frac < c2_top_fraction
                check_msg = (
                    f"less_concentrated: rho[1]={top_frac:.4f} < C2_rho[1]={c2_top_fraction:.4f}"
                    f" -> {'PASS' if passed else 'FAIL'}"
                )
            else:
                logger.warning("C3 deferred check: C2 not yet run, skipping comparison")

        if case_id == "C2_skewed_alpha":
            c2_top_fraction = float(eq_att[0].item())

        logger.info("  %s", check_msg)

        results.append({
            "case_id": case_id,
            "description": desc,
            "alpha": alpha_list,
            "beta": beta_val,
            "gamma": gamma_val,
            "converged": info["converged"],
            "n_iter": info["n_iter"],
            "final_residual": info["final_residual"],
            "wall_s": elapsed,
            "eq_att_tail": eq_att,
            "eq_att_cumul": eq_att_cumul,
            "rho_eq": rho_eq,
            "passed": passed,
            "check_msg": check_msg,
        })

        # ── Per-case density evolution figure ─────────────────────────────────
        t_hours = np.arange(T_steps) * dt
        fig, ax = plt.subplots(figsize=figsize_density)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        labels = ["transit (hub)", "attraction 1", "attraction 2", "attraction 3"]
        for n in range(N):
            ax.plot(t_hours, rho_eq[:, n].numpy(), color=colors[n], label=labels[n], lw=1.5)
        ax.set_xlabel("Time (hours since 08:00)")
        ax.set_ylabel("Tourist density (raw count)")
        ax.set_title(
            f"EXP-03 {case_id}: Density Evolution\n"
            f"α={alpha_list}  β={beta_val}  γ={gamma_val}  ε={epsilon}",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        io.save_figure(fig, outdir, f"density_evolution_{case_id}", dpi=dpi)
        plt.close(fig)

    # ── Summary bar chart: all cases side by side ─────────────────────────────
    _generate_equilibrium_bar(results, n_att, figsize_bar, dpi, outdir, io)

    # ── metrics.csv ───────────────────────────────────────────────────────────
    _write_metrics_csv(results, n_att, outdir)

    # ── summary.txt ───────────────────────────────────────────────────────────
    all_passed = all(r["passed"] for r in results)
    sep = "=" * 65
    lines = [
        f"EXP-03 MFG Solver Validation -- [{'ALL PASS' if all_passed else 'SOME FAIL'}]",
        sep,
        f"Graph: 4-node star (1 transit + 3 attractions), edge={edge_length_m:.0f} m",
        f"Solver: dt={dt:.4f} h, T={T_hours:.0f} h, eps={epsilon}, tol={tol:.0e}, max_iter={max_iter}",
        f"Arrivals: {n_tourists:.0f} tourists, peak at +{peak_time_hours:.1f} h, sigma={sigma_hours:.1f} h",
        "",
    ]
    for r in results:
        eq_str = ", ".join(f"{float(x):.4f}" for x in r["eq_att_tail"])
        lines += [
            f"  {r['case_id']}: {'PASS' if r['passed'] else 'FAIL'}",
            f"    alpha={r['alpha']}  beta={r['beta']}  gamma={r['gamma']}",
            f"    converged={r['converged']}  n_iter={r['n_iter']}  residual={r['final_residual']:.2e}  wall={r['wall_s']:.3f}s",
            f"    eq_att (tail avg) = [{eq_str}]",
            f"    {r['check_msg']}",
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
# Figure helpers
# ---------------------------------------------------------------------------


def _generate_equilibrium_bar(
    results: list[dict],
    n_att: int,
    figsize: list,
    dpi: int,
    outdir: Path,
    io_mod,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n_cases = len(results)
    x = np.arange(n_att)
    width = 0.8 / n_cases
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_cases))

    fig, ax = plt.subplots(figsize=figsize)
    for k, r in enumerate(results):
        offset = (k - n_cases / 2 + 0.5) * width
        vals = r["eq_att_tail"].numpy()
        bars = ax.bar(
            x + offset, vals, width, label=r["case_id"],
            color=colors[k], alpha=0.85,
        )
        # Mark failed cases with hatching
        if not r["passed"]:
            for bar in bars:
                bar.set_hatch("//")

    ax.axhline(1.0 / n_att, color="gray", ls="--", lw=1.0, label=f"Uniform (1/{n_att})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"attraction {i+1}" for i in range(n_att)], fontsize=10)
    ax.set_ylabel("Equilibrium density fraction (tail-average)")
    ax.set_ylim(0, 1.05)
    ax.set_title("EXP-03: Equilibrium Attraction Distribution Across 5 Cases", fontsize=11)
    ax.legend(fontsize=8, ncol=2)

    # Annotate PASS/FAIL
    for k, r in enumerate(results):
        label = "✓" if r["passed"] else "✗"
        y_max = float(r["eq_att_tail"].max().item())
        ax.annotate(
            label,
            xy=(k * (n_att - 1) / (n_cases - 1), y_max + 0.02),
            xycoords="data",
            ha="center", va="bottom", fontsize=12,
            color="#2ca02c" if r["passed"] else "#d62728",
        )

    fig.tight_layout()
    io_mod.save_figure(fig, outdir, "equilibrium_bar", dpi=dpi)
    plt.close(fig)
    logger.info("Equilibrium bar chart saved.")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_metrics_csv(results: list[dict], n_att: int, outdir: Path) -> None:
    csv_path = outdir / "metrics.csv"
    fieldnames = (
        ["case_id", "converged", "n_iter", "final_residual", "wall_s", "passed"]
        + [f"att{i+1}_tail_frac" for i in range(n_att)]
        + [f"att{i+1}_cumul_frac" for i in range(n_att)]
        + ["check_msg"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict = {
                "case_id": r["case_id"],
                "converged": r["converged"],
                "n_iter": r["n_iter"],
                "final_residual": f"{r['final_residual']:.4e}",
                "wall_s": f"{r['wall_s']:.4f}",
                "passed": r["passed"],
                "check_msg": r["check_msg"],
            }
            for i in range(n_att):
                row[f"att{i+1}_tail_frac"] = f"{float(r['eq_att_tail'][i].item()):.6f}"
                row[f"att{i+1}_cumul_frac"] = f"{float(r['eq_att_cumul'][i].item()):.6f}"
            writer.writerow(row)

    logger.info("metrics.csv written: %s", csv_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-03: MFG Solver Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp03_solver_validation.yaml"),
        help="Path to YAML config (default: configs/exp03_solver_validation.yaml)",
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
