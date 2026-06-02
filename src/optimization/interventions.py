"""Intervention design: entrance metering and routing recommendations.

Implements two complementary intervention strategies evaluated in EXP-07/08/09.
Both are formulated as gradient-descent optimisations through the calibrated
MFG forward simulator.

Intervention A — Entrance Metering (EXP-07):
    Optimise a time-varying arrival rate cap R*(t) at source nodes
    (ferry terminal, border gate) to reduce peak density at bottleneck nodes.

Intervention B — Routing Recommendations (EXP-08):
    Optimise an additive policy bonus eta_uv(t) on specific edges that
    nudges tourists toward less-congested paths without enforcement.

Combined intervention (EXP-09) uses both A and B jointly.

See docs/03_methodology.md §Phase 4 for mathematical formulation.

EntranceMeteringOptimizer implemented 2026-06-02 (EXP-07, consolidated).
RoutingOptimizer implemented 2026-05-30 (EXP-08).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.evaluation.metrics import gini_coefficient

logger = logging.getLogger(__name__)


class EntranceMeteringOptimizer:
    """Entrance metering via arrival redistribution and a Pareto sweep over caps.

    This is the single source of truth for the EXP-07 metering intervention. A
    metering policy caps the arrival rate at a source node (e.g. the Outer
    Harbour Ferry Terminal) at ``R*`` and *redistributes* the excess arrivals to
    off-peak slots (conserving the daily total — redistribution, not deterrence).
    The optimiser is a 1-D parametric sweep over ``R*`` rather than gradient
    descent: the metering operator is piecewise-constant in ``R*`` (clamp +
    capacity-proportional fill), so a dense sweep traces the full
    peak-vs-visit Pareto frontier more robustly than a local gradient.

    Formulation (per cap R*):
        rho_eq(R*) = MFG fixed point under the metered arrival tensor g(R*)
        peak(R*)   = max_t rho_eq[t, bottleneck]
        visits(R*) = sum_{attractions, t} rho_eq * dt
    A cap is *feasible* if it achieves at least ``min_peak_reduction_pct`` peak
    reduction while losing at most ``max_visit_reduction_pct`` attraction-hours.

    Args:
        solver: Calibrated MFGSolver instance.
        source_col: Column index of the transit node whose arrivals are metered.
        bottleneck_idx: Column index of the node whose peak density is reduced
            (e.g. Ruins of St. Paul's). Defaults to 0.
        attraction_count: Number of leading columns that are heritage attractions
            (used for the total attraction-hours visit metric). Defaults to 10.
        damping: Fixed-point damping coefficient in (0, 1]. Defaults to 0.5.
    """

    def __init__(
        self,
        solver: Any,
        source_col: int,
        bottleneck_idx: int = 0,
        attraction_count: int = 10,
        damping: float = 0.5,
    ) -> None:
        self.solver = solver
        self.source_col = int(source_col)
        self.bottleneck_idx = int(bottleneck_idx)
        self.attraction_count = int(attraction_count)
        self.damping = float(damping)
        self.dt = float(solver.dt)

    @staticmethod
    def meter_arrivals(
        g: torch.Tensor,
        r_star: float,
        dt: float,
        source_col: int,
    ) -> torch.Tensor:
        """Redistribute excess arrivals at ``source_col`` above R* to slack steps.

        Total arrivals at ``source_col`` are conserved (redistribution, not
        deterrence); all other columns are unchanged. Excess steps are clamped to
        ``r_star`` and the freed tourist-hours are filled into the remaining
        headroom of slack steps in proportion to that headroom, guaranteeing no
        step exceeds ``r_star``.

        Args:
            g: Arrival tensor (T_steps x N_nodes).
            r_star: Cap on arrival rate (tourists / hour) at the source column.
            dt: Time step in hours.
            source_col: Column index of the transit node to meter.

        Returns:
            New arrival tensor of same shape; source_col peak capped at r_star.
        """
        g_out = g.clone()
        col = g_out[:, source_col]

        excess_mask = col > r_star
        if not excess_mask.any():
            return g_out  # r_star is above the current peak — no change

        # Total excess tourist-hours to redistribute
        excess = float(((col - r_star) * excess_mask.float() * dt).sum().item())
        if excess <= 0.0:
            return g_out

        # Clamp excess steps
        col_clamped = col.clone()
        col_clamped[excess_mask] = r_star
        g_out[:, source_col] = col_clamped

        # Capacity-proportional redistribution: fill each slack step in proportion
        # to its remaining headroom below r_star. Guarantees no step exceeds r_star.
        headroom = (r_star - col_clamped).clamp(min=0.0)
        total_capacity = float((headroom * dt).sum().item())

        if total_capacity > 0.0:
            # If total_capacity >= excess we fully redistribute (conservation holds);
            # if not (very aggressive cap with long tails) we fill all capacity and a
            # small residual is lost as the arrival day ends.
            fill_fraction = min(1.0, excess / total_capacity)
            g_out[:, source_col] = col_clamped + headroom * fill_fraction

        return g_out

    @staticmethod
    def compute_metrics(
        rho: torch.Tensor,
        g: torch.Tensor,
        dt: float,
        bottleneck_idx: int = 0,
        attraction_count: int = 10,
    ) -> dict[str, float]:
        """Compute intervention metrics from an equilibrium density trajectory.

        Args:
            rho: Density trajectory (T_steps x N_nodes).
            g: Arrival tensor (T_steps x N_nodes) used for this run.
            dt: Time step in hours.
            bottleneck_idx: Node index whose peak density is reported.
            attraction_count: Number of leading attraction columns.

        Returns:
            Dict with keys ``peak_density_ruins``, ``total_att_hours``,
            ``total_arrivals`` and ``gini``.
        """
        return {
            "peak_density_ruins": float(rho[:, bottleneck_idx].max().item()),
            "total_att_hours": float((rho[:, :attraction_count] * dt).sum().item()),
            "total_arrivals": float((g * dt).sum().item()),
            "gini": gini_coefficient(rho, t_idx=-1),
        }

    def _solve(self, g: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Solve the MFG fixed point for an arrival tensor (no grad)."""
        with torch.no_grad():
            rho, _, info = self.solver.fixed_point_iteration(g, damping=self.damping)
        return rho, info

    def evaluate_cap(
        self,
        g_base: torch.Tensor,
        r_star: float,
        baseline_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Evaluate a single metering cap against a precomputed baseline.

        Args:
            g_base: Unmetered baseline arrival tensor (T_steps x N_nodes).
            r_star: Cap to evaluate (tourists / hour) at the source column.
            baseline_metrics: Output of :meth:`compute_metrics` on the baseline.

        Returns:
            Record dict with the cap, its absolute metrics, the peak/visit
            reductions (%) versus baseline, and fixed-point diagnostics.
        """
        g_metered = self.meter_arrivals(g_base, r_star, self.dt, self.source_col)
        rho_met, info = self._solve(g_metered)
        m = self.compute_metrics(
            rho_met, g_metered, self.dt, self.bottleneck_idx, self.attraction_count
        )

        peak_red = 100.0 * (
            baseline_metrics["peak_density_ruins"] - m["peak_density_ruins"]
        ) / (baseline_metrics["peak_density_ruins"] + 1e-12)
        visit_red = 100.0 * (
            baseline_metrics["total_att_hours"] - m["total_att_hours"]
        ) / (baseline_metrics["total_att_hours"] + 1e-12)

        return {
            "r_star": r_star,
            "peak_density_ruins": m["peak_density_ruins"],
            "total_att_hours": m["total_att_hours"],
            "gini": m["gini"],
            "peak_reduction_pct": peak_red,
            "visit_reduction_pct": visit_red,
            "n_fp_iter": info["n_iter"],
            "fp_converged": info["converged"],
        }

    def baseline(self, g_base: torch.Tensor) -> tuple[dict[str, float], dict[str, Any]]:
        """Solve the unmetered baseline and return (metrics, fixed-point info)."""
        rho_base, info = self._solve(g_base)
        metrics = self.compute_metrics(
            rho_base, g_base, self.dt, self.bottleneck_idx, self.attraction_count
        )
        return metrics, info

    def sweep(
        self,
        g_base: torch.Tensor,
        r_values: list[float],
        baseline_metrics: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate a list of metering caps and return one record per cap.

        Args:
            g_base: Unmetered baseline arrival tensor (T_steps x N_nodes).
            r_values: Caps (tourists / hour) to evaluate, in any order.
            baseline_metrics: Optional precomputed baseline metrics; computed from
                ``g_base`` if omitted.

        Returns:
            List of records as produced by :meth:`evaluate_cap`, one per cap, in
            the order of ``r_values``. Each record additionally carries
            ``r_star_fraction`` relative to the baseline source-column peak.
        """
        if baseline_metrics is None:
            baseline_metrics, _ = self.baseline(g_base)
        r_peak_base = float(g_base[:, self.source_col].max().item())

        records: list[dict[str, Any]] = []
        for r_star in r_values:
            rec = self.evaluate_cap(g_base, r_star, baseline_metrics)
            rec["r_star_fraction"] = r_star / (r_peak_base + 1e-12)
            records.append(rec)
        return records

    def pareto_frontier(
        self,
        g_base: torch.Tensor,
        n_points: int = 25,
        r_min_fraction: float = 0.05,
        r_max_fraction: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Sweep caps over a fraction range and return the peak/visit trade-off.

        Args:
            g_base: Unmetered baseline arrival tensor (T_steps x N_nodes).
            n_points: Number of caps in the sweep. Defaults to 25.
            r_min_fraction: Smallest cap as a fraction of the source peak rate.
            r_max_fraction: Largest cap as a fraction of the source peak rate.

        Returns:
            List of records (see :meth:`sweep`), sorted by ascending ``r_star``.
        """
        r_peak_base = float(g_base[:, self.source_col].max().item())
        r_min = r_peak_base * float(r_min_fraction)
        r_max = r_peak_base * float(r_max_fraction)
        if n_points < 2:
            raise ValueError("n_points must be >= 2 for a Pareto sweep.")
        r_values = [
            r_min + (r_max - r_min) * i / (n_points - 1) for i in range(n_points)
        ]
        return self.sweep(g_base, r_values)


class RoutingOptimizer:
    """Optimise an additive policy bonus eta_uv to reduce bottleneck congestion.

    Models signage / app-based routing recommendations as an additive bonus on
    specific edges of the choice value, nudging the MFG equilibrium policy toward
    less-congested paths *without* enforcement (a behavioural nudge). The bonus
    enters ``Q_cont[v, w] = -gamma*D[v,w] + eta[v,w] + u[t+1, w]`` in both the HJB
    max and the FP softmax policy, so the equilibrium stays self-consistent
    (agents *perceive* recommended edges as more valuable and best-respond).

    The optimisation mirrors ``CalibrationEstimator.fit``'s one-step consistency
    gradient: each step runs the fixed point to convergence under torch.no_grad
    for a stable density, then a single HJB+FP pass with autograd active so the
    gradient flows to ``eta``. This is the validated, stable pattern for this
    solver (see the damping note for high beta).

    Formulation (system-wide max over a bottleneck set B):
        min_{eta}  smooth_max_{v in B, t}  rho_eq(eta)[t, v]
                   + lambda_l1 * ||eta||_1
                   + lambda_visit * relu(visit_reduction - delta)
    where ``eta`` is box-bounded to |eta| <= eta_max via a tanh squash (smooth,
    free-signed, zero at the origin) for interpretable recommendations.

    The objective is the *maximum* peak over the whole bottleneck set B (not a
    single node), so the optimiser cannot lower one node's peak by simply
    relocating the crowd onto a neighbour — doing so would raise that neighbour's
    peak and hence the objective. This forces genuine load-balancing (which
    *lowers* the spatial Gini) rather than a degenerate hot-spot swap. The
    visit-preservation penalty additionally stops the optimiser from cheaply
    lowering peaks by driving tourists to exit early.

    Args:
        solver: Calibrated MFGSolver instance (its ``routing_bonus`` attribute is
            overwritten during optimisation).
        g: Exogenous arrival tensor (T_steps, N_nodes).
        bottleneck_idx: Column index (or list of indices) of the nodes forming
            the bottleneck set B whose joint peak density is minimised. Passing a
            list implements the system-wide max objective; a single int reduces to
            the single-node objective.
        report_idx: Headline node index for reported peak reduction. Defaults to
            the first bottleneck index.
        candidate_edges: List of (src_idx, dst_idx) index pairs eligible for a
            bonus. If None, uses all finite-distance off-diagonal edges
            (D[i,j] < inf, i != j).
        attraction_count: Number of leading columns that are heritage attractions
            (used for the total-attraction-hours visit metric). Defaults to 10.
        node_labels: Optional list of human-readable node names (length N_nodes)
            used to format ``top_edges``. If None, integer indices are reported.
        eta_max: Box bound on |eta| (utility units). Defaults to 1.0.
        lambda_l1: L1 sparsity weight on eta. Defaults to 1e-3.
        lambda_visit: Weight on the visit-preservation penalty. Defaults to 1.0.
        delta: Maximum tolerated fractional reduction in total attraction-hours
            before the penalty activates. Defaults to 0.10.
        peak_temp: Temperature tau of the smooth max (log-sum-exp over time and
            the bottleneck nodes). Larger tau → closer to the hard max.
            Defaults to 10.0.
        damping: Fixed-point damping coefficient in (0, 1]. Defaults to 0.5.
    """

    def __init__(
        self,
        solver: Any,
        g: torch.Tensor,
        bottleneck_idx: int | list[int],
        report_idx: int | None = None,
        candidate_edges: list[tuple[int, int]] | None = None,
        attraction_count: int = 10,
        node_labels: list[str] | None = None,
        eta_max: float = 1.0,
        lambda_l1: float = 1e-3,
        lambda_visit: float = 1.0,
        delta: float = 0.10,
        peak_temp: float = 10.0,
        damping: float = 0.5,
    ) -> None:
        self.solver = solver
        self.g = g.float()
        if isinstance(bottleneck_idx, int):
            self.bottleneck_idxs = [int(bottleneck_idx)]
        else:
            self.bottleneck_idxs = [int(i) for i in bottleneck_idx]
        self.report_idx = (
            int(report_idx) if report_idx is not None else self.bottleneck_idxs[0]
        )
        self.attraction_count = int(attraction_count)
        self.node_labels = node_labels
        self.eta_max = float(eta_max)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_visit = float(lambda_visit)
        self.delta = float(delta)
        self.peak_temp = float(peak_temp)
        self.damping = float(damping)
        self.dt = float(solver.dt)
        self.N = int(solver.N_nodes)

        self.candidate_edges = (
            list(candidate_edges)
            if candidate_edges is not None
            else self._default_candidates()
        )
        if not self.candidate_edges:
            raise ValueError("No candidate edges for routing optimisation.")
        # Index tensors for scattering the flat parameter into an (N, N) matrix.
        self._src = torch.tensor([i for i, _ in self.candidate_edges], dtype=torch.long)
        self._dst = torch.tensor([j for _, j in self.candidate_edges], dtype=torch.long)

    def _default_candidates(self) -> list[tuple[int, int]]:
        """All finite-distance off-diagonal edges (real walking corridors)."""
        D = self.solver.D
        N = self.N
        edges: list[tuple[int, int]] = []
        for i in range(N):
            for j in range(N):
                if i != j and torch.isfinite(D[i, j]):
                    edges.append((i, j))
        return edges

    def _eta_matrix(self, raw: torch.Tensor) -> torch.Tensor:
        """Scatter the flat raw parameter into a bounded (N, N) bonus matrix.

        Uses ``eta = eta_max * tanh(raw)`` so the bonus is smooth, free-signed,
        bounded to (-eta_max, eta_max), and exactly zero when raw is zero.

        Args:
            raw: Flat parameter vector of length ``len(candidate_edges)``.

        Returns:
            (N, N) tensor with the bonus at candidate positions, 0 elsewhere.
        """
        values = self.eta_max * torch.tanh(raw)
        eta = torch.zeros(self.N, self.N, dtype=torch.float32)
        eta = eta.index_put((self._src, self._dst), values)
        return eta

    def _visit_hours(self, rho: torch.Tensor) -> torch.Tensor:
        """Total attraction-hours = sum over attraction nodes & time of rho*dt."""
        return (rho[:, : self.attraction_count] * self.dt).sum()

    def _smooth_peak(self, series: torch.Tensor) -> torch.Tensor:
        """Differentiable surrogate for max over all elements of ``series``.

        Computes (1/tau) * logsumexp(tau * series) over the flattened tensor, so
        it works for a 1-D time series or a 2-D (time x bottleneck nodes) block
        (the system-wide max over set B).
        """
        return torch.logsumexp(self.peak_temp * series.flatten(), dim=0) / self.peak_temp

    def optimise(
        self,
        n_steps: int = 300,
        lr: float = 5e-2,
        lr_decay: float = 0.99,
        grad_clip: float = 1.0,
        log_every: int = 50,
    ) -> dict[str, Any]:
        """Run gradient descent to find the optimal routing bonus.

        Args:
            n_steps: Optimisation steps. Defaults to 300.
            lr: Initial Adam learning rate. Defaults to 5e-2.
            lr_decay: ExponentialLR multiplicative decay per step. Defaults to 0.99.
            grad_clip: Gradient-norm clip. Defaults to 1.0.
            log_every: Log progress every this many steps. Defaults to 50.

        Returns:
            Dict with keys:
            - ``eta``: Optimal bonus matrix (N, N), detached.
            - ``peak_density_reduction_pct``: Peak reduction at the bottleneck vs
              the eta=0 baseline (positive = improvement).
            - ``visit_reduction_pct``: Reduction in total attraction-hours vs
              baseline (positive = fewer attraction-hours).
            - ``peak_baseline`` / ``peak_optimized``: Raw peak densities.
            - ``top_edges``: Top-5 candidate edges by |eta|, as dicts with
              ``src``, ``dst`` (labels or indices) and ``eta`` value.
            - ``loss_history``: List[float] of objective values per step.
        """
        B = torch.tensor(self.bottleneck_idxs, dtype=torch.long)

        # ── Baseline (eta = 0) ────────────────────────────────────────────────
        zero_eta = torch.zeros(self.N, self.N, dtype=torch.float32)
        with torch.no_grad():
            self.solver.routing_bonus = zero_eta
            rho_base, _, base_info = self.solver.fixed_point_iteration(
                self.g, damping=self.damping
            )
        system_peak_base = float(rho_base[:, B].max().item())
        report_peak_base = float(rho_base[:, self.report_idx].max().item())
        visits_base = float(self._visit_hours(rho_base).item())

        raw = nn.Parameter(torch.zeros(len(self.candidate_edges), dtype=torch.float32))
        optimizer = torch.optim.Adam([raw], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        loss_history: list[float] = []

        for step in range(n_steps):
            optimizer.zero_grad()
            eta = self._eta_matrix(raw)  # (N, N), differentiable

            # Step 1: fixed point to convergence (no grad) for a stable density.
            with torch.no_grad():
                self.solver.routing_bonus = eta.detach()
                rho_fp, _, fp_info = self.solver.fixed_point_iteration(
                    self.g, damping=self.damping
                )

            # Step 2: one HJB+FP pass with autograd so gradient flows to eta.
            self.solver.routing_bonus = eta
            u = self.solver.solve_hjb_backward(rho_fp)
            rho_pred = self.solver.solve_fp_forward(u, self.g)

            # Step 3: loss = normalised system-wide peak + L1 sparsity + visit pres.
            # The peak is normalised by the baseline system peak so it is O(1) at
            # the start, making lambda_l1/lambda_visit scale-free and meaningful.
            peak = self._smooth_peak(rho_pred[:, B])
            peak_term = peak / (system_peak_base + 1e-12)
            l1 = eta.abs().sum()
            visit_red_frac = (visits_base - self._visit_hours(rho_pred)) / (
                visits_base + 1e-12
            )
            visit_penalty = torch.relu(visit_red_frac - self.delta)
            loss = peak_term + self.lambda_l1 * l1 + self.lambda_visit * visit_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], grad_clip)
            optimizer.step()
            scheduler.step()
            loss_history.append(float(loss.item()))

            if (step + 1) % log_every == 0:
                logger.info(
                    "Routing opt step %d/%d: loss=%.4e | smooth_peak=%.4f | "
                    "L1=%.3f | visit_red=%.2f%% | fp_iter=%d",
                    step + 1, n_steps, loss.item(), float(peak.item()),
                    float(l1.item()), 100.0 * float(visit_red_frac.item()),
                    fp_info["n_iter"],
                )

        # ── Final evaluation at the optimised eta (full fixed point, no grad) ──
        with torch.no_grad():
            eta_final = self._eta_matrix(raw).detach()
            self.solver.routing_bonus = eta_final
            rho_opt, _, opt_info = self.solver.fixed_point_iteration(
                self.g, damping=self.damping
            )
        self.solver.routing_bonus = zero_eta  # leave solver in a clean state

        system_peak_opt = float(rho_opt[:, B].max().item())
        report_peak_opt = float(rho_opt[:, self.report_idx].max().item())
        visits_opt = float(self._visit_hours(rho_opt).item())
        peak_red_pct = 100.0 * (report_peak_base - report_peak_opt) / (report_peak_base + 1e-12)
        system_red_pct = 100.0 * (system_peak_base - system_peak_opt) / (system_peak_base + 1e-12)
        visit_red_pct = 100.0 * (visits_base - visits_opt) / (visits_base + 1e-12)

        if not opt_info["converged"]:
            logger.warning(
                "Optimised-eta fixed point did NOT converge (residual=%.2e); "
                "reported peaks may be unreliable.",
                opt_info["final_residual"],
            )

        return {
            "eta": eta_final,
            # Headline: peak reduction at the report node (e.g. ruins).
            "peak_density_reduction_pct": peak_red_pct,
            "peak_baseline": report_peak_base,
            "peak_optimized": report_peak_opt,
            # System-wide max over the bottleneck set B.
            "system_peak_reduction_pct": system_red_pct,
            "system_peak_baseline": system_peak_base,
            "system_peak_optimized": system_peak_opt,
            "visit_reduction_pct": visit_red_pct,
            "visits_baseline": visits_base,
            "visits_optimized": visits_opt,
            "baseline_converged": bool(base_info["converged"]),
            "optimized_converged": bool(opt_info["converged"]),
            "top_edges": self.top_edges(eta_final, k=5),
            "loss_history": loss_history,
        }

    def top_edges(self, eta: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k candidate edges by |eta| (most influential nudges).

        Args:
            eta: Bonus matrix (N, N).
            k: Number of edges to return. Defaults to 5.

        Returns:
            List of dicts sorted by descending |eta|, each with keys ``src``,
            ``dst`` (node labels if available, else indices) and ``eta`` (float).
        """
        scored = [
            (i, j, float(eta[i, j].item())) for (i, j) in self.candidate_edges
        ]
        scored.sort(key=lambda e: abs(e[2]), reverse=True)

        def label(idx: int) -> Any:
            if self.node_labels is not None and 0 <= idx < len(self.node_labels):
                return self.node_labels[idx]
            return idx

        return [
            {"src": label(i), "dst": label(j), "eta": val}
            for (i, j, val) in scored[: min(k, len(scored))]
        ]


def sample_beta_compliance(
    mean: float,
    concentration: float,
    n_samples: int,
    seed: int = 0,
) -> torch.Tensor:
    """Draw compliance fractions phi ~ Beta with a given mean and concentration.

    Parameterises the Beta in the intuitive (mean, concentration) form rather than
    (a, b): ``a = mean * kappa``, ``b = (1 - mean) * kappa`` where ``kappa`` is the
    concentration (larger kappa => tighter around the mean). This models a
    *population* of tourists with heterogeneous / uncertain willingness to follow a
    routing nudge, instead of one shared deterministic fraction.

    Args:
        mean: Population mean compliance in (0, 1).
        concentration: Beta concentration kappa = a + b (> 0).
        n_samples: Number of phi draws.
        seed: RNG seed for reproducibility.

    Returns:
        Float32 tensor of shape (n_samples,) with values in (0, 1).
    """
    if not 0.0 < mean < 1.0:
        raise ValueError(f"mean must be in (0, 1); got {mean}")
    if concentration <= 0.0:
        raise ValueError(f"concentration must be > 0; got {concentration}")
    a = mean * concentration
    b = (1.0 - mean) * concentration
    torch.manual_seed(seed)  # torch.distributions.Beta has no per-call generator
    samples = torch.distributions.Beta(
        torch.tensor(a), torch.tensor(b)
    ).sample((int(n_samples),))
    return samples.float()


def compliance_robustness_band(
    solver: Any,
    g: torch.Tensor,
    eta: torch.Tensor,
    report_idx: int,
    phi_samples: torch.Tensor,
    damping: float = 0.5,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
) -> dict[str, Any]:
    """Routing peak-reduction distribution under uncertain/heterogeneous compliance.

    The headline routing result assumes a single deterministic compliance fraction
    (and the ``phi=1`` perfect-compliance upper bound). This propagates a *whole
    distribution* of ``phi`` (e.g. from :func:`sample_beta_compliance`) through the
    two-type compliance equilibrium and reports the resulting band of peak
    reductions at ``report_idx`` — a deployable robustness interval rather than a
    point estimate.

    Args:
        solver: Calibrated MFGSolver (uses ``fixed_point_iteration_compliance``).
        g: Arrival tensor (T_steps, N_nodes).
        eta: Optimised routing bonus (N, N).
        report_idx: Node index whose peak reduction is reported (e.g. the ruins).
        phi_samples: 1-D tensor of compliance fractions to evaluate.
        damping: Fixed-point damping in (0, 1].
        percentiles: Percentiles (in %) to report from the reduction distribution.

    Returns:
        Dict with the per-sample reductions, summary statistics (mean, std,
        min/max), the requested percentiles, the mean sampled compliance, and the
        fraction of equilibria that converged.
    """
    phi_samples = torch.as_tensor(phi_samples, dtype=torch.float32).flatten()
    if phi_samples.numel() == 0:
        raise ValueError("phi_samples must be non-empty.")

    # Baseline (no routing) peak == the phi=0 compliance equilibrium.
    rho0, _ = solver.fixed_point_iteration_compliance(g, eta, phi=0.0, damping=damping)
    peak0 = float(rho0[:, report_idx].max().item())

    reductions: list[float] = []
    n_converged = 0
    for phi in phi_samples.tolist():
        phi = min(max(float(phi), 0.0), 1.0)  # Beta draws are already in (0,1)
        rho, info = solver.fixed_point_iteration_compliance(
            g, eta, phi=phi, damping=damping
        )
        peak = float(rho[:, report_idx].max().item())
        reductions.append(100.0 * (peak0 - peak) / (peak0 + 1e-12))
        n_converged += int(bool(info["converged"]))

    red = torch.tensor(reductions, dtype=torch.float32)
    pct_values = {
        f"reduction_p{p:g}": float(torch.quantile(red, p / 100.0).item())
        for p in percentiles
    }
    return {
        "n_samples": int(red.numel()),
        "phi_mean_sampled": float(phi_samples.mean().item()),
        "reduction_mean": float(red.mean().item()),
        "reduction_std": float(red.std(unbiased=False).item()),
        "reduction_min": float(red.min().item()),
        "reduction_max": float(red.max().item()),
        **pct_values,
        "frac_converged": n_converged / red.numel(),
        "reductions": reductions,
    }
