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

TODO (Week 7): implement EntranceMeteringOptimizer.
RoutingOptimizer implemented 2026-05-30 (EXP-08).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EntranceMeteringOptimizer:
    """Optimise arrival rate caps at source nodes to reduce bottleneck congestion.

    Formulation:
        min_{R*(t)} peak_density(rho_eq, bottleneck_node)
        subject to: mean_attractions_visited(rho_eq) >= (1-delta) * baseline

    where rho_eq is the MFG equilibrium under the capped arrival rate R*(t),
    and delta is the acceptable reduction in tourist experience (e.g. 0.10).

    Args:
        solver: Calibrated MFGSolver instance.
        source_node_ids: List of attraction_id strings for arrival source nodes.
        bottleneck_node_ids: List of attraction_id strings to minimise congestion at.
        baseline_rho: Baseline density (no intervention) for constraint computation.
        delta: Maximum allowed reduction in mean attractions visited. Default 0.10.
    """

    def __init__(
        self,
        solver: Any,
        source_node_ids: list[str],
        bottleneck_node_ids: list[str],
        baseline_rho: torch.Tensor,
        delta: float = 0.10,
    ) -> None:
        self.solver = solver
        self.source_node_ids = source_node_ids
        self.bottleneck_node_ids = bottleneck_node_ids
        self.baseline_rho = baseline_rho
        self.delta = delta

    def optimise(
        self,
        n_steps: int = 200,
        lr: float = 1e-2,
        r_min: float = 0.0,
        r_max: float = 1.0,
    ) -> dict[str, Any]:
        """Run gradient descent to find the optimal metering policy.

        Args:
            n_steps: Optimisation steps. Defaults to 200.
            lr: Learning rate. Defaults to 1e-2.
            r_min: Minimum allowed cap (fraction of uncapped arrival rate).
            r_max: Maximum allowed cap (≤1.0 means only restricting, not adding).

        Returns:
            Dict with keys:
            - ``r_star``: Optimal cap tensor (T_steps, len(source_node_ids)).
            - ``peak_density_reduction_pct``: Reduction vs baseline.
            - ``mean_attractions_visited_reduction_pct``: Visitor experience cost.
            - ``loss_history``: List of objective values.

        Raises:
            NotImplementedError: Until Week 7 implementation.
        """
        # TODO (Week 7): implement entrance metering optimisation.
        raise NotImplementedError(
            "EntranceMeteringOptimizer.optimise — implement in Week 7."
        )

    def pareto_frontier(
        self,
        n_points: int = 20,
    ) -> list[dict[str, float]]:
        """Compute the Pareto frontier of peak density vs. attractions visited.

        Sweeps the constraint ``delta`` over a range and records the optimal
        trade-off at each point.

        Args:
            n_points: Number of frontier points. Defaults to 20.

        Returns:
            List of dicts with keys ``delta``, ``peak_density``,
            ``mean_attractions_visited``.

        Raises:
            NotImplementedError: Until Week 7 implementation.
        """
        # TODO (Week 7): implement Pareto sweep.
        raise NotImplementedError(
            "EntranceMeteringOptimizer.pareto_frontier — implement in Week 7."
        )


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
