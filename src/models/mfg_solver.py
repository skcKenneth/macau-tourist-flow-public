"""MFG solver: backward HJB + forward Fokker-Planck + fixed-point iteration.

Mathematical formulation: see docs/03_methodology.md §Phase 2.

All density/cost tensors have shape (T_steps, N_nodes) where:
- T_steps = number of discrete time steps (typically T/dt = 168 for 14h at 5min)
- N_nodes  = number of graph nodes (13 in the Macau heritage graph)

Notation follows the project convention in CLAUDE.md:
- rho_v(t) : tourist density at node v at time t  (shape: T_steps × N_nodes)
- u_v(t)   : cost-to-go at node v at time t       (shape: T_steps × N_nodes)
- g_v(t)   : exogenous arrival rate at node v      (shape: T_steps × N_nodes)

Discrete-time formulation
--------------------------
At each step t, a tourist at node v:
  1. Receives running reward: alpha[v]*dt − beta*rho[t,v]*dt
  2. Chooses destination w ∈ {0, …, N-1} ∪ {exit}, paying walk cost gamma*D[v,w].
     Choosing w=v (self-loop, D[v,v]=0) means "stay at v".

HJB (backward, hard max):
    u[t, v] = alpha[v]*dt − beta*rho[t,v]*dt + max(max_w {−gamma*D[v,w] + u[t+1,w]}, 0)
    u[T-1, v] = alpha[v]*dt − beta*rho[T-1,v]*dt   (terminal: no continuation)

Policy (soft argmax for FP, differentiable):
    Q_cont[v, w] = −gamma*D[v,w] + u[t+1,w]
    logits[v, :] = cat([Q_cont[v,:], 0_exit])          (N+1 options)
    pi[t, v, :] = softmax(logits[v,:] / epsilon)        (N+1 probs)

FP (forward):
    rho[t+1] = rho[t] @ move_pi[t] + g[t] * dt
    where move_pi[t][u,v] = pi[t,u,v]  for v < N  (N×N sub-matrix)

Fixed-point iteration:
    rho^(0) = 0
    rho^(k+1) = FP(HJB(rho^(k)), g)   until ||rho^(k+1) − rho^(k)||_inf < tol
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import torch

logger = logging.getLogger(__name__)


class MFGSolver:
    """Solve the MFG Nash equilibrium on the Macau heritage graph.

    Couples a backward Hamilton-Jacobi-Bellman (HJB) equation for the
    individual cost-to-go u_v(t) with a forward Fokker-Planck (FP) equation
    for the crowd density rho_v(t). Nash equilibrium is found via
    fixed-point iteration (alternating HJB/FP solves until convergence).

    Args:
        G: Simplified attraction graph (networkx DiGraph or Graph). Nodes must
            be sortable (e.g. integers 0..N-1). Edge attribute ``length`` is
            used as walking distance in metres.
        params: Dict of model parameters:
            - ``alpha``: Per-node attractiveness (tensor or list, length N_nodes).
              Units: utility per hour. Must be non-negative.
            - ``beta``: Congestion cost coefficient (scalar ≥ 0).
              Units: utility per (density × hour).
            - ``gamma``: Walking cost per metre (scalar ≥ 0).
              Units: utility per metre.
        dt: Time step in hours. Defaults to 5/60 (5 minutes).
        T: Time horizon in hours. Defaults to 14 (08:00–22:00).
        epsilon: Softmax temperature for smooth policy. Larger ε → more
            uniform policy; smaller ε → closer to greedy argmax. Defaults to 0.1.
        tol: Convergence tolerance for fixed-point iteration (L∞ norm of
            rho change). Defaults to 1e-4.
        max_iter: Maximum fixed-point iterations. Defaults to 100.
        node_order: Optional list of node IDs defining tensor column ordering.
            If None, uses sorted(G.nodes()).
        routing_bonus: Optional additive policy bonus eta of shape (N, N) added
            to the edge choice value Q_cont[v, w] in both the HJB max and the FP
            softmax policy (see EXP-08 / docs/03_methodology.md §Phase 4). Models
            an informational nudge (signage / app recommendation) that biases
            tourists toward recommended edges. If None, initialised to zeros
            (no-op — behaviour identical to no bonus). May be reassigned after
            construction via ``solver.routing_bonus = eta`` (e.g. a learnable
            nn.Parameter during optimisation).
    """

    def __init__(
        self,
        G: nx.Graph,
        params: dict[str, Any],
        dt: float = 5 / 60,
        T: float = 14.0,
        epsilon: float = 0.1,
        tol: float = 1e-4,
        max_iter: int = 100,
        node_order: list | None = None,
        routing_bonus: torch.Tensor | None = None,
    ) -> None:
        self.G = G
        self.dt = dt
        self.T = T
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        self.T_steps = int(round(T / dt))
        self.N_nodes = G.number_of_nodes()

        self.node_order = node_order or sorted(G.nodes())
        assert len(self.node_order) == self.N_nodes, (
            f"node_order length {len(self.node_order)} ≠ graph nodes {self.N_nodes}"
        )
        self._node_to_idx: dict = {n: i for i, n in enumerate(self.node_order)}

        # Convert params to tensors
        self.params = self._parse_params(params)

        # Build distance matrix D[v, w] in metres (shape: N×N, D[v,v]=0)
        self.D = self._build_distance_matrix()

        # Routing-recommendation bonus eta[v, w] (N×N). Default zeros = no-op.
        self.routing_bonus = self._init_routing_bonus(routing_bonus)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _parse_params(self, raw: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert raw params (lists, scalars, tensors) to float32 tensors."""
        alpha = raw["alpha"]
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        alpha = alpha.float()
        assert alpha.shape == (self.N_nodes,), (
            f"alpha must have shape ({self.N_nodes},), got {alpha.shape}"
        )

        beta = torch.tensor(float(raw["beta"]), dtype=torch.float32)
        gamma = torch.tensor(float(raw["gamma"]), dtype=torch.float32)
        return {"alpha": alpha, "beta": beta, "gamma": gamma}

    def _init_routing_bonus(self, eta: torch.Tensor | None) -> torch.Tensor:
        """Validate / default the (N, N) routing bonus matrix.

        Args:
            eta: Optional (N, N) tensor. If None, returns zeros (no-op).

        Returns:
            Float32 tensor of shape (N, N).
        """
        N = self.N_nodes
        if eta is None:
            return torch.zeros(N, N, dtype=torch.float32)
        if not isinstance(eta, torch.Tensor):
            eta = torch.tensor(eta, dtype=torch.float32)
        assert eta.shape == (N, N), (
            f"routing_bonus must have shape ({N}, {N}), got {tuple(eta.shape)}"
        )
        return eta.float()

    def _build_distance_matrix(self) -> torch.Tensor:
        """Build (N, N) walking distance matrix from graph edge attributes.

        Returns:
            Float32 tensor of shape (N, N). D[v, v] = 0 (self-loop / stay).
            D[v, w] = walking distance from v to w in metres for (v,w) in E.
            D[v, w] = inf if no direct edge exists (handled as zero-prob move).
        """
        N = self.N_nodes
        D = torch.full((N, N), float("inf"), dtype=torch.float32)
        D.fill_diagonal_(0.0)  # stay action has zero walk cost

        is_multi = isinstance(self.G, (nx.MultiDiGraph, nx.MultiGraph))

        if is_multi:
            for u, v, data in self.G.edges(data=True):
                i = self._node_to_idx[u]
                j = self._node_to_idx[v]
                length = float(data.get("length", 1.0))
                if length < D[i, j]:
                    D[i, j] = length
        else:
            for u, v, data in self.G.edges(data=True):
                i = self._node_to_idx[u]
                j = self._node_to_idx[v]
                D[i, j] = float(data.get("length", 1.0))

        # Symmetrize for undirected graphs
        if not self.G.is_directed():
            D = torch.minimum(D, D.T)

        return D

    # ------------------------------------------------------------------
    # Core solver
    # ------------------------------------------------------------------

    def solve_hjb_backward(self, rho: torch.Tensor) -> torch.Tensor:
        """Solve the backward HJB equation given a density trajectory.

        Given the density rho_v(t) (from the previous FP solve or an initial
        guess), computes the cost-to-go u_v(t) by integrating backward from T.

        Discrete HJB:
            u[t, v] = alpha[v]*dt − beta*rho[t,v]*dt
                      + max(max_w {−gamma*D[v,w] + u[t+1,w]}, 0)

        The terminal condition u[T-1, v] uses only the immediate reward
        (no continuation), and u[T, v] = 0 is used implicitly.

        Args:
            rho: Density tensor of shape (T_steps, N_nodes).

        Returns:
            Cost-to-go tensor u of shape (T_steps, N_nodes).
        """
        alpha = self.params["alpha"]   # (N,)
        beta = self.params["beta"]     # scalar tensor
        gamma = self.params["gamma"]   # scalar tensor
        dt = self.dt
        N = self.N_nodes
        T = self.T_steps

        if not torch.is_grad_enabled():
            # Fast path: pre-allocate and use in-place indexed assignments.
            # Safe here because params are detached inside torch.no_grad().
            u = torch.zeros(T, N, dtype=torch.float32)
            u[T - 1] = alpha * dt - beta * rho[T - 1] * dt
            for t in range(T - 2, -1, -1):
                Q_cont = -gamma * self.D + self.routing_bonus + u[t + 1].unsqueeze(0)
                Q_cont = torch.where(
                    self.D == float("inf"), torch.full_like(Q_cont, -1e9), Q_cont
                )
                continuation = torch.clamp(Q_cont.max(dim=1).values, min=0.0)
                u[t] = alpha * dt - beta * rho[t] * dt + continuation
            return u

        # Autograd-safe path: collect steps in a list to avoid in-place ops
        # that break the computation graph when params are nn.Parameters.
        u_steps: list[torch.Tensor] = [None] * T
        u_steps[T - 1] = alpha * dt - beta * rho[T - 1] * dt
        for t in range(T - 2, -1, -1):
            # Q_cont[v, w] = -gamma * D[v, w] + eta[v, w] + u[t+1, w]
            Q_cont = -gamma * self.D + self.routing_bonus + u_steps[t + 1].unsqueeze(0)  # (N, N)
            Q_cont = torch.where(
                self.D == float("inf"), torch.full_like(Q_cont, -1e9), Q_cont
            )
            continuation = torch.clamp(Q_cont.max(dim=1).values, min=0.0)
            u_steps[t] = alpha * dt - beta * rho[t] * dt + continuation
        return torch.stack(u_steps, dim=0)  # (T, N)

    def solve_fp_forward(self, u: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Solve the forward Fokker-Planck equation given a cost-to-go trajectory.

        Derives the softmax policy from u[t+1] and integrates density forward.

        Policy:
            Q_cont[v, w] = −gamma*D[v,w] + u[t+1, w]
            pi[t, v, :] = softmax( cat([Q_cont[v,:], 0_exit]) / epsilon )

        FP step:
            rho[t+1, v] = sum_u rho[t,u] * pi[t, u, v]  +  g[t,v] * dt

        Args:
            u: Cost-to-go tensor of shape (T_steps, N_nodes).
            g: Exogenous arrival rate tensor of shape (T_steps, N_nodes).

        Returns:
            Density trajectory rho of shape (T_steps, N_nodes), non-negative.
        """
        gamma = self.params["gamma"]
        dt = self.dt
        N = self.N_nodes
        T = self.T_steps
        eps = self.epsilon

        if not torch.is_grad_enabled():
            # Fast path: pre-allocate tensor with in-place indexed assignments.
            rho = torch.zeros(T, N, dtype=torch.float32)
            for t in range(T - 1):
                Q_cont = -gamma * self.D + self.routing_bonus + u[t + 1].unsqueeze(0)
                Q_cont = torch.where(
                    self.D == float("inf"), torch.full_like(Q_cont, -1e9), Q_cont
                )
                Q_exit = torch.zeros(N, 1, dtype=torch.float32)
                logits = torch.cat([Q_cont, Q_exit], dim=1)
                pi = torch.softmax(logits / eps, dim=1)
                rho[t + 1] = rho[t] @ pi[:, :N] + g[t] * dt
            return rho

        # Autograd-safe path: collect steps in a list to avoid in-place ops.
        rho_steps: list[torch.Tensor] = [torch.zeros(N, dtype=torch.float32)]
        for t in range(T - 1):
            # Q_cont[v, w] = -gamma * D[v, w] + eta[v, w] + u[t+1, w]
            Q_cont = -gamma * self.D + self.routing_bonus + u[t + 1].unsqueeze(0)  # (N, N)
            Q_cont = torch.where(
                self.D == float("inf"), torch.full_like(Q_cont, -1e9), Q_cont
            )
            Q_exit = torch.zeros(N, 1, dtype=torch.float32)
            logits = torch.cat([Q_cont, Q_exit], dim=1)   # (N, N+1)
            pi = torch.softmax(logits / eps, dim=1)        # (N, N+1)
            rho_steps.append(rho_steps[t] @ pi[:, :N] + g[t] * dt)
        return torch.stack(rho_steps, dim=0)  # (T, N)

    def fixed_point_iteration(
        self,
        g: torch.Tensor,
        rho_init: torch.Tensor | None = None,
        damping: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Run HJB-FP fixed-point iteration to find the MFG Nash equilibrium.

        Alternates between:
        1. Solve HJB backward given current rho → get u
        2. Solve FP forward given u and g → get new rho
        3. Apply damped update: rho = (1-damping)*rho + damping*rho_new
        4. Check convergence: ||rho_new − rho_prev||_∞ < tol

        Args:
            g: Exogenous arrival rate tensor of shape (T_steps, N_nodes).
                Units: tourists per hour at each node at each step.
            rho_init: Initial density guess (T_steps, N_nodes). If None, starts
                with rho = 0 (no tourists), which is appropriate when g > 0.
            damping: Mixing coefficient in (0, 1]. damping=1.0 (default) is
                standard Picard iteration. damping<1.0 damps oscillations at
                high congestion (β≥0.5) at the cost of slower convergence.

        Returns:
            Tuple (rho_eq, u_eq, info) where:
            - rho_eq: Equilibrium density (T_steps, N_nodes).
            - u_eq: Equilibrium cost-to-go (T_steps, N_nodes).
            - info: Dict with keys:
                - ``n_iter`` (int): number of iterations run.
                - ``converged`` (bool): True if tol was reached.
                - ``final_residual`` (float): L∞ norm of last rho update.
        """
        if rho_init is not None:
            rho = rho_init.clone().float()
        else:
            rho = torch.zeros(self.T_steps, self.N_nodes, dtype=torch.float32)

        g = g.float()

        final_residual = float("inf")
        n_iter = 0
        converged = False
        u = torch.zeros_like(rho)

        for k in range(self.max_iter):
            rho_prev = rho
            u = self.solve_hjb_backward(rho)
            rho_new = self.solve_fp_forward(u, g)

            # Damped update: blend old and new density to suppress oscillations
            rho = (1.0 - damping) * rho_prev + damping * rho_new
            residual = float((rho - rho_prev).abs().max().item())
            logger.debug("FP iter %d: residual=%.4e", k + 1, residual)

            n_iter = k + 1
            final_residual = residual

            if residual < self.tol:
                converged = True
                logger.info(
                    "MFG fixed-point converged in %d iterations (residual=%.2e)",
                    n_iter,
                    final_residual,
                )
                break

        if not converged:
            logger.warning(
                "MFG fixed-point did NOT converge after %d iterations "
                "(final residual=%.2e, tol=%.2e)",
                self.max_iter,
                final_residual,
                self.tol,
            )

        return rho, u, {
            "n_iter": n_iter,
            "converged": converged,
            "final_residual": final_residual,
        }

    # ------------------------------------------------------------------
    # Compliance-aware (two-type) forward solve — EXP-09 / Goal D
    # ------------------------------------------------------------------

    def _move_matrix(self, u_next: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """Build the (N, N) movement sub-matrix for one step given u[t+1] and a bonus.

        ``move[v, w]`` is the probability a tourist at ``v`` moves to ``w`` under the
        softmax policy with perceived edge bonus ``eta`` (the exit column is dropped).
        """
        gamma = self.params["gamma"]
        N = self.N_nodes
        Q = -gamma * self.D + eta + u_next.unsqueeze(0)
        Q = torch.where(self.D == float("inf"), torch.full_like(Q, -1e9), Q)
        logits = torch.cat([Q, torch.zeros(N, 1, dtype=torch.float32)], dim=1)
        pi = torch.softmax(logits / self.epsilon, dim=1)
        return pi[:, :N]

    def fixed_point_iteration_compliance(
        self,
        g: torch.Tensor,
        eta: torch.Tensor,
        phi: float,
        rho_init: torch.Tensor | None = None,
        damping: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Two-type MFG equilibrium under partial routing compliance.

        A fraction ``phi`` of tourists perceive and best-respond to the routing
        bonus ``eta`` (the "compliers"); the remaining ``1 - phi`` ignore it and
        best-respond to the true utility. Both types share the single congestion
        field ``rho`` (the mean-field coupling), so at each iteration we solve the
        HJB twice against the *same* density and blend the forward policies:

            rho[t+1] = phi * rho[t] @ move(u_comply, eta)
                       + (1 - phi) * rho[t] @ move(u_ignore, 0)
                       + g[t] * dt

        ``phi = 1`` recovers the full-compliance routing equilibrium; ``phi = 0``
        recovers the no-intervention baseline. Evaluation-only (no autograd).

        Args:
            g: Arrival tensor (T_steps, N_nodes).
            eta: Routing bonus (N, N).
            phi: Compliance fraction in [0, 1].
            rho_init: Optional initial density.
            damping: Fixed-point damping in (0, 1].

        Returns:
            Tuple (rho_eq, info) with info keys ``n_iter``/``converged``/``final_residual``.
        """
        N = self.N_nodes
        T = self.T_steps
        dt = self.dt
        g = g.float()
        eta = eta.float() if eta is not None else torch.zeros(N, N, dtype=torch.float32)
        zero_eta = torch.zeros(N, N, dtype=torch.float32)
        saved_bonus = self.routing_bonus

        rho = rho_init.clone().float() if rho_init is not None else torch.zeros(T, N, dtype=torch.float32)
        converged = False
        final_residual = float("inf")
        n_iter = 0

        with torch.no_grad():
            for k in range(self.max_iter):
                rho_prev = rho

                self.routing_bonus = eta
                u_c = self.solve_hjb_backward(rho)
                self.routing_bonus = zero_eta
                u_i = self.solve_hjb_backward(rho)

                rho_new = torch.zeros(T, N, dtype=torch.float32)
                for t in range(T - 1):
                    move = phi * self._move_matrix(u_c[t + 1], eta) + (1.0 - phi) * self._move_matrix(u_i[t + 1], zero_eta)
                    rho_new[t + 1] = rho_new[t] @ move + g[t] * dt

                rho = (1.0 - damping) * rho_prev + damping * rho_new
                final_residual = float((rho - rho_prev).abs().max().item())
                n_iter = k + 1
                if final_residual < self.tol:
                    converged = True
                    break

        self.routing_bonus = saved_bonus
        return rho, {"n_iter": n_iter, "converged": converged, "final_residual": final_residual}
