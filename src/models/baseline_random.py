"""Random-walk baseline: time-inhomogeneous Markov chain on the attraction graph.

Used in EXP-02 to demonstrate that without optimization/congestion avoidance,
the density distribution does NOT match observed visitor concentration —
thereby motivating the full MFG formulation.

The baseline models each tourist as following a random walk with transition
probabilities proportional to 1/edge_length (shorter paths preferred),
without any congestion awareness or utility maximisation.
"""

from __future__ import annotations

import logging

import networkx as nx
import torch

from src.utils.io import set_all_seeds

logger = logging.getLogger(__name__)


class RandomWalkBaseline:
    """Time-inhomogeneous random walk on the attraction graph.

    Tourists enter the graph at source nodes according to the exogenous
    arrival rate g_v(t) and move to neighbouring nodes with probability
    proportional to 1/edge_weight (uniform if weights are equal).

    This is deliberately the simplest possible model — it ignores congestion,
    utility, and strategic behaviour. Its purpose is to quantify how much of
    the observed concentration can be explained by graph topology alone.

    Args:
        G: Attraction graph (networkx DiGraph). Edge attribute ``length``
            used for transition probabilities (shorter edges preferred).
            Typically the 13-node fully-connected subgraph from run_exp02.
        arrival_rates: Tensor of shape (T_steps, N_nodes) giving the exogenous
            arrival rate at each node at each time step (tourists per hour).
        dt: Time step in hours. Defaults to 5/60 (5 minutes).
        node_order: List of node IDs defining the tensor column ordering.
            If None, uses sorted(G.nodes()).
        exit_rate: Fraction of tourists at each node that leave the system per
            time step (uniform across all nodes). Defaults to 0.02, giving a
            mean dwell time of 50 steps = ~4 hours.
        weight_attr: Edge attribute to use as distance weight. Defaults to
            ``"length"`` (metres).
    """

    def __init__(
        self,
        G: nx.DiGraph,
        arrival_rates: torch.Tensor,
        dt: float = 5 / 60,
        node_order: list | None = None,
        exit_rate: float = 0.02,
        weight_attr: str = "length",
    ) -> None:
        self.G = G
        self.arrival_rates = arrival_rates
        self.dt = dt
        self.node_order = node_order or sorted(G.nodes())
        self.exit_rate = exit_rate
        self.weight_attr = weight_attr
        self.T_steps, self.N_nodes = arrival_rates.shape
        # Build at construction time so __init__ fails fast on bad graphs
        self.P = self._build_transition_matrix()

    def _build_transition_matrix(self) -> torch.Tensor:
        """Build the (N_nodes, N_nodes) row-stochastic transition matrix.

        Transition probabilities are proportional to 1/length for each
        outgoing edge. Nodes with no outgoing edges (sinks) stay in place
        via a self-loop.

        Returns:
            Float32 tensor of shape (N_nodes, N_nodes) where P[i, j] is the
            probability of moving from node i to node j in one step.
        """
        N = self.N_nodes
        node_to_idx = {nid: i for i, nid in enumerate(self.node_order)}
        P = torch.zeros(N, N, dtype=torch.float32)

        for i, nid in enumerate(self.node_order):
            if nid not in self.G:
                # Node not in graph — self-loop
                logger.warning("Node '%s' not in graph; adding self-loop.", nid)
                P[i, i] = 1.0
                continue

            neighbors = list(self.G.successors(nid))
            if not neighbors:
                logger.warning(
                    "Node '%s' has no outgoing edges (sink); adding self-loop.", nid
                )
                P[i, i] = 1.0
                continue

            weights: list[float] = []
            valid_neighbors: list[int] = []
            for nb in neighbors:
                edge_data = self.G.get_edge_data(nid, nb)
                length = edge_data.get(self.weight_attr, 1.0) if edge_data else 1.0
                if length <= 0:
                    length = 1.0  # guard against zero/negative lengths
                weights.append(1.0 / length)
                valid_neighbors.append(node_to_idx[nb])

            w_sum = sum(weights)
            for j, w in zip(valid_neighbors, weights):
                P[i, j] = w / w_sum

        # Sanity check: every row must sum to 1
        row_sums = P.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones(N), atol=1e-5):
            bad = (row_sums - 1.0).abs().argmax().item()
            raise RuntimeError(
                f"Transition matrix row {bad} sums to {row_sums[bad]:.6f}, "
                "not 1.0. Check graph structure."
            )

        logger.info(
            "Transition matrix built: shape=%s, weight_attr='%s'",
            list(P.shape),
            self.weight_attr,
        )
        return P

    def simulate(self, seed: int = 42) -> torch.Tensor:
        """Simulate the random-walk baseline and return the density trajectory.

        At each time step:
          1. A fraction ``exit_rate`` of current tourists leaves the system.
          2. Remaining tourists redistribute according to the transition matrix P.
          3. New tourists arrive according to ``arrival_rates[t] * dt``.

        Args:
            seed: RNG seed for reproducibility (passed to set_all_seeds).

        Returns:
            Raw tourist count tensor of shape (T_steps, N_nodes) where entry
            [t, v] is the number of tourists at node v at the *start* of step t
            (before movement and new arrivals at step t are applied).
            Normalize rows by their sum to obtain density fractions.
        """
        set_all_seeds(seed)

        rho = torch.zeros(self.N_nodes, dtype=torch.float32)
        trajectory = torch.zeros(self.T_steps, self.N_nodes, dtype=torch.float32)
        P_T = self.P.T  # precompute transpose: P_T[j, i] = P[i, j]

        for t in range(self.T_steps):
            trajectory[t] = rho
            # Exit: fraction (exit_rate) of tourists leave the system
            rho_stay = rho * (1.0 - self.exit_rate)
            # Redistribute according to transition matrix
            rho = P_T @ rho_stay
            # Inject new arrivals (rate × dt = tourists this step)
            rho = rho + self.arrival_rates[t] * self.dt
            # Clamp to non-negative (defensive against float rounding)
            rho = rho.clamp(min=0.0)

        return trajectory
