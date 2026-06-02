"""Tests for src.utils.synthetic_arrivals."""

from __future__ import annotations

import pytest
import torch

from src.utils.attractions import ATTRACTION_IDS, NODE_INDEX, TRANSIT_IDS
from src.utils.synthetic_arrivals import (
    build_arrival_tensor,
    build_observed_distribution,
)

# Canonical node order (all 13 nodes in NODE_INDEX order)
_NODE_ORDER = sorted(NODE_INDEX, key=lambda k: NODE_INDEX[k])

# Default simulation parameters
_T = 168   # 14 h at 5-min steps
_DT = 5 / 60
_N_TOURISTS = 10_000
_TRANSIT_SHARES = {"ferry_outer": 0.70, "border_gate": 0.25, "hotel_belt": 0.05}


@pytest.fixture()
def arrivals() -> torch.Tensor:
    return build_arrival_tensor(
        node_order=_NODE_ORDER,
        T_steps=_T,
        dt=_DT,
        n_tourists=_N_TOURISTS,
        peak_time_hours=2.0,
        sigma_hours=1.5,
        transit_shares=_TRANSIT_SHARES,
    )


# ---------------------------------------------------------------------------
# build_arrival_tensor
# ---------------------------------------------------------------------------


class TestBuildArrivalTensor:
    def test_shape(self, arrivals):
        assert arrivals.shape == (168, 13)

    def test_dtype(self, arrivals):
        assert arrivals.dtype == torch.float32

    def test_nonnegative(self, arrivals):
        assert arrivals.min().item() >= 0.0

    def test_attraction_nodes_are_zero(self, arrivals):
        """Arrival rate must be exactly zero for all heritage attraction nodes."""
        node_to_col = {nid: NODE_INDEX[nid] for nid in _NODE_ORDER}
        for att_id in ATTRACTION_IDS:
            col = node_to_col[att_id]
            assert arrivals[:, col].sum().item() == pytest.approx(0.0, abs=1e-6), (
                f"Attraction node '{att_id}' has non-zero arrivals"
            )

    def test_transit_nodes_nonzero(self, arrivals):
        """Transit nodes must receive arrivals."""
        node_to_col = {nid: NODE_INDEX[nid] for nid in _NODE_ORDER}
        for tid in TRANSIT_IDS:
            col = node_to_col[tid]
            assert arrivals[:, col].sum().item() > 0.0

    def test_total_arrivals_matches_n_tourists(self, arrivals):
        """Integrating arrival rates over time should recover n_tourists."""
        total = float((arrivals.sum(dim=0) * _DT).sum().item())
        assert total == pytest.approx(_N_TOURISTS, rel=0.01)

    def test_transit_shares_respected(self, arrivals):
        """Ferry : border : hotel arrivals should be ~ 70:25:5."""
        node_to_col = {nid: NODE_INDEX[nid] for nid in _NODE_ORDER}
        ferry_total = float((arrivals[:, node_to_col["ferry_outer"]] * _DT).sum())
        border_total = float((arrivals[:, node_to_col["border_gate"]] * _DT).sum())
        hotel_total = float((arrivals[:, node_to_col["hotel_belt"]] * _DT).sum())
        grand_total = ferry_total + border_total + hotel_total

        assert ferry_total / grand_total == pytest.approx(0.70, abs=0.01)
        assert border_total / grand_total == pytest.approx(0.25, abs=0.01)
        assert hotel_total / grand_total == pytest.approx(0.05, abs=0.01)

    def test_gaussian_peak_is_within_window(self, arrivals):
        """The peak arrival step should be near t=2h (step 24 at 5-min resolution)."""
        # Sum over all nodes at each step
        total_per_step = arrivals.sum(dim=1)
        peak_step = int(total_per_step.argmax().item())
        # Peak at 2h → step 2 / (5/60) = 24; allow ±3 steps tolerance
        assert abs(peak_step - 24) <= 3

    def test_raises_on_bad_shares(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            build_arrival_tensor(
                node_order=_NODE_ORDER,
                T_steps=10,
                dt=_DT,
                n_tourists=100,
                peak_time_hours=2.0,
                sigma_hours=1.5,
                transit_shares={"ferry_outer": 0.5, "border_gate": 0.1, "hotel_belt": 0.1},
            )

    def test_raises_on_unknown_node(self):
        with pytest.raises(ValueError, match="not found in node_order"):
            build_arrival_tensor(
                node_order=_NODE_ORDER,
                T_steps=10,
                dt=_DT,
                n_tourists=100,
                peak_time_hours=2.0,
                sigma_hours=1.5,
                transit_shares={"nonexistent_node": 1.0},
            )


# ---------------------------------------------------------------------------
# build_observed_distribution
# ---------------------------------------------------------------------------


class TestBuildObservedDistribution:
    @pytest.fixture()
    def obs(self) -> torch.Tensor:
        return build_observed_distribution(_NODE_ORDER)

    def test_length(self, obs):
        """Should have one entry per attraction (10 nodes, not 13)."""
        assert obs.shape == (len(ATTRACTION_IDS),)

    def test_sums_to_one(self, obs):
        assert float(obs.sum().item()) == pytest.approx(1.0, abs=1e-5)

    def test_nonnegative(self, obs):
        assert obs.min().item() >= 0.0

    def test_dtype(self, obs):
        assert obs.dtype == torch.float32

    def test_ruins_has_highest_share(self, obs):
        """Ruins of St. Paul's (2M visitors) should have the largest fraction."""
        # ATTRACTION_IDS ordering matches the order in ATTRACTION_NODES
        ruins_idx = ATTRACTION_IDS.index("ruins_st_pauls")
        assert int(obs.argmax().item()) == ruins_idx

    def test_senado_second(self, obs):
        """Senado Square (1.5M visitors) should be second largest."""
        senado_idx = ATTRACTION_IDS.index("senado_square")
        ruins_idx = ATTRACTION_IDS.index("ruins_st_pauls")
        # Senado should be larger than all others except ruins
        for i, nid in enumerate(ATTRACTION_IDS):
            if i not in (ruins_idx, senado_idx):
                assert obs[senado_idx] > obs[i]
