"""Unit tests for src/utils/attractions.py."""

from __future__ import annotations

import pytest

from src.utils.attractions import (
    ATTRACTION_IDS,
    ATTRACTION_NODES,
    BOTTLENECK_IDS,
    NODE_BY_ID,
    NODE_INDEX,
    TRANSIT_IDS,
    AttractionNode,
    NodeType,
    get_coords_array,
)

# Expected number of nodes (10 attractions + 3 transit)
EXPECTED_N_NODES = 13
EXPECTED_N_ATTRACTIONS = 10
EXPECTED_N_TRANSIT = 3

# Macau WGS84 bounding box (generous padding)
LAT_MIN, LAT_MAX = 22.18, 22.23
LON_MIN, LON_MAX = 113.52, 113.56


class TestNodeRegistry:
    """Tests for the ATTRACTION_NODES list and its derived lookups."""

    def test_total_node_count(self) -> None:
        assert len(ATTRACTION_NODES) == EXPECTED_N_NODES

    def test_attraction_count(self) -> None:
        assert len(ATTRACTION_IDS) == EXPECTED_N_ATTRACTIONS

    def test_transit_count(self) -> None:
        assert len(TRANSIT_IDS) == EXPECTED_N_TRANSIT

    def test_all_ids_unique(self) -> None:
        ids = [n.node_id for n in ATTRACTION_NODES]
        assert len(ids) == len(set(ids)), "Duplicate node_id detected"

    def test_node_by_id_covers_all_nodes(self) -> None:
        assert set(NODE_BY_ID.keys()) == {n.node_id for n in ATTRACTION_NODES}

    def test_node_index_covers_all_nodes(self) -> None:
        assert set(NODE_INDEX.keys()) == {n.node_id for n in ATTRACTION_NODES}

    def test_node_index_values_are_unique_and_sequential(self) -> None:
        indices = sorted(NODE_INDEX.values())
        assert indices == list(range(EXPECTED_N_NODES))

    def test_attractions_listed_before_transit(self) -> None:
        """Attractions must come first in ATTRACTION_NODES for tensor indexing."""
        types = [n.node_type for n in ATTRACTION_NODES]
        # Find first transit index
        transit_indices = [i for i, t in enumerate(types) if t == NodeType.TRANSIT]
        attraction_indices = [
            i for i, t in enumerate(types) if t == NodeType.ATTRACTION
        ]
        if transit_indices and attraction_indices:
            assert max(attraction_indices) < min(transit_indices), (
                "All attraction nodes must appear before transit nodes "
                "for stable tensor indexing."
            )


class TestNodeCoordinates:
    """Tests for coordinate validity of all nodes."""

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_latitude_in_macau_bounds(self, node: AttractionNode) -> None:
        assert LAT_MIN <= node.lat <= LAT_MAX, (
            f"{node.node_id}: lat={node.lat} outside [{LAT_MIN}, {LAT_MAX}]"
        )

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_longitude_in_macau_bounds(self, node: AttractionNode) -> None:
        assert LON_MIN <= node.lon <= LON_MAX, (
            f"{node.node_id}: lon={node.lon} outside [{LON_MIN}, {LON_MAX}]"
        )

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_coordinates_are_finite(self, node: AttractionNode) -> None:
        import math

        assert math.isfinite(node.lat), f"{node.node_id}: lat is not finite"
        assert math.isfinite(node.lon), f"{node.node_id}: lon is not finite"


class TestNodeFields:
    """Tests that all required string fields are populated."""

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_node_id_nonempty(self, node: AttractionNode) -> None:
        assert node.node_id.strip() != ""

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_name_en_nonempty(self, node: AttractionNode) -> None:
        assert node.name_en.strip() != ""

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_name_zh_nonempty(self, node: AttractionNode) -> None:
        assert node.name_zh.strip() != ""

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_annual_visitors_positive(self, node: AttractionNode) -> None:
        assert node.annual_visitors_est > 0

    @pytest.mark.parametrize("node", ATTRACTION_NODES, ids=lambda n: n.node_id)
    def test_node_type_is_valid_enum(self, node: AttractionNode) -> None:
        assert isinstance(node.node_type, NodeType)


class TestBottleneckDefinition:
    """Tests for bottleneck flag consistency."""

    def test_ruins_st_pauls_is_bottleneck(self) -> None:
        """Ruins of St. Paul's is the documented primary bottleneck."""
        assert NODE_BY_ID["ruins_st_pauls"].is_bottleneck

    def test_at_least_one_bottleneck(self) -> None:
        assert len(BOTTLENECK_IDS) >= 1

    def test_all_bottlenecks_are_attractions(self) -> None:
        """Transit nodes should not be flagged as bottlenecks."""
        for bid in BOTTLENECK_IDS:
            assert NODE_BY_ID[bid].node_type == NodeType.ATTRACTION, (
                f"{bid} is a transit node but flagged as bottleneck"
            )


class TestGetCoordsArray:
    """Tests for get_coords_array helper."""

    def test_shape(self) -> None:
        import numpy as np

        arr = get_coords_array()
        assert arr.shape == (EXPECTED_N_NODES, 2)

    def test_dtype(self) -> None:
        import numpy as np

        arr = get_coords_array()
        assert arr.dtype == np.float64

    def test_column_order_lat_lon(self) -> None:
        """First column should be latitude (22.xx), second longitude (113.xx)."""
        arr = get_coords_array()
        assert arr[:, 0].mean() == pytest.approx(22.19, abs=0.05), (
            "First column should be latitude (~22.19)"
        )
        assert arr[:, 1].mean() == pytest.approx(113.535, abs=0.02), (
            "Second column should be longitude (~113.535)"
        )
