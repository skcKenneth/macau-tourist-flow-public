"""Tests for src/utils/graph_loader.py.

Network-dependent tests are marked @pytest.mark.slow and skipped in the
default test run. Run them with: pytest -m slow
"""

from __future__ import annotations

import pytest


class TestLoadMacauGraphUnit:
    """Unit tests that do not require network access."""

    def test_default_bbox_values_are_sane(self) -> None:
        from src.utils.graph_loader import MACAU_HISTORIC_BBOX

        bbox = MACAU_HISTORIC_BBOX
        assert bbox["north"] > bbox["south"], "north must be > south"
        assert bbox["east"] > bbox["west"], "east must be > west"
        # Macau is roughly 22.18–22.22 N, 113.52–113.56 E
        assert 22.15 < bbox["south"] < 22.20
        assert 22.20 < bbox["north"] < 22.25
        assert 113.50 < bbox["west"] < 113.53
        assert 113.54 < bbox["east"] < 113.58

    def test_graphml_filename_constant(self) -> None:
        from src.utils.graph_loader import _GRAPHML_FILENAME

        assert _GRAPHML_FILENAME.endswith(".graphml")

    def test_walk_speed_is_positive(self) -> None:
        from src.utils.graph_loader import DEFAULT_WALK_SPEED_KMH

        assert DEFAULT_WALK_SPEED_KMH > 0


class TestComputeShortestPathsUnit:
    """Unit tests for compute_shortest_paths using a synthetic graph."""

    def _make_triangle_graph(self) -> tuple:
        """Create a simple 3-node undirected graph with known distances."""
        import networkx as nx

        G = nx.MultiDiGraph()
        # Three nodes: A, B, C
        G.add_node(1, y=0.0, x=0.0)
        G.add_node(2, y=0.0, x=1.0)
        G.add_node(3, y=1.0, x=0.0)
        # Edges (bidirectional)
        G.add_edge(1, 2, length=100.0)
        G.add_edge(2, 1, length=100.0)
        G.add_edge(2, 3, length=200.0)
        G.add_edge(3, 2, length=200.0)
        G.add_edge(1, 3, length=150.0)
        G.add_edge(3, 1, length=150.0)
        snap_map = {"node_a": 1, "node_b": 2, "node_c": 3}
        return G, snap_map

    def test_pairwise_distances_computed(self) -> None:
        from src.utils.graph_loader import compute_shortest_paths

        G, snap_map = self._make_triangle_graph()
        dists = compute_shortest_paths(G, snap_map, weight="length")
        # 3 nodes → 3 pairs × 2 directions = 6 entries
        assert len(dists) == 6

    def test_symmetric_distances(self) -> None:
        from src.utils.graph_loader import compute_shortest_paths

        G, snap_map = self._make_triangle_graph()
        dists = compute_shortest_paths(G, snap_map, weight="length")
        for src, dst in list(dists.keys()):
            assert dists[(src, dst)] == pytest.approx(dists[(dst, src)])

    def test_direct_edge_distance(self) -> None:
        from src.utils.graph_loader import compute_shortest_paths

        G, snap_map = self._make_triangle_graph()
        dists = compute_shortest_paths(G, snap_map, weight="length")
        assert dists[("node_a", "node_b")] == pytest.approx(100.0)
        assert dists[("node_a", "node_c")] == pytest.approx(150.0)

    def test_shortest_path_used(self) -> None:
        """node_b to node_c: direct=200, via node_a=250 → should be 200."""
        from src.utils.graph_loader import compute_shortest_paths

        G, snap_map = self._make_triangle_graph()
        dists = compute_shortest_paths(G, snap_map, weight="length")
        assert dists[("node_b", "node_c")] == pytest.approx(200.0)


@pytest.mark.slow
class TestLoadMacauGraphIntegration:
    """Integration tests requiring OSM network access.

    Run with: pytest -m slow
    These tests download real OSM data and may take 10–30 seconds.
    """

    def test_graph_is_nonempty(self, tmp_path) -> None:
        from src.utils.graph_loader import load_macau_graph

        G = load_macau_graph(cache_path=tmp_path / "osm")
        assert G.number_of_nodes() > 100
        assert G.number_of_edges() > 200

    def test_graph_has_length_attribute(self, tmp_path) -> None:
        from src.utils.graph_loader import load_macau_graph

        G = load_macau_graph(cache_path=tmp_path / "osm")
        # Spot-check: every edge should have a length attribute
        for _, _, data in list(G.edges(data=True))[:10]:
            assert "length" in data, "Edge missing 'length' attribute"

    def test_graph_has_travel_time_attribute(self, tmp_path) -> None:
        from src.utils.graph_loader import load_macau_graph

        G = load_macau_graph(cache_path=tmp_path / "osm")
        for _, _, data in list(G.edges(data=True))[:10]:
            assert "travel_time" in data, "Edge missing 'travel_time' attribute"

    def test_cache_is_written_and_reloaded(self, tmp_path) -> None:
        from src.utils.graph_loader import load_macau_graph

        cache_dir = tmp_path / "osm"
        G1 = load_macau_graph(cache_path=cache_dir)
        # Cache file should now exist
        assert (cache_dir / "macau_walk.graphml").exists()
        # Second load should use cache (and be faster)
        G2 = load_macau_graph(cache_path=cache_dir)
        assert G1.number_of_nodes() == G2.number_of_nodes()

    def test_all_attractions_snap_successfully(self, tmp_path) -> None:
        from src.utils.attractions import snap_to_graph
        from src.utils.graph_loader import load_macau_graph

        G = load_macau_graph(cache_path=tmp_path / "osm")
        snap_map = snap_to_graph(G)
        from src.utils.attractions import ATTRACTION_NODES

        assert len(snap_map) == len(ATTRACTION_NODES), (
            "Not all attraction nodes were snapped successfully"
        )
