"""OSM walking-network extraction for the Macau historic centre.

Provides functions to:
1. Download / load-from-cache the pedestrian walking graph.
2. Annotate snapped attraction nodes with metadata.
3. Compute pairwise shortest-path distances between attraction nodes.

All graph operations are performed on the **post-consolidation** graph
(i.e. after ``osmnx.consolidation.consolidate_intersections``), which
renumbers OSM node IDs. Always snap attractions *after* consolidation.

Usage::

    from pathlib import Path
    from src.utils import graph_loader

    G = graph_loader.load_macau_graph(cache_path=Path("data/raw/osm"))
    snap_map = graph_loader.snap_attractions(G)
    G = graph_loader.add_attraction_attributes(G, snap_map)
    dist_matrix = graph_loader.compute_shortest_paths(G, snap_map)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default bounding box: covers full UNESCO-listed historic centre of Macau
# plus Outer Harbour Ferry Terminal and Border Gate transit nodes.
# Verified visually against the UNESCO boundary map (May 2026).
# EPSG:4326 (WGS84) decimal degrees.
# ---------------------------------------------------------------------------
MACAU_HISTORIC_BBOX: dict[str, float] = {
    "north": 22.220,
    "south": 22.183,
    "east": 113.550,
    "west": 113.525,
}

# Default walking speed for travel_time edge attribute (km/h).
# 5 km/h is the standard pedestrian planning assumption (AASHTO 2004).
DEFAULT_WALK_SPEED_KMH: float = 5.0

# Highway type → speed mapping (km/h) for osmnx travel time calculation.
_WALK_HWY_SPEEDS: dict[str, float] = {
    "footway": DEFAULT_WALK_SPEED_KMH,
    "path": DEFAULT_WALK_SPEED_KMH,
    "pedestrian": DEFAULT_WALK_SPEED_KMH,
    "living_street": DEFAULT_WALK_SPEED_KMH,
    "residential": DEFAULT_WALK_SPEED_KMH,
    "unclassified": DEFAULT_WALK_SPEED_KMH,
    "service": DEFAULT_WALK_SPEED_KMH,
    "steps": 2.0,  # stairs are slower
}

# GraphML cache filename
_GRAPHML_FILENAME = "macau_walk.graphml"


def load_macau_graph(
    bbox: dict[str, float] | None = None,
    network_type: str = "walk",
    simplify: bool = True,
    retain_all: bool = False,
    consolidate_tolerance_m: float = 15.0,
    walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH,
    cache_path: Path | str | None = None,
) -> nx.MultiDiGraph:
    """Extract the Macau historic-centre walking graph from OpenStreetMap.

    On first call, downloads from Overpass API and saves a GraphML cache.
    Subsequent calls load from cache (fast, no network required).

    Pipeline:
    1. ``osmnx.graph_from_bbox`` with ``network_type='walk'``
    2. Project to EPSG:32649 (UTM zone 49N — correct for Macau)
    3. ``consolidate_intersections(tolerance=consolidate_tolerance_m)`` to
       merge micro-nodes within plazas (e.g. Senado Square clutter)
    4. Extract the largest weakly-connected component
    5. Add ``travel_time`` edge attribute at the given walking speed

    Args:
        bbox: Dict with keys ``north``, ``south``, ``east``, ``west``
            in WGS84 decimal degrees. Defaults to ``MACAU_HISTORIC_BBOX``.
        network_type: osmnx network type. Keep ``"walk"`` for pedestrian paths.
        simplify: Collapse degree-2 nodes to simplify geometry (recommended).
        retain_all: If True, keep disconnected subgraphs. Default False
            (keeps only the largest weakly-connected component).
        consolidate_tolerance_m: Merge OSM nodes within this radius (metres).
            15 m eliminates plaza micro-node clutter without losing topology.
        walk_speed_kmh: Walking speed for travel_time calculation (km/h).
        cache_path: Directory to save/load the GraphML file. If ``None``,
            no caching is performed (re-downloads every time).

    Returns:
        MultiDiGraph in WGS84 projection with node attrs:
        ``osmid``, ``y`` (lat), ``x`` (lon), ``street_count``;
        and edge attrs: ``osmid``, ``length`` (metres),
        ``travel_time`` (seconds), ``name``, ``highway``.

    Raises:
        RuntimeError: If the graph is empty after extraction (indicates
            bbox or network_type issue).
    """
    import osmnx as ox

    if bbox is None:
        bbox = MACAU_HISTORIC_BBOX

    # Enable osmnx's built-in HTTP cache to avoid hammering Overpass/Nominatim
    ox.settings.use_cache = True
    if cache_path is not None:
        ox.settings.cache_folder = str(Path(cache_path).parent / "osmnx_http_cache")

    # Try loading from GraphML cache first
    graphml_path: Path | None = None
    if cache_path is not None:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        graphml_path = cache_dir / _GRAPHML_FILENAME

        if graphml_path.exists():
            logger.info("Loading graph from cache: %s", graphml_path)
            G = ox.load_graphml(graphml_path)
            return G

    # ── Download from OSM ────────────────────────────────────────────────────
    logger.info(
        "Downloading OSM walk graph for bbox N=%.4f S=%.4f E=%.4f W=%.4f …",
        bbox["north"],
        bbox["south"],
        bbox["east"],
        bbox["west"],
    )
    # osmnx ≥2.0: graph_from_bbox takes (left, bottom, right, top) = (W, S, E, N)
    G = ox.graph_from_bbox(
        (bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
    )

    if len(G.nodes) == 0:
        raise RuntimeError(
            "OSM graph is empty. Check bbox values and network_type. "
            f"bbox={bbox}, network_type={network_type!r}"
        )

    logger.info(
        "Raw graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )

    # ── Project to UTM 49N for metric consolidation ──────────────────────────
    G_proj = ox.project_graph(G, to_crs="EPSG:32649")

    # Consolidate intersections: merge nodes within tolerance_m radius.
    # This eliminates the OSM micro-node clutter typical of plazas.
    # NOTE: consolidation renumbers node IDs — snap attractions after this step.
    # osmnx ≥2.0: consolidate_intersections is a top-level function
    G_proj = ox.consolidate_intersections(
        G_proj,
        tolerance=consolidate_tolerance_m,
        rebuild_graph=True,
        dead_ends=False,
        reconnect_edges=True,
    )
    logger.info(
        "After consolidation: %d nodes, %d edges",
        G_proj.number_of_nodes(),
        G_proj.number_of_edges(),
    )

    # ── Extract largest weakly-connected component ───────────────────────────
    # osmnx ≥2.0 removed utils_graph; use networkx directly
    if not retain_all:
        largest_wcc = max(nx.weakly_connected_components(G_proj), key=len)
        G_proj = G_proj.subgraph(largest_wcc).copy()
        logger.info(
            "Largest component: %d nodes, %d edges",
            G_proj.number_of_nodes(),
            G_proj.number_of_edges(),
        )

    # ── Re-project back to WGS84 for consistent lat/lon node attributes ──────
    G = ox.project_graph(G_proj, to_latlong=True)

    # ── Add travel_time edge attribute ───────────────────────────────────────
    # osmnx ≥2.0: must call add_edge_speeds first, then add_edge_travel_times
    hwy_speeds = {k: v for k, v in _WALK_HWY_SPEEDS.items()}
    for k in hwy_speeds:
        if k not in ("steps",):
            hwy_speeds[k] = walk_speed_kmh
    G = ox.add_edge_speeds(G, hwy_speeds=hwy_speeds)
    G = ox.add_edge_travel_times(G)

    # ── Save GraphML cache ───────────────────────────────────────────────────
    if graphml_path is not None:
        ox.save_graphml(G, filepath=graphml_path)
        logger.info("Graph cached to: %s", graphml_path)

    return G


def add_attraction_attributes(
    G: nx.MultiDiGraph,
    snap_map: dict[str, int],
) -> nx.MultiDiGraph:
    """Annotate the graph with attraction metadata on snapped nodes.

    Adds the following attributes to each snapped OSM node:
    ``attraction_id``, ``attraction_name_en``, ``attraction_name_zh``,
    ``node_type``, ``is_bottleneck``.

    Args:
        G: Walking graph (in-place modification).
        snap_map: Output of ``attractions.snap_to_graph(G)`` mapping
            attraction_id → osm_node_id.

    Returns:
        The same graph G (modified in-place) for method chaining.
    """
    from src.utils.attractions import NODE_BY_ID

    for attraction_id, osm_id in snap_map.items():
        if osm_id not in G.nodes:
            logger.warning(
                "snap_map node %d for %s not found in graph; skipping.",
                osm_id,
                attraction_id,
            )
            continue
        node = NODE_BY_ID[attraction_id]
        G.nodes[osm_id]["attraction_id"] = node.node_id
        G.nodes[osm_id]["attraction_name_en"] = node.name_en
        G.nodes[osm_id]["attraction_name_zh"] = node.name_zh
        G.nodes[osm_id]["node_type"] = node.node_type.value
        G.nodes[osm_id]["is_bottleneck"] = node.is_bottleneck

    return G


def compute_shortest_paths(
    G: nx.MultiDiGraph,
    snap_map: dict[str, int],
    weight: str = "length",
) -> dict[tuple[str, str], float]:
    """Compute pairwise shortest-path distances between all attraction nodes.

    Uses Dijkstra's algorithm via NetworkX. Treats the MultiDiGraph as
    undirected for distance computation (pedestrians can walk both ways).

    Args:
        G: Walking graph with edge weight attribute ``weight``.
        snap_map: Mapping from attraction_id (str) to OSM node ID (int).
        weight: Edge attribute to use as distance. Use ``"length"`` for
            metres or ``"travel_time"`` for seconds.

    Returns:
        Dict mapping ``(src_attraction_id, dst_attraction_id)`` to distance.
        All pairs are included (including reversed direction). Self-distances
        are omitted.

    Raises:
        nx.NetworkXNoPath: If two attraction nodes are not connected in G.
            This indicates a graph coverage problem — check bbox or connectivity.
    """
    G_undirected = G.to_undirected()
    dist_matrix: dict[tuple[str, str], float] = {}

    attraction_ids = list(snap_map.keys())
    for i, src_id in enumerate(attraction_ids):
        for dst_id in attraction_ids[i + 1 :]:
            src_osm = snap_map[src_id]
            dst_osm = snap_map[dst_id]
            try:
                d = nx.shortest_path_length(
                    G_undirected, source=src_osm, target=dst_osm, weight=weight
                )
                dist_matrix[(src_id, dst_id)] = d
                dist_matrix[(dst_id, src_id)] = d
            except nx.NetworkXNoPath:
                logger.error(
                    "No path between %s (osm %d) and %s (osm %d). "
                    "Check graph connectivity.",
                    src_id,
                    src_osm,
                    dst_id,
                    dst_osm,
                )
                raise

    return dist_matrix
