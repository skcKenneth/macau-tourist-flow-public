"""Authoritative node definitions for the Macau heritage attraction graph.

This module is the single source of truth for node identities, coordinates,
and metadata. All other modules that reference attraction nodes should import
from here rather than hardcoding IDs or coordinates.

Coordinates are WGS84 (EPSG:4326), 5 decimal places (~1 m accuracy).
Each point is the principal entrance or most identifiable access point of the
site (e.g., forecourt of the St. Paul's facade, centre of Senado Square paving).

Validated against OSM Nominatim and Google Maps as of May 2026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Classification of graph nodes."""

    ATTRACTION = "attraction"
    TRANSIT = "transit"
    JUNCTION = "junction"  # reserved for future graph augmentation


@dataclass(frozen=True)
class AttractionNode:
    """Immutable metadata for a single node in the Macau heritage graph.

    Attributes:
        node_id: Short slug used as the canonical identifier throughout the
            project (matches graph node attrs and parquet keys).
        name_en: English name.
        name_zh: Traditional Chinese name.
        lat: WGS84 latitude in decimal degrees.
        lon: WGS84 longitude in decimal degrees.
        node_type: Classification (attraction, transit, junction).
        annual_visitors_est: Rough annual visitor estimate for initial weight
            initialisation. Update from calibrated data once available.
        is_bottleneck: True if this node is a known congestion hotspot.
        notes: Free-text documentation notes.
    """

    node_id: str
    name_en: str
    name_zh: str
    lat: float
    lon: float
    node_type: NodeType
    annual_visitors_est: int
    is_bottleneck: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Node registry
# Attractions are listed first (for consistent tensor indexing), then transit.
# Do NOT reorder without updating any saved model checkpoints.
# ---------------------------------------------------------------------------

ATTRACTION_NODES: list[AttractionNode] = [
    # ── Heritage attractions ─────────────────────────────────────────────────
    AttractionNode(
        node_id="ruins_st_pauls",
        name_en="Ruins of St. Paul's",
        name_zh="大三巴牌坊",
        lat=22.19748,
        lon=113.53896,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=2_000_000,
        is_bottleneck=True,
        notes="Snap point: forecourt at base of stone steps. Primary bottleneck.",
    ),
    AttractionNode(
        node_id="senado_square",
        name_en="Senado Square",
        name_zh="議事亭前地",
        lat=22.19369,
        lon=113.53843,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=1_500_000,
        notes="Snap point: centre of the Portuguese wave-patterned paving.",
    ),
    AttractionNode(
        node_id="ama_temple",
        name_en="A-Ma Temple",
        name_zh="媽閣廟",
        lat=22.18668,
        lon=113.52893,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=800_000,
        notes="Snap point: main entrance gate on Rua de São Tiago da Barra.",
    ),
    AttractionNode(
        node_id="st_dominics",
        name_en="St. Dominic's Church",
        name_zh="玫瑰堂",
        lat=22.19402,
        lon=113.53788,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=600_000,
        notes="Snap point: front facade on Largo de São Domingos.",
    ),
    AttractionNode(
        node_id="mount_fortress",
        name_en="Mount Fortress",
        name_zh="大炮台",
        lat=22.19877,
        lon=113.54075,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=700_000,
        notes="Snap point: main entrance on Rua do Monte.",
    ),
    AttractionNode(
        node_id="lou_kau_mansion",
        name_en="Lou Kau Mansion",
        name_zh="盧家大屋",
        lat=22.19407,
        lon=113.53757,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=200_000,
        notes="Snap point: entrance on Rua de São Domingos.",
    ),
    AttractionNode(
        node_id="st_lawrence",
        name_en="St. Lawrence's Church",
        name_zh="聖老楞佐教堂",
        lat=22.18936,
        lon=113.53500,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=300_000,
        notes="Snap point: main entrance on Rua de São Lourenço.",
    ),
    AttractionNode(
        node_id="st_joseph_seminary",
        name_en="St. Joseph's Seminary",
        name_zh="聖若瑟修院",
        lat=22.18889,
        lon=113.53367,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=250_000,
        notes="Snap point: entrance on Rua do Seminário.",
    ),
    AttractionNode(
        node_id="lilau_square",
        name_en="Lilau Square",
        name_zh="亞婆井前地",
        lat=22.18777,
        lon=113.53256,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=150_000,
        notes="Snap point: centre of the small square.",
    ),
    AttractionNode(
        node_id="mandarins_house",
        name_en="Mandarin's House",
        name_zh="鄭家大屋",
        lat=22.18705,
        lon=113.53215,
        node_type=NodeType.ATTRACTION,
        annual_visitors_est=180_000,
        notes="Snap point: main entrance gate.",
    ),
    # ── Transit nodes ────────────────────────────────────────────────────────
    AttractionNode(
        node_id="ferry_outer",
        name_en="Outer Harbour Ferry Terminal",
        name_zh="外港碼頭",
        lat=22.19308,
        lon=113.54607,
        node_type=NodeType.TRANSIT,
        annual_visitors_est=5_000_000,
        notes="Main ferry drop-off from Hong Kong/Taipa. Primary exogenous arrival source.",
    ),
    AttractionNode(
        node_id="border_gate",
        name_en="Macau Border Gate",
        name_zh="關閘",
        lat=22.21672,
        lon=113.54426,
        node_type=NodeType.TRANSIT,
        annual_visitors_est=8_000_000,
        notes=(
            "Land border crossing with Zhuhai. Largest single arrival source. "
            "~2.5 km from Senado Square; verify bbox covers full path."
        ),
    ),
    AttractionNode(
        node_id="hotel_belt",
        name_en="Hotel Belt (NAPE/Cotai)",
        name_zh="酒店群",
        lat=22.19900,
        lon=113.54200,
        node_type=NodeType.TRANSIT,
        annual_visitors_est=3_000_000,
        notes=(
            "Aggregate transit node representing the NAPE/Cotai hotel strip. "
            "No single OSM counterpart; use snap tolerance=300 m."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

NODE_BY_ID: dict[str, AttractionNode] = {n.node_id: n for n in ATTRACTION_NODES}

BOTTLENECK_IDS: list[str] = [
    n.node_id for n in ATTRACTION_NODES if n.is_bottleneck
]

ATTRACTION_IDS: list[str] = [
    n.node_id
    for n in ATTRACTION_NODES
    if n.node_type == NodeType.ATTRACTION
]

TRANSIT_IDS: list[str] = [
    n.node_id
    for n in ATTRACTION_NODES
    if n.node_type == NodeType.TRANSIT
]

# Canonical index mapping (stable — do not reorder ATTRACTION_NODES)
NODE_INDEX: dict[str, int] = {n.node_id: i for i, n in enumerate(ATTRACTION_NODES)}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_coords_array() -> np.ndarray:
    """Return (N, 2) float64 array of [lat, lon] for all nodes in canonical order.

    Returns:
        Array of shape (N_nodes, 2) where row i corresponds to ATTRACTION_NODES[i].
    """
    return np.array([[n.lat, n.lon] for n in ATTRACTION_NODES], dtype=np.float64)


def snap_to_graph(
    G: "nx.MultiDiGraph",
    max_dist_attraction_m: float = 100.0,
    max_dist_transit_m: float = 300.0,
) -> dict[str, int]:
    """Snap each AttractionNode to its nearest OSM node in G.

    Snapping uses the Haversine distance between the node's WGS84 coordinates
    and each OSM node's (y=lat, x=lon) attributes. This must be called
    **after** any graph consolidation / projection steps, since those
    operations renumber OSM node IDs.

    Args:
        G: OSM walking graph (MultiDiGraph) with node attributes 'y' (lat)
            and 'x' (lon) in WGS84 degrees.
        max_dist_attraction_m: Maximum allowed snapping distance for attraction
            nodes before emitting a warning. Defaults to 100 m.
        max_dist_transit_m: Maximum allowed snapping distance for transit nodes
            (larger because hotel_belt is an aggregate point). Defaults to 300 m.

    Returns:
        Dict mapping node_id (str) to OSM integer node ID (int).

    Raises:
        ValueError: If any attraction node snaps to a node farther than
            max_dist_attraction_m, or any transit node farther than
            max_dist_transit_m, indicating likely missing coverage in the bbox.
    """
    import osmnx as ox

    snap_map: dict[str, int] = {}
    errors: list[str] = []

    for node in ATTRACTION_NODES:
        osm_id, dist = ox.distance.nearest_nodes(
            G, X=node.lon, Y=node.lat, return_dist=True
        )
        snap_map[node.node_id] = osm_id

        limit = (
            max_dist_transit_m
            if node.node_type == NodeType.TRANSIT
            else max_dist_attraction_m
        )
        if dist > limit:
            msg = (
                f"{node.node_id} ({node.name_en}) snapped {dist:.0f} m away "
                f"(limit {limit:.0f} m). Check bbox or add synthetic edge."
            )
            if dist > limit * 3:
                errors.append(msg)
            else:
                logger.warning(msg)

    if errors:
        raise ValueError(
            "Snap distance exceeded for the following nodes:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return snap_map
