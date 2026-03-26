"""Graph construction for volatility surface topology.

This module builds a static graph representing the structure of the options
volatility surface. Nodes are (tenor, delta_bucket) pairs, and edges connect:
- Adjacent delta buckets within the same tenor (delta adjacency)
- Adjacent tenors within the same delta bucket (tenor adjacency)

No diagonal edges are included to maintain locality in both dimensions.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch_geometric.data import Data

from ..config.constants import SurfaceConstants

# Default surface structure (from centralized constants)
DEFAULT_DELTA_BUCKETS = list(SurfaceConstants.DELTA_BUCKETS_GRAPH)
DEFAULT_TENORS_DAYS = list(SurfaceConstants.TENOR_DAYS_DEFAULT)


@dataclass
class SurfaceGraphConfig:
    """Configuration for surface graph construction.

    Attributes:
        delta_buckets: Ordered list of delta bucket names (put-side to call-side)
        tenors_days: Ordered list of tenor bins in days (short to long)
        include_edge_attr: Whether to include edge attributes
    """

    delta_buckets: list[str] = field(default_factory=lambda: DEFAULT_DELTA_BUCKETS.copy())
    tenors_days: list[int] = field(default_factory=lambda: DEFAULT_TENORS_DAYS.copy())
    include_edge_attr: bool = True


def build_surface_graph(config: Optional[SurfaceGraphConfig] = None) -> Data:
    """Build static graph representing volatility surface topology.

    Creates a graph where each node represents a (tenor, delta_bucket) pair
    on the volatility surface. Edges connect:
    - Adjacent delta buckets within each tenor (horizontal adjacency)
    - Adjacent tenors within each delta bucket (vertical adjacency)

    This creates a grid-like structure without diagonal connections.

    Args:
        config: Graph configuration. Uses defaults if None.

    Returns:
        torch_geometric.data.Data with:
            - edge_index: (2, num_edges) tensor of edge connections
            - edge_attr: (num_edges, 2) tensor of [delta_distance, tenor_distance]
              (only if include_edge_attr=True)
            - num_nodes: Total number of nodes
            - node_mapping: Dict mapping (tenor, bucket) -> node_index
            - index_mapping: Dict mapping node_index -> (tenor, bucket)

    Example:
        >>> config = SurfaceGraphConfig(
        ...     delta_buckets=["P25", "ATM", "C25"],
        ...     tenors_days=[30, 60, 90],
        ... )
        >>> graph = build_surface_graph(config)
        >>> graph.num_nodes
        9
        >>> graph.edge_index.shape
        torch.Size([2, 24])  # 6 delta edges + 6 tenor edges, bidirectional
    """
    if config is None:
        config = SurfaceGraphConfig()

    buckets = config.delta_buckets
    tenors = config.tenors_days
    num_buckets = len(buckets)
    num_tenors = len(tenors)

    # Create node mapping: (tenor, bucket) -> index
    # Nodes are ordered by tenor first, then bucket
    # Index: tenor_idx * num_buckets + bucket_idx
    node_mapping: dict[tuple[int, str], int] = {}
    index_mapping: dict[int, tuple[int, str]] = {}

    for t_idx, tenor in enumerate(tenors):
        for b_idx, bucket in enumerate(buckets):
            node_idx = t_idx * num_buckets + b_idx
            node_mapping[(tenor, bucket)] = node_idx
            index_mapping[node_idx] = (tenor, bucket)

    num_nodes = num_tenors * num_buckets

    # Build edge list
    edge_list: list[tuple[int, int]] = []
    edge_attrs: list[list[float]] = []

    # Delta adjacency: connect adjacent buckets within each tenor
    for t_idx, tenor in enumerate(tenors):
        for b_idx in range(num_buckets - 1):
            src_bucket = buckets[b_idx]
            dst_bucket = buckets[b_idx + 1]

            src = node_mapping[(tenor, src_bucket)]
            dst = node_mapping[(tenor, dst_bucket)]

            # Bidirectional edges
            edge_list.append((src, dst))
            edge_list.append((dst, src))

            # Edge attributes: [delta_distance=1, tenor_distance=0]
            edge_attrs.append([1.0, 0.0])
            edge_attrs.append([1.0, 0.0])

    # Tenor adjacency: connect adjacent tenors within each bucket
    for b_idx, bucket in enumerate(buckets):
        for t_idx in range(num_tenors - 1):
            src_tenor = tenors[t_idx]
            dst_tenor = tenors[t_idx + 1]

            src = node_mapping[(src_tenor, bucket)]
            dst = node_mapping[(dst_tenor, bucket)]

            # Bidirectional edges
            edge_list.append((src, dst))
            edge_list.append((dst, src))

            # Edge attributes: [delta_distance=0, tenor_distance=normalized]
            # Normalize tenor distance by dividing by first tenor gap
            tenor_dist = float(dst_tenor - src_tenor) / float(tenors[1] - tenors[0])
            edge_attrs.append([0.0, tenor_dist])
            edge_attrs.append([0.0, tenor_dist])

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if config.include_edge_attr else None
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32) if config.include_edge_attr else None

    # Create Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

    if config.include_edge_attr and edge_attr is not None:
        data.edge_attr = edge_attr

    # Store mappings as attributes for later use
    data.node_mapping = node_mapping
    data.index_mapping = index_mapping
    data.delta_buckets = buckets
    data.tenors_days = tenors

    return data


def get_node_index(
    graph: Data,
    tenor: int,
    delta_bucket: str,
) -> int:
    """Get node index for a (tenor, delta_bucket) pair.

    Args:
        graph: Graph data object with node_mapping attribute
        tenor: Tenor in days
        delta_bucket: Delta bucket name

    Returns:
        Node index in the graph

    Raises:
        KeyError: If (tenor, delta_bucket) not in graph
    """
    return graph.node_mapping[(tenor, delta_bucket)]


def get_node_position(graph: Data, node_idx: int) -> tuple[int, str]:
    """Get (tenor, delta_bucket) for a node index.

    Args:
        graph: Graph data object with index_mapping attribute
        node_idx: Node index in the graph

    Returns:
        Tuple of (tenor_days, delta_bucket)

    Raises:
        KeyError: If node_idx not in graph
    """
    return graph.index_mapping[node_idx]


def get_adjacency_matrix(graph: Data) -> torch.Tensor:
    """Convert edge_index to dense adjacency matrix.

    Args:
        graph: Graph data object

    Returns:
        Dense adjacency matrix of shape (num_nodes, num_nodes)
    """
    num_nodes = graph.num_nodes
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    if graph.edge_index.shape[1] > 0:
        src = graph.edge_index[0]
        dst = graph.edge_index[1]
        adj[src, dst] = 1.0

    return adj
