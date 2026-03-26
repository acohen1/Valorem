"""Unit tests for graph construction module."""

import pytest
import torch

from src.models.graph import (
    SurfaceGraphConfig,
    build_surface_graph,
    get_node_index,
    get_node_position,
    get_adjacency_matrix,
    DEFAULT_DELTA_BUCKETS,
    DEFAULT_TENORS_DAYS,
)


class TestSurfaceGraphConfig:
    """Tests for SurfaceGraphConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SurfaceGraphConfig()

        assert config.delta_buckets == DEFAULT_DELTA_BUCKETS
        assert config.tenors_days == DEFAULT_TENORS_DAYS
        assert config.include_edge_attr is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM", "C25"],
            tenors_days=[30, 60, 90],
            include_edge_attr=False,
        )

        assert config.delta_buckets == ["P25", "ATM", "C25"]
        assert config.tenors_days == [30, 60, 90]
        assert config.include_edge_attr is False


class TestBuildSurfaceGraph:
    """Tests for build_surface_graph function."""

    def test_default_graph_structure(self):
        """Test graph built with default config."""
        graph = build_surface_graph()

        # Default: 7 buckets × 6 tenors = 42 nodes
        assert graph.num_nodes == 42
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "edge_attr")
        assert hasattr(graph, "node_mapping")
        assert hasattr(graph, "index_mapping")

    def test_node_count(self):
        """Test node count matches buckets × tenors."""
        config = SurfaceGraphConfig(
            delta_buckets=["P10", "ATM", "C10"],
            tenors_days=[7, 30, 90],
        )
        graph = build_surface_graph(config)

        # 3 buckets × 3 tenors = 9 nodes
        assert graph.num_nodes == 9

    def test_edge_count_delta_adjacency(self):
        """Test delta adjacency edges (within tenor)."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM", "C25"],  # 3 buckets
            tenors_days=[30],  # 1 tenor
        )
        graph = build_surface_graph(config)

        # 3 buckets, 1 tenor: 2 delta edges (P25-ATM, ATM-C25), bidirectional = 4
        # No tenor edges (only 1 tenor)
        assert graph.edge_index.shape[1] == 4

    def test_edge_count_tenor_adjacency(self):
        """Test tenor adjacency edges (within bucket)."""
        config = SurfaceGraphConfig(
            delta_buckets=["ATM"],  # 1 bucket
            tenors_days=[30, 60, 90],  # 3 tenors
        )
        graph = build_surface_graph(config)

        # 1 bucket, 3 tenors: 2 tenor edges (30-60, 60-90), bidirectional = 4
        # No delta edges (only 1 bucket)
        assert graph.edge_index.shape[1] == 4

    def test_edge_count_full_grid(self):
        """Test edge count for full grid (no diagonal)."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM", "C25"],  # 3 buckets
            tenors_days=[30, 60, 90],  # 3 tenors
        )
        graph = build_surface_graph(config)

        # Delta edges: 3 tenors × 2 edges per tenor × 2 (bidirectional) = 12
        # Tenor edges: 3 buckets × 2 edges per bucket × 2 (bidirectional) = 12
        # Total: 24
        assert graph.edge_index.shape[1] == 24

    def test_no_diagonal_edges(self):
        """Test that no diagonal edges exist."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM", "C25"],
            tenors_days=[30, 60],
        )
        graph = build_surface_graph(config)

        # Build adjacency matrix
        adj = get_adjacency_matrix(graph)

        # Node layout (tenor × bucket):
        # (30, P25)=0, (30, ATM)=1, (30, C25)=2
        # (60, P25)=3, (60, ATM)=4, (60, C25)=5

        # Check no diagonal edges (e.g., 0-4, 1-3, 1-5, 2-4)
        assert adj[0, 4] == 0  # (30,P25) to (60,ATM)
        assert adj[4, 0] == 0
        assert adj[1, 3] == 0  # (30,ATM) to (60,P25)
        assert adj[3, 1] == 0
        assert adj[1, 5] == 0  # (30,ATM) to (60,C25)
        assert adj[5, 1] == 0
        assert adj[2, 4] == 0  # (30,C25) to (60,ATM)
        assert adj[4, 2] == 0

    def test_edge_attributes(self):
        """Test edge attributes for delta and tenor edges."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM"],
            tenors_days=[30, 60],
            include_edge_attr=True,
        )
        graph = build_surface_graph(config)

        assert graph.edge_attr is not None
        assert graph.edge_attr.shape[1] == 2  # [delta_dist, tenor_dist]

        # Check that edge attrs are correctly labeled
        for i in range(graph.edge_index.shape[1]):
            delta_dist, tenor_dist = graph.edge_attr[i]
            # Either delta or tenor edge, not both
            assert (delta_dist > 0 and tenor_dist == 0) or (delta_dist == 0 and tenor_dist > 0)

    def test_no_edge_attributes(self):
        """Test graph without edge attributes."""
        config = SurfaceGraphConfig(include_edge_attr=False)
        graph = build_surface_graph(config)

        assert not hasattr(graph, "edge_attr") or graph.edge_attr is None

    def test_node_mapping(self):
        """Test node mapping is correct."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM"],
            tenors_days=[30, 60],
        )
        graph = build_surface_graph(config)

        # Check mapping
        assert graph.node_mapping[(30, "P25")] == 0
        assert graph.node_mapping[(30, "ATM")] == 1
        assert graph.node_mapping[(60, "P25")] == 2
        assert graph.node_mapping[(60, "ATM")] == 3

    def test_index_mapping(self):
        """Test index mapping is correct."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM"],
            tenors_days=[30, 60],
        )
        graph = build_surface_graph(config)

        # Check reverse mapping
        assert graph.index_mapping[0] == (30, "P25")
        assert graph.index_mapping[1] == (30, "ATM")
        assert graph.index_mapping[2] == (60, "P25")
        assert graph.index_mapping[3] == (60, "ATM")

    def test_edge_index_dtype(self):
        """Test edge_index has correct dtype."""
        graph = build_surface_graph()

        assert graph.edge_index.dtype == torch.long

    def test_edge_attr_dtype(self):
        """Test edge_attr has correct dtype."""
        graph = build_surface_graph()

        assert graph.edge_attr.dtype == torch.float32

    def test_bidirectional_edges(self):
        """Test that all edges are bidirectional."""
        config = SurfaceGraphConfig(
            delta_buckets=["ATM", "C25"],
            tenors_days=[30, 60],
        )
        graph = build_surface_graph(config)

        # Check each edge has reverse edge
        edge_set = set()
        for i in range(graph.edge_index.shape[1]):
            src, dst = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            edge_set.add((src, dst))

        for src, dst in edge_set:
            assert (dst, src) in edge_set, f"Missing reverse edge for ({src}, {dst})"


class TestGraphHelperFunctions:
    """Tests for helper functions."""

    def test_get_node_index(self):
        """Test get_node_index function."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM", "C25"],
            tenors_days=[30, 60],
        )
        graph = build_surface_graph(config)

        assert get_node_index(graph, 30, "P25") == 0
        assert get_node_index(graph, 30, "ATM") == 1
        assert get_node_index(graph, 60, "C25") == 5

    def test_get_node_index_invalid(self):
        """Test get_node_index with invalid key."""
        graph = build_surface_graph()

        with pytest.raises(KeyError):
            get_node_index(graph, 999, "INVALID")

    def test_get_node_position(self):
        """Test get_node_position function."""
        config = SurfaceGraphConfig(
            delta_buckets=["P25", "ATM"],
            tenors_days=[30, 60],
        )
        graph = build_surface_graph(config)

        assert get_node_position(graph, 0) == (30, "P25")
        assert get_node_position(graph, 3) == (60, "ATM")

    def test_get_node_position_invalid(self):
        """Test get_node_position with invalid index."""
        graph = build_surface_graph()

        with pytest.raises(KeyError):
            get_node_position(graph, 999)

    def test_get_adjacency_matrix(self):
        """Test adjacency matrix construction."""
        config = SurfaceGraphConfig(
            delta_buckets=["ATM", "C25"],
            tenors_days=[30],
        )
        graph = build_surface_graph(config)

        adj = get_adjacency_matrix(graph)

        assert adj.shape == (2, 2)
        assert adj[0, 1] == 1.0  # ATM -> C25
        assert adj[1, 0] == 1.0  # C25 -> ATM
        assert adj[0, 0] == 0.0  # No self-loop
        assert adj[1, 1] == 0.0

    def test_adjacency_matrix_symmetric(self):
        """Test adjacency matrix is symmetric."""
        graph = build_surface_graph()
        adj = get_adjacency_matrix(graph)

        assert torch.allclose(adj, adj.T)


class TestGraphEdgeCases:
    """Tests for edge cases."""

    def test_single_node_graph(self):
        """Test graph with single node."""
        config = SurfaceGraphConfig(
            delta_buckets=["ATM"],
            tenors_days=[30],
        )
        graph = build_surface_graph(config)

        assert graph.num_nodes == 1
        assert graph.edge_index.shape[1] == 0  # No edges

    def test_single_bucket_multiple_tenors(self):
        """Test single bucket with multiple tenors."""
        config = SurfaceGraphConfig(
            delta_buckets=["ATM"],
            tenors_days=[7, 14, 30, 60],
        )
        graph = build_surface_graph(config)

        assert graph.num_nodes == 4
        # 3 tenor edges × 2 (bidirectional) = 6
        assert graph.edge_index.shape[1] == 6

    def test_single_tenor_multiple_buckets(self):
        """Test single tenor with multiple buckets."""
        config = SurfaceGraphConfig(
            delta_buckets=["P10", "P25", "ATM", "C25", "C10"],
            tenors_days=[30],
        )
        graph = build_surface_graph(config)

        assert graph.num_nodes == 5
        # 4 delta edges × 2 (bidirectional) = 8
        assert graph.edge_index.shape[1] == 8

    def test_graph_preserves_bucket_order(self):
        """Test that bucket order is preserved in graph attributes."""
        buckets = ["P40", "P25", "ATM", "C25", "C40"]
        config = SurfaceGraphConfig(delta_buckets=buckets)
        graph = build_surface_graph(config)

        assert graph.delta_buckets == buckets

    def test_graph_preserves_tenor_order(self):
        """Test that tenor order is preserved in graph attributes."""
        tenors = [14, 30, 60, 90, 120, 180]
        config = SurfaceGraphConfig(tenors_days=tenors)
        graph = build_surface_graph(config)

        assert graph.tenors_days == tenors


class TestDefaultGraphTopology:
    """Tests for default graph topology matching spec."""

    def test_default_42_nodes(self):
        """Test default graph has 42 nodes (7 buckets × 6 tenors)."""
        graph = build_surface_graph()
        assert graph.num_nodes == 42

    def test_default_edge_count(self):
        """Test default graph edge count.

        7 buckets × 6 delta edges per bucket row × 2 (bidir) = 72 delta edges
        6 tenors × 5 tenor edges per bucket col × 2 (bidir) = 60 tenor edges

        Wait, let me recalculate:
        - Delta edges: 6 tenors × (7-1) pairs × 2 = 6 × 6 × 2 = 72
        - Tenor edges: 7 buckets × (6-1) pairs × 2 = 7 × 5 × 2 = 70
        - Total: 142
        """
        graph = build_surface_graph()

        # 7 buckets, 6 tenors
        # Delta adjacency: 6 tenors × 6 pairs × 2 = 72
        # Tenor adjacency: 7 buckets × 5 pairs × 2 = 70
        # Total: 142
        assert graph.edge_index.shape[1] == 142

    def test_all_nodes_connected(self):
        """Test that graph is connected (no isolated nodes)."""
        graph = build_surface_graph()
        adj = get_adjacency_matrix(graph)

        # Each node should have at least one neighbor
        degree = adj.sum(dim=1)
        assert (degree > 0).all()
