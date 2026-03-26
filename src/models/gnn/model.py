"""Graph Neural Network model for cross-sectional surface learning.

This module implements the GNN component of the PatchTST+GNN ensemble.
The GNN propagates information across the volatility surface graph,
allowing nodes to share information based on their structural relationships
(delta neighbors, tenor neighbors).

The GNN supports two layer types:
- GAT (Graph Attention): Supports edge attributes for weighted message passing
- GCN (Graph Convolution): Standard spectral convolution without edge attributes
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


@dataclass
class GNNModelConfig:
    """Configuration for SurfaceGNN model.

    Attributes:
        model_type: Type of GNN layer ("GAT" or "GCN").
        hidden_dim: Hidden dimension for GNN layers.
        n_layers: Number of GNN layers.
        heads: Number of attention heads (GAT only).
        dropout: Dropout rate.
        use_edge_attr: Whether to use edge attributes (GAT only).
        use_learnable_edge_attr: Whether edge attributes are learnable parameters.
            When True, edge_attr must be stored in the ensemble model as nn.Parameter.
            When False, uses fixed edge attributes (backward compatible).
    """

    model_type: Literal["GAT", "GCN"] = "GAT"
    hidden_dim: int = 64
    n_layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    use_edge_attr: bool = True
    use_learnable_edge_attr: bool = False
    edge_dim: int = 2
    use_dynamic_volume_edges: bool = False


class SurfaceGNN(nn.Module):
    """Graph Neural Network over options volatility surface.

    Processes node features through multiple GNN layers, respecting the
    graph topology defined by edge_index. Supports both GAT (with edge
    attributes) and GCN layer types.

    Uses a batched super-graph approach: all samples in a batch are combined
    into a single disconnected graph by offsetting node indices per sample.
    This enables a single GNN forward pass over the entire batch instead of
    a sequential Python loop over samples.

    Architecture:
        Input: (batch, nodes, features) + edge_index (2, edges)
        -> Build batched super-graph: (batch*nodes, features) + (2, batch*edges)
        -> For each GNN layer:
            - Message passing over super-graph
            - ReLU activation (except last layer)
            - Dropout (training only, except last layer)
        -> Reshape: (batch, nodes, hidden_dim)

    Example:
        >>> config = GNNModelConfig(model_type="GAT", hidden_dim=64, n_layers=2)
        >>> model = SurfaceGNN(config, input_dim=128)
        >>> x = torch.randn(32, 42, 128)  # (batch, nodes, features)
        >>> edge_index = torch.randint(0, 42, (2, 100))
        >>> edge_attr = torch.randn(100, 2)
        >>> out = model(x, edge_index, edge_attr)  # (32, 42, 64)
    """

    def __init__(
        self, config: GNNModelConfig, input_dim: int, output_horizons: int = 0
    ):
        """Initialize SurfaceGNN.

        Args:
            config: Model configuration.
            input_dim: Input feature dimension (typically d_model from PatchTST).
            output_horizons: Number of output horizons for predictions. If 0,
                operates in encoder-only mode (returns embeddings). If > 0,
                adds a prediction head (returns predictions).
        """
        super().__init__()
        self._config = config
        self._input_dim = input_dim

        # Build GNN layers
        self._layers = nn.ModuleList()

        for i in range(config.n_layers):
            in_channels = input_dim if i == 0 else config.hidden_dim

            if config.model_type == "GAT":
                # GAT supports edge_dim for edge attributes
                edge_dim = config.edge_dim if config.use_edge_attr else None
                layer = GATConv(
                    in_channels=in_channels,
                    out_channels=config.hidden_dim,
                    heads=config.heads,
                    edge_dim=edge_dim,
                    dropout=config.dropout,
                    concat=False,  # Average heads instead of concatenating
                )
            else:  # GCN
                layer = GCNConv(
                    in_channels=in_channels,
                    out_channels=config.hidden_dim,
                )

            self._layers.append(layer)

        # Prediction head (only if output_horizons > 0)
        self._head: nn.Linear | None
        if output_horizons > 0:
            self._head = nn.Linear(config.hidden_dim, output_horizons)
        else:
            self._head = None
        self._output_horizons = output_horizons

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through GNN using batched super-graph.

        Constructs a single disconnected super-graph containing all samples
        in the batch. Each sample's node indices are offset so the shared
        edge_index is replicated per sample without cross-sample edges.
        This allows a single GNN forward pass over the entire batch.

        Args:
            x: Node features of shape (batch, nodes, features).
            edge_index: Graph connectivity of shape (2, num_edges).
            edge_attr: Edge attributes. Supports two shapes:
                - 2D (num_edges, D): Static attributes, replicated per sample.
                - 3D (batch, num_edges, D): Per-sample dynamic attributes.
                Only used with GAT.
            mask: Optional boolean mask of shape (batch, num_nodes) indicating
                valid nodes. Only used when output_horizons > 0.

        Returns:
            If output_horizons > 0: Predictions of shape (batch, nodes, output_horizons)
            If output_horizons = 0: Node embeddings of shape (batch, nodes, hidden_dim)
        """
        batch_size, num_nodes, feat_dim = x.shape

        # Build batched super-graph: offset edge_index per sample
        # Creates one large graph with batch_size disconnected subgraphs
        offsets = torch.arange(batch_size, device=x.device) * num_nodes  # (batch,)
        # edge_index: (2, E) -> expand to (2, batch, E) -> offset -> reshape (2, batch*E)
        batched_edge_index = (
            edge_index.unsqueeze(1) + offsets.unsqueeze(0).unsqueeze(-1)
        ).reshape(2, -1)

        # Replicate edge_attr for each sample in the batch
        batched_edge_attr: torch.Tensor | None = None
        if edge_attr is not None:
            if edge_attr.ndim == 2:
                # Static: (num_edges, D) -> repeat -> (batch*num_edges, D)
                batched_edge_attr = edge_attr.repeat(batch_size, 1)
            elif edge_attr.ndim == 3:
                # Dynamic per-sample: (batch, num_edges, D) -> (batch*num_edges, D)
                batched_edge_attr = edge_attr.reshape(-1, edge_attr.size(-1))

        # Flatten node features: (batch, nodes, features) -> (batch*nodes, features)
        node_features = x.reshape(batch_size * num_nodes, feat_dim)

        # Single forward pass through all GNN layers
        for layer_idx, layer in enumerate(self._layers):
            if self._config.model_type == "GAT" and self._config.use_edge_attr:
                node_features = layer(
                    node_features, batched_edge_index, edge_attr=batched_edge_attr
                )
            else:
                node_features = layer(node_features, batched_edge_index)

            # Apply activation (except on last layer)
            if layer_idx < len(self._layers) - 1:
                node_features = F.relu(node_features)
                node_features = F.dropout(
                    node_features, p=self._config.dropout, training=self.training
                )

        # Reshape back: (batch*nodes, hidden_dim) -> (batch, nodes, hidden_dim)
        embeddings = node_features.reshape(batch_size, num_nodes, -1)

        # If encoder-only mode, return embeddings
        if self._head is None:
            return embeddings

        # Apply prediction head
        preds: torch.Tensor = self._head(embeddings)

        # Apply mask if provided
        if mask is not None:
            preds = preds * mask.unsqueeze(-1).float()

        return preds

    @property
    def config(self) -> GNNModelConfig:
        """Return model configuration."""
        return self._config

    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self._input_dim

    @property
    def hidden_dim(self) -> int:
        """Return hidden dimension."""
        return self._config.hidden_dim

    @property
    def output_horizons(self) -> int:
        """Return number of output horizons."""
        return self._output_horizons
