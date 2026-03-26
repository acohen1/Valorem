"""PatchTST + GNN ensemble model for volatility surface prediction.

This module combines the temporal encoding capabilities of PatchTST with
the cross-sectional learning of GNN to predict volatility mispricing
at each node on the surface.

Architecture:
    X: (batch, time, nodes, features)
       ↓
    PatchTST.encode(): (batch, nodes, d_model)
       ↓
    SurfaceGNN: (batch, nodes, hidden_dim)
       ↓
    Linear head: (batch, nodes, horizons)
       ↓
    Mask + Output
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.gnn import GNNModelConfig, SurfaceGNN
from src.models.patchtst import PatchTSTModel, PatchTSTModelConfig


class PatchTST_GNN_Ensemble(nn.Module):
    """Combined PatchTST + GNN model for volatility surface prediction.

    This ensemble leverages the complementary strengths of both architectures:
    - PatchTST captures temporal dynamics for each surface node independently
    - GNN propagates information across the surface, capturing cross-sectional
      dependencies (delta neighbors share skew, tenor neighbors share term structure)

    Example:
        >>> patchtst_config = PatchTSTModelConfig(d_model=128)
        >>> gnn_config = GNNModelConfig(hidden_dim=64)
        >>> model = PatchTST_GNN_Ensemble(patchtst_config, gnn_config, input_dim=13, output_horizons=3)
        >>> x = torch.randn(32, 22, 42, 13)
        >>> edge_index = torch.randint(0, 42, (2, 100))
        >>> edge_attr = torch.randn(100, 2)
        >>> preds = model(x, edge_index, edge_attr)  # (32, 42, 3)
    """

    def __init__(
        self,
        patchtst_config: PatchTSTModelConfig,
        gnn_config: GNNModelConfig,
        input_dim: int,
        output_horizons: int,
        graph: Data | None = None,
        volume_feature_idx: int | None = None,
    ):
        """Initialize ensemble model.

        Args:
            patchtst_config: Configuration for PatchTST temporal encoder.
            gnn_config: Configuration for GNN cross-sectional encoder.
            input_dim: Number of input features per node per time step.
            output_horizons: Number of prediction horizons (e.g., 3 for 5d/10d/21d).
            graph: Optional surface graph with edge_index and edge_attr. If provided,
                edge structure is stored internally. If None, edge_index/edge_attr must
                be provided to forward() method.
            volume_feature_idx: Index of log_volume in input features. Required when
                gnn_config.use_dynamic_volume_edges is True.
        """
        super().__init__()
        self._patchtst_config = patchtst_config
        self._gnn_config = gnn_config
        self._input_dim = input_dim
        self._output_horizons = output_horizons
        self._volume_feature_idx = volume_feature_idx

        # Store graph structure if provided
        if graph is not None:
            # Edge index is always a buffer (not learnable)
            self.register_buffer('_edge_index', graph.edge_index.clone())

            # Edge attributes: learnable parameter or fixed buffer
            if graph.edge_attr is not None:
                if gnn_config.use_learnable_edge_attr:
                    # Learnable: appears in model.parameters()
                    self._edge_attr = nn.Parameter(
                        graph.edge_attr.clone(),
                        requires_grad=True,
                    )
                else:
                    # Fixed: stored in state_dict but no gradients
                    self.register_buffer('_edge_attr', graph.edge_attr.clone())
            else:
                self._edge_attr = None
        else:
            # No graph provided - backward compatibility mode
            self._edge_index = None
            self._edge_attr = None

        # PatchTST for temporal encoding (no prediction head)
        self._patchtst = PatchTSTModel(
            config=patchtst_config,
            input_dim=input_dim,
            output_horizons=0,  # No head - we use GNN output
        )

        # GNN for cross-sectional encoding
        # Input dimension is PatchTST's d_model
        self._gnn = SurfaceGNN(
            config=gnn_config,
            input_dim=patchtst_config.d_model,
        )

        # Prediction head
        self._head = nn.Linear(gnn_config.hidden_dim, output_horizons)

    def _augment_edges_with_volume(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Augment static edge attributes with per-sample volume gradient.

        Computes |log_volume_src - log_volume_dst| for each edge in each sample,
        then concatenates with the static edge attributes to produce a 3D tensor.

        Args:
            x: Raw input features (batch, time, nodes, features).
            edge_index: Graph connectivity (2, num_edges).
            edge_attr: Static edge attributes (num_edges, D) or None.

        Returns:
            Augmented edge attributes (batch, num_edges, D+1), or original
            edge_attr if dynamic volume edges are disabled.
        """
        if (
            not self._gnn_config.use_dynamic_volume_edges
            or self._volume_feature_idx is None
        ):
            return edge_attr

        # Extract log_volume at last timestep: (batch, nodes)
        vol = x[:, -1, :, self._volume_feature_idx]
        vol = torch.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute per-edge volume gradient: |vol_src - vol_dst|
        vol_src = vol[:, edge_index[0]]  # (batch, num_edges)
        vol_dst = vol[:, edge_index[1]]  # (batch, num_edges)
        vol_diff = torch.abs(vol_src - vol_dst)  # (batch, num_edges)

        if edge_attr is not None:
            # Expand static attrs: (num_edges, D) -> (batch, num_edges, D)
            static_expanded = edge_attr.unsqueeze(0).expand(
                x.size(0), -1, -1
            )
            # Concatenate: (batch, num_edges, D+1)
            return torch.cat([static_expanded, vol_diff.unsqueeze(-1)], dim=-1)
        else:
            # Volume diff only: (batch, num_edges, 1)
            return vol_diff.unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through ensemble.

        Args:
            x: Input tensor of shape (batch, time_steps, num_nodes, input_dim).
            edge_index: Graph connectivity of shape (2, num_edges). If None, uses
                internal edge_index from graph provided at initialization.
            edge_attr: Edge attributes of shape (num_edges, 2). If None, uses
                internal edge_attr (learnable or fixed depending on config).
            mask: Optional boolean mask of shape (batch, num_nodes) indicating
                valid nodes. Invalid nodes will have their predictions zeroed.

        Returns:
            Predictions of shape (batch, num_nodes, output_horizons).
        """
        # Resolve edge_index (external arg takes precedence over internal)
        edge_index_used = edge_index if edge_index is not None else self._edge_index
        if edge_index_used is None:
            raise ValueError(
                "edge_index must be provided either during initialization (via graph) "
                "or as forward() argument"
            )

        # Resolve edge_attr (external arg takes precedence over internal)
        edge_attr_used = edge_attr if edge_attr is not None else self._edge_attr

        # Augment edges with per-sample volume gradient (before PatchTST encoding)
        edge_attr_used = self._augment_edges_with_volume(
            x, edge_index_used, edge_attr_used
        )

        # Step 1: Temporal encoding
        # (batch, time, nodes, features) -> (batch, nodes, d_model)
        temporal_emb = self._patchtst.encode(x)

        # Step 2: Cross-sectional encoding
        # (batch, nodes, d_model) -> (batch, nodes, hidden_dim)
        graph_emb = self._gnn(temporal_emb, edge_index_used, edge_attr_used)

        # Step 3: Prediction head
        # (batch, nodes, hidden_dim) -> (batch, nodes, horizons)
        preds: torch.Tensor = self._head(graph_emb)

        # Step 4: Apply mask if provided
        if mask is not None:
            # mask: (batch, nodes) -> (batch, nodes, 1)
            preds = preds * mask.unsqueeze(-1).float()

        return preds

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract embeddings without prediction head.

        Useful for analysis or custom downstream tasks.

        Args:
            x: Input tensor of shape (batch, time_steps, num_nodes, input_dim).
            edge_index: Graph connectivity. If None, uses internal edge_index.
            edge_attr: Edge attributes. If None, uses internal edge_attr.

        Returns:
            Node embeddings of shape (batch, num_nodes, hidden_dim).
        """
        # Resolve edge_index and edge_attr
        edge_index_used = edge_index if edge_index is not None else self._edge_index
        if edge_index_used is None:
            raise ValueError("edge_index required")

        edge_attr_used = edge_attr if edge_attr is not None else self._edge_attr

        # Augment edges with per-sample volume gradient
        edge_attr_used = self._augment_edges_with_volume(
            x, edge_index_used, edge_attr_used
        )

        temporal_emb = self._patchtst.encode(x)
        graph_emb: torch.Tensor = self._gnn(temporal_emb, edge_index_used, edge_attr_used)
        return graph_emb

    @property
    def patchtst(self) -> PatchTSTModel:
        """Return PatchTST component."""
        return self._patchtst

    @property
    def gnn(self) -> SurfaceGNN:
        """Return GNN component."""
        return self._gnn

    @property
    def patchtst_config(self) -> PatchTSTModelConfig:
        """Return PatchTST configuration."""
        return self._patchtst_config

    @property
    def gnn_config(self) -> GNNModelConfig:
        """Return GNN configuration."""
        return self._gnn_config

    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self._input_dim

    @property
    def output_horizons(self) -> int:
        """Return number of output horizons."""
        return self._output_horizons
