"""PatchTST model for per-node temporal encoding.

This module implements the PatchTST architecture for time series forecasting
on volatility surface nodes. Each node is processed independently through
the same transformer encoder, enabling per-node temporal pattern learning.

Reference: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
https://arxiv.org/abs/2211.14730
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.patchtst.encoder import PatchEmbedding


@dataclass
class PatchTSTModelConfig:
    """Configuration for PatchTST model.

    Attributes:
        patch_len: Length of each patch in time steps.
        stride: Stride between consecutive patches.
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward layer dimension.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout rate.
    """

    patch_len: int = 12
    stride: int = 6
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    n_layers: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal PE, got {self.d_model}")


class PatchTSTModel(nn.Module):
    """PatchTST model for per-node temporal encoding.

    Processes time series data for each node independently using a shared
    transformer encoder with patching. This enables efficient temporal
    pattern learning while respecting the per-node structure.

    Architecture:
        1. Reshape: (batch, time, nodes, features) -> (batch*nodes, time, features)
        2. Patch embedding: (batch*nodes, time, features) -> (batch*nodes, patches, d_model)
        3. Transformer encoder: (batch*nodes, patches, d_model) -> (batch*nodes, patches, d_model)
        4. Mean pooling: (batch*nodes, patches, d_model) -> (batch*nodes, d_model)
        5. Prediction head: (batch*nodes, d_model) -> (batch*nodes, horizons)
        6. Reshape: (batch*nodes, horizons) -> (batch, nodes, horizons)

    Example:
        >>> config = PatchTSTModelConfig(d_model=128, n_layers=3)
        >>> model = PatchTSTModel(config, input_dim=13, output_horizons=3)
        >>> x = torch.randn(32, 22, 42, 13)  # (batch, time, nodes, features)
        >>> mask = torch.ones(32, 42, dtype=torch.bool)
        >>> preds = model(x, mask)  # (32, 42, 3)
    """

    def __init__(
        self,
        config: PatchTSTModelConfig,
        input_dim: int,
        output_horizons: int,
    ):
        """Initialize PatchTST model.

        Args:
            config: Model configuration.
            input_dim: Number of input features per node per time step.
            output_horizons: Number of prediction horizons (e.g., 3 for 5d/10d/21d).
        """
        super().__init__()
        self._config = config
        self._input_dim = input_dim
        self._output_horizons = output_horizons

        # Patch embedding layer
        self._patch_embedding = PatchEmbedding(
            patch_len=config.patch_len,
            stride=config.stride,
            input_dim=input_dim,
            d_model=config.d_model,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # Prediction head (only if output_horizons > 0)
        self._head: nn.Linear | None
        if output_horizons > 0:
            self._head = nn.Linear(config.d_model, output_horizons)
        else:
            self._head = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through PatchTST model.

        Args:
            x: Input tensor of shape (batch, time_steps, num_nodes, input_dim).
            mask: Optional boolean mask of shape (batch, num_nodes) indicating
                valid nodes. Invalid nodes will have their predictions zeroed.

        Returns:
            Predictions of shape (batch, num_nodes, output_horizons).

        Raises:
            ValueError: If output_horizons is 0 (use encode() instead).
        """
        if self._head is None:
            raise ValueError(
                "Cannot call forward() when output_horizons=0. Use encode() instead."
            )

        batch_size, time_steps, num_nodes, input_dim = x.shape

        # Reshape: (batch, time, nodes, features) -> (batch * nodes, time, features)
        x = x.permute(0, 2, 1, 3)  # (batch, nodes, time, features)
        x = x.reshape(batch_size * num_nodes, time_steps, input_dim)

        # Patch embedding: (batch * nodes, time, features) -> (batch * nodes, patches, d_model)
        x = self._patch_embedding(x)

        # Transformer encoding: (batch * nodes, patches, d_model)
        x = self._encoder(x)

        # Mean pooling over patches: (batch * nodes, d_model)
        x = x.mean(dim=1)

        # Prediction head: (batch * nodes, horizons)
        preds: torch.Tensor = self._head(x)

        # Reshape: (batch, nodes, horizons)
        preds = preds.reshape(batch_size, num_nodes, self._output_horizons)

        # Apply mask if provided
        if mask is not None:
            # mask: (batch, nodes) -> (batch, nodes, 1)
            preds = preds * mask.unsqueeze(-1).float()

        return preds

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without prediction head.

        This method is useful for ensemble models where PatchTST provides
        temporal embeddings that are further processed by a GNN.

        Args:
            x: Input tensor of shape (batch, time_steps, num_nodes, input_dim).

        Returns:
            Node embeddings of shape (batch, num_nodes, d_model).
        """
        batch_size, time_steps, num_nodes, input_dim = x.shape

        # Reshape: (batch, time, nodes, features) -> (batch * nodes, time, features)
        x = x.permute(0, 2, 1, 3)  # (batch, nodes, time, features)
        x = x.reshape(batch_size * num_nodes, time_steps, input_dim)

        # Patch embedding: (batch * nodes, time, features) -> (batch * nodes, patches, d_model)
        x = self._patch_embedding(x)

        # Transformer encoding: (batch * nodes, patches, d_model)
        x = self._encoder(x)

        # Mean pooling over patches: (batch * nodes, d_model)
        x = x.mean(dim=1)

        # Reshape: (batch, nodes, d_model)
        embeddings = x.reshape(batch_size, num_nodes, self._config.d_model)

        return embeddings

    @property
    def config(self) -> PatchTSTModelConfig:
        """Return model configuration."""
        return self._config

    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self._input_dim

    @property
    def output_horizons(self) -> int:
        """Return number of output horizons."""
        return self._output_horizons

    @property
    def d_model(self) -> int:
        """Return model dimension."""
        return self._config.d_model
