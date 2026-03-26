"""Linear baseline model for ablation study.

The simplest possible learned model: a single shared linear layer applied
per-node on the last timestep. No temporal context, no graph structure.
Establishes the floor for what neural architecture adds beyond raw feature
relationships.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LinearBaselineConfig:
    """Configuration for LinearBaseline model.

    Attributes:
        dropout: Dropout rate applied to input features.
    """

    dropout: float = 0.0


class LinearBaseline(nn.Module):
    """Shared linear layer applied per-node on last timestep.

    Takes the last timestep of each node's features and maps directly
    to prediction horizons through a single shared linear layer. This
    establishes a minimum bar: any model that cannot beat this provides
    no useful inductive bias beyond a simple regression.

    Architecture:
        Input: (batch, time, nodes, features)
          -> Take last timestep: (batch, nodes, features)
          -> Dropout (if configured)
          -> Linear(features, horizons): (batch, nodes, horizons)
          -> Mask + Output

    Example:
        >>> config = LinearBaselineConfig()
        >>> model = LinearBaseline(config, input_dim=29, output_horizons=3)
        >>> x = torch.randn(32, 22, 42, 29)
        >>> preds = model(x)  # (32, 42, 3)
    """

    def __init__(
        self,
        config: LinearBaselineConfig,
        input_dim: int,
        output_horizons: int,
    ):
        """Initialize LinearBaseline model.

        Args:
            config: Model configuration.
            input_dim: Number of input features per node per time step.
            output_horizons: Number of prediction horizons (e.g., 3 for 5d/10d/21d).
        """
        super().__init__()
        self._config = config
        self._input_dim = input_dim
        self._output_horizons = output_horizons

        self._dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self._linear = nn.Linear(input_dim, output_horizons)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through linear baseline.

        Args:
            x: Input tensor of shape (batch, time_steps, num_nodes, input_dim).
            mask: Optional boolean mask of shape (batch, num_nodes) indicating
                valid nodes. Invalid nodes will have their predictions zeroed.

        Returns:
            Predictions of shape (batch, num_nodes, output_horizons).
        """
        # Take last timestep only — no temporal information
        x_last = x[:, -1, :, :]  # (batch, nodes, features)

        x_last = self._dropout(x_last)

        # Shared linear layer applied per-node
        preds: torch.Tensor = self._linear(x_last)  # (batch, nodes, horizons)

        # Apply mask if provided
        if mask is not None:
            preds = preds * mask.unsqueeze(-1).float()

        return preds

    @property
    def config(self) -> LinearBaselineConfig:
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
