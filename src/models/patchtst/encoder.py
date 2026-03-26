"""Patch embedding layer for PatchTST.

This module implements the patching and embedding step that converts
time series data into a sequence of patch embeddings for transformer processing.
"""

import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert time series to patch embeddings.

    Takes a time series of shape (batch, time_steps, input_dim) and converts it
    to patch embeddings of shape (batch, num_patches, d_model).

    The patching process:
    1. Divide the time series into overlapping patches using a sliding window
    2. Flatten each patch to (patch_len * input_dim)
    3. Project to d_model dimensions via a linear layer
    4. Add learnable positional encoding

    Example:
        >>> embed = PatchEmbedding(patch_len=12, stride=6, input_dim=13, d_model=128)
        >>> x = torch.randn(32, 22, 13)  # (batch, time, features)
        >>> out = embed(x)  # (32, 2, 128) - 2 patches with stride 6
    """

    def __init__(
        self,
        patch_len: int,
        stride: int,
        input_dim: int,
        d_model: int,
    ):
        """Initialize patch embedding layer.

        Args:
            patch_len: Length of each patch in time steps.
            stride: Stride between consecutive patches.
            input_dim: Number of input features per time step.
            d_model: Output embedding dimension.
        """
        super().__init__()
        self._patch_len = patch_len
        self._stride = stride
        self._input_dim = input_dim
        self._d_model = d_model

        # Linear projection: (patch_len * input_dim) -> d_model
        self._proj = nn.Linear(patch_len * input_dim, d_model)

        # Sinusoidal positional encoding (non-learnable).
        # Stored as a buffer so it moves with the model but does not
        # participate in the backward pass (no wasted gradient storage).
        self._max_num_patches = 100
        self.register_buffer(
            "_pos_encoding",
            self._create_positional_encoding(self._max_num_patches, d_model),
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding.

        Args:
            max_len: Maximum sequence length.
            d_model: Embedding dimension.

        Returns:
            Positional encoding tensor of shape (max_len, d_model).
        """
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert time series to patch embeddings.

        Args:
            x: Input tensor of shape (batch, time_steps, input_dim).

        Returns:
            Patch embeddings of shape (batch, num_patches, d_model).

        Raises:
            ValueError: If time_steps < patch_len.
        """
        batch_size, time_steps, input_dim = x.shape

        if time_steps < self._patch_len:
            raise ValueError(
                f"Time steps ({time_steps}) must be >= patch_len ({self._patch_len})"
            )

        # Calculate number of patches
        num_patches = (time_steps - self._patch_len) // self._stride + 1

        # Extract patches using unfold
        # unfold(dimension, size, step) -> adds a dimension with unfolded slices
        # x: (batch, time_steps, input_dim)
        # After unfold on dim 1: (batch, num_patches, input_dim, patch_len)
        patches = x.unfold(dimension=1, size=self._patch_len, step=self._stride)

        # Reshape to (batch, num_patches, patch_len * input_dim)
        # patches shape after unfold: (batch, num_patches, input_dim, patch_len)
        patches = patches.permute(0, 1, 3, 2)  # (batch, num_patches, patch_len, input_dim)
        patches = patches.reshape(batch_size, num_patches, -1)

        # Project to d_model
        embeddings = self._proj(patches)  # (batch, num_patches, d_model)

        # Add positional encoding
        result: torch.Tensor = embeddings + self._pos_encoding[:num_patches].unsqueeze(0)

        return result

    @property
    def patch_len(self) -> int:
        """Return patch length."""
        return self._patch_len

    @property
    def stride(self) -> int:
        """Return stride between patches."""
        return self._stride

    @property
    def d_model(self) -> int:
        """Return embedding dimension."""
        return self._d_model

    def compute_num_patches(self, time_steps: int) -> int:
        """Compute number of patches for a given sequence length.

        Args:
            time_steps: Number of time steps in input sequence.

        Returns:
            Number of patches that will be created.
        """
        return (time_steps - self._patch_len) // self._stride + 1
