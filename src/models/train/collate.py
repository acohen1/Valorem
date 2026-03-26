"""Custom collate functions for DataLoader.

This module provides collate functions for batching surface data samples.
"""

import torch
from torch import Tensor


def surface_collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate batch of surface dataset samples.

    Stacks X, y, mask, and label_mask tensors. Assumes all samples share the same
    graph structure (static graph for the volatility surface).

    Args:
        batch: List of sample dictionaries with keys:
            - 'X': Input features, shape (time, nodes, features)
            - 'y': Target values, shape (nodes, horizons)
            - 'mask': Valid node mask, shape (nodes,)
            - 'label_mask': Valid label mask, shape (nodes, horizons), optional

    Returns:
        Dictionary with batched tensors:
            - 'X': shape (batch, time, nodes, features)
            - 'y': shape (batch, nodes, horizons)
            - 'mask': shape (batch, nodes)
            - 'label_mask': shape (batch, nodes, horizons)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=surface_collate_fn)
    """
    X = torch.stack([sample["X"] for sample in batch])
    y = torch.stack([sample["y"] for sample in batch])
    mask = torch.stack([sample["mask"] for sample in batch])

    if "label_mask" in batch[0]:
        label_mask = torch.stack([sample["label_mask"] for sample in batch])
    else:
        # Backward compatibility for synthetic/test datasets that only provide
        # node-level masks.
        #
        # .expand_as() returns a view with shared storage along the expanded
        # dimension; pin-memory cannot handle overlapping writes on that view.
        # Clone to materialize an independent tensor.
        label_mask = mask.unsqueeze(-1).expand_as(y).clone()

    return {
        "X": X,
        "y": y,
        "mask": mask,
        "label_mask": label_mask,
    }
