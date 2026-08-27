"""Residual gated GNN used by SAGE-HC.

The reported experiments use three ``ResGatedGraphConv`` layers with widths
256 -> 128 -> 64, batch normalization, ReLU activations, and a linear scalar
prediction head. During evaluation the training utilities clamp predictions to
[0, 1], matching the code path used for the reported results.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import ResGatedGraphConv


class ResidualGatedGCN(nn.Module):
    """SAGE-HC residual gated graph regressor."""

    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        norm: str = "batch",
    ) -> None:
        super().__init__()

        if norm == "batch":
            norm_layer = nn.BatchNorm1d
        elif norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "instance":
            norm_layer = nn.InstanceNorm1d
        else:
            raise ValueError("norm must be one of: batch, layer, instance")

        dims = [num_features, *hidden_dims]
        self.convs = nn.ModuleList(
            ResGatedGraphConv(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(norm_layer(dim) for dim in dims[1:])
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(conv(x, edge_index))
            x = norm(x)
        return self.head(x)
