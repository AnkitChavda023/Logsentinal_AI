"""
GCN layer — ablation baseline (§7.5).

h_v^(k+1) = σ(W · MEAN({h_u^(k) : u ∈ N(v) ∪ {v}}))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Network layer with ReLU activation.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.conv(x, edge_index, edge_weight)
        return self.norm(self.act(out))


class GCNStack(nn.Module):
    """
    Multi-layer GCN stack — ablation counterpart to GATStack, so the
    two "primary vs ablation" GNN options share the same call signature
    and can be swapped via ModelConfig.use_gat with no other code changes.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
