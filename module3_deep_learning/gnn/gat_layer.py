"""
GAT layer — primary GNN model (§7.5).

h_v^(k+1) = σ(Σ_{u∈N(v)} α_vu · W · h_u^(k))

Attention weight α_vu learns how much each neighbour matters for
predicting the node's failure (e.g., database health matters more
to payment-service than load-balancer health does).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.
    3-step message passing for a 3-hop receptive field.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # out_channels // heads per head; concatenate → out_channels
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.conv = GATConv(
            in_channels,
            out_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Args:
            x:           (num_nodes, in_channels)
            edge_index:  (2, num_edges)

        Returns:
            (num_nodes, out_channels) or ((num_nodes, out_channels), alpha)
        """
        if return_attention_weights:
            out, (edge_idx, alpha) = self.conv(
                x, edge_index, return_attention_weights=True
            )
            out = self.norm(self.act(out))
            return out, alpha
        else:
            out = self.conv(x, edge_index)
            return self.norm(self.act(out))


class GATStack(nn.Module):
    """
    3-layer GAT stack for 3-hop message passing.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        out_channels: int = 256,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(in_channels, hidden_channels, heads, dropout),
            GATLayer(hidden_channels, hidden_channels, heads, dropout),
            GATLayer(hidden_channels, out_channels, heads, dropout),
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
