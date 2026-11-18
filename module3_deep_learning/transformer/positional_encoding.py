"""
Timestamp-based sinusoidal positional encoding (§7.4).

Unlike index-based PE, this uses actual log timestamps (as seconds
since the window start) so the model knows *when* each log occurred.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Supports both:
    - Index-based (classic Transformer): position = 0, 1, 2, ...
    - Timestamp-based: position = elapsed_seconds
    """

    def __init__(self, d_model: int = 128, max_len: int = 500) -> None:
        super().__init__()
        self.d_model = d_model

        # Pre-compute standard sinusoidal table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x:          (batch, seq_len, d_model)
            timestamps: (batch, seq_len) optional; elapsed seconds from window start.
                        If provided, used instead of sequential indices.

        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        if timestamps is not None:
            # Timestamp-based: encode each position using its actual elapsed seconds
            # Normalise to [0, max_len)
            max_t = timestamps.max().item() + 1.0
            norm_pos = (timestamps / max_t * (self.pe.size(1) - 1)).long()
            norm_pos = norm_pos.clamp(0, self.pe.size(1) - 1)
            # Gather per-sample PE
            # pe: (1, max_len, d_model) → index with norm_pos: (batch, seq_len)
            batch_size = x.size(0)
            pos_enc = self.pe[0][norm_pos]   # (batch, seq_len, d_model)
        else:
            pos_enc = self.pe[:, :seq_len, :]  # (1, seq_len, d_model)

        return x + pos_enc
