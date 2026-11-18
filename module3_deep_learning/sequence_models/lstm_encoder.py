"""
LSTM encoder — 2-layer LSTM with hidden=256, dropout=0.3 (§7.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    2-layer LSTM that encodes a variable-length sequence of 128-dim
    log embeddings into a 256-dim per-service health vector.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (batch, seq_len, 128)
            lengths: (batch,) actual sequence lengths for packing.

        Returns:
            (batch, seq_len, 256) — per-timestep hidden states, so callers
            can pool them per-service (a window mixes logs from every
            service; only after pooling does "one vector per service" exist).
        """
        seq_len = x.size(1)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
        else:
            out, _ = self.lstm(x)

        return out   # (batch, seq_len, 256)
