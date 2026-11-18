"""
Bidirectional RNN encoder — reads forward and backward (§7.3).
Output: 512-dim (256 forward + 256 backward) projected to 256-dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiRNNEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.birnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        # Project 512 → 256
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            (batch, seq_len, 256) — per-timestep forward+backward states,
            projected down from 512 to 256 (see LSTMEncoder.forward for
            why per-timestep, not last-only).
        """
        seq_len = x.size(1)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.birnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
        else:
            out, _ = self.birnn(x)

        return self.projection(out)   # (batch, seq_len, 256)
