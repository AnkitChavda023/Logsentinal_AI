"""
Vanilla RNN encoder — same shape as LSTM/GRU, the lightest-weight
sequential baseline (no gating), single direction (§7.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """
    2-layer vanilla (tanh) RNN encoder with hidden=256, dropout=0.3.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
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
            (batch, seq_len, 256) — per-timestep hidden states (see
            LSTMEncoder.forward for why per-timestep, not last-only).
        """
        seq_len = x.size(1)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
        else:
            out, _ = self.rnn(x)

        return out   # (batch, seq_len, 256)
