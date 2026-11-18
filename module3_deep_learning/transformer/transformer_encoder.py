"""
4-layer, 8-head Transformer encoder for cross-service attention (§7.4).

Architecture:
  All logs (last 5 min) → PositionalEncoding →
  [TransformerBlock × 4] → per-log 256-dim representation

Per-service pooling (mean over each service's log positions within the
window) happens one level up, in models/full_model.py — this class only
contextualises each individual log line against every other log line in
the window; it doesn't know about service boundaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from module3_deep_learning.transformer.positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    """
    4-layer Transformer encoder with 8-head self-attention.
    Learns cross-service correlations via full self-attention over
    the entire 5-minute log window.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        # Project d_model -> output_dim to match the sequence encoders'
        # hidden_dim (full_model.py concatenates this branch's output with
        # theirs). output_dim used to be hardcoded to 256, silently coupled
        # to config/defaults.py's SEQ_HIDDEN_DIM default only by
        # coincidence — changing seq_hidden_dim without also changing this
        # literal would raise a matrix-shape error in full_model.py's
        # self.combine the moment the two stopped matching.
        self.output_proj = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                    (batch, seq_len, d_model)
            src_key_padding_mask: (batch, seq_len) bool mask; True = padding.
            timestamps:           (batch, seq_len) elapsed seconds.

        Returns:
            (batch, seq_len, output_dim) — contextualised per-log representations.
        """
        x = self.pos_enc(x, timestamps)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_proj(out)   # (batch, seq_len, output_dim)