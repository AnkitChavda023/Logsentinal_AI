"""
Log embedding layer — 37-dim input → 128-dim embedding vector (§7.2).

Component breakdown:
  Service name:        Learned embedding (16-dim)
  Log level:          One-hot (5-dim)
  Error code:         Learned embedding (8-dim, 0 if none)
  Latency (norm):     Min-max scalar (1-dim)
  CPU usage:          Passthrough normalised (1-dim)
  Memory usage:       Passthrough normalised (1-dim)
  Time since error:   Log-scaled (1-dim)
  Hour of day:        Sinusoidal sin+cos (2-dim)
  Is anomaly:         Binary (1-dim)
  Trace depth:        Normalised 0-1 (1-dim)
  ─────────────────── Subtotal = 37-dim
  → Linear(37 → 128)  Learnable projection
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LogEmbedder(nn.Module):
    """
    Converts a batch of raw log feature tensors to 128-dim embeddings.

    Input tensor shape: (batch, seq_len, raw_feature_dim=37) or
                        (batch, raw_feature_dim=37)
    """

    def __init__(
        self,
        num_services: int,
        num_error_codes: int = 32,
        embedding_dim: int = 128,
        service_embed_dim: int = 16,
        error_embed_dim: int = 8,
        max_latency_ms: float = 10_000.0,
        max_memory_mb: float = 8_192.0,
        max_trace_depth: int = 20,
        max_time_since_error: float = 3_600.0,
    ) -> None:
        super().__init__()
        self.num_services = num_services
        self.max_latency = max_latency_ms
        self.max_memory = max_memory_mb
        self.max_depth = max_trace_depth
        self.max_time_error = max_time_since_error

        self.service_embed = nn.Embedding(num_services, service_embed_dim)
        self.error_embed = nn.Embedding(num_error_codes + 1, error_embed_dim, padding_idx=0)

        # Raw dim: 16 (svc) + 5 (level) + 8 (error) + 1 + 1 + 1 + 1 + 2 + 1 + 1 = 37
        raw_dim = service_embed_dim + 5 + error_embed_dim + 1 + 1 + 1 + 1 + 2 + 1 + 1
        self.projection = nn.Linear(raw_dim, embedding_dim)

    def forward(
        self,
        service_ids: torch.Tensor,       # (B, T) or (B,)
        log_level_onehot: torch.Tensor,   # (B, T, 5)
        error_code_ids: torch.Tensor,     # (B, T) or (B,)
        latency_ms: torch.Tensor,         # (B, T)
        cpu_usage: torch.Tensor,          # (B, T)
        memory_mb: torch.Tensor,          # (B, T)
        time_since_error: torch.Tensor,   # (B, T)
        hour_of_day: torch.Tensor,        # (B, T)
        is_anomaly: torch.Tensor,         # (B, T)
        trace_depth: torch.Tensor,        # (B, T)
    ) -> torch.Tensor:
        """
        Returns:
            Embedded tensor of shape (B, T, 128) or (B, 128).
        """
        # Learned embeddings
        svc_emb = self.service_embed(service_ids)          # (B, T, 16)
        err_emb = self.error_embed(error_code_ids)         # (B, T, 8)

        # Normalised scalars (add embedding dimension)
        lat_norm = (latency_ms / self.max_latency).unsqueeze(-1)
        cpu_norm = cpu_usage.unsqueeze(-1)
        mem_norm = (memory_mb / self.max_memory).unsqueeze(-1)

        # Log-scale time since last error
        t_err = torch.log1p(time_since_error / 60.0).unsqueeze(-1)

        # Sinusoidal encoding for hour of day
        sin_h = torch.sin(2 * math.pi * hour_of_day / 24.0).unsqueeze(-1)
        cos_h = torch.cos(2 * math.pi * hour_of_day / 24.0).unsqueeze(-1)
        hour_enc = torch.cat([sin_h, cos_h], dim=-1)         # (B, T, 2)

        anom = is_anomaly.unsqueeze(-1).float()
        depth_norm = (trace_depth / self.max_depth).unsqueeze(-1)

        raw = torch.cat([
            svc_emb, log_level_onehot.float(), err_emb,
            lat_norm, cpu_norm, mem_norm, t_err,
            hour_enc, anom, depth_norm,
        ], dim=-1)  # (B, T, 37)

        return self.projection(raw)   # (B, T, 128)
