"""
Quantitative per-service count/statistics branch (LogAnomaly, Meng et al.
IJCAI'19, §3.3): alongside the sequential (LSTM) branch, LogAnomaly feeds a
count vector of template occurrences per window through its own LSTM, and
its ablation (Table/Fig. in §4.2) shows this quantitative branch measurably
improves precision/recall over sequential-only — the two branches catch
different failure signatures (a sudden log-level/error burst is a
quantitative pattern; a subtle ordering change is a sequential one).

Adapted here as a compact, deterministic per-service statistics vector
computed directly from the raw per-log tensors LogEmbedder already consumes
(no dataset.py changes) — a low-variance signal that doesn't depend on the
sequence encoder reliably learning to notice a volume/error burst from a
small number of training windows. All ops mirror the masked-sum pattern
`models/full_model.py`'s `_pool_by_service` already uses, so behavior for
a service absent from a window (all-zero features, not NaN) matches the
rest of the pipeline.

Feature layout (QUANT_FEATURE_DIM = 14), per (batch, service):
  [0]     log volume, normalized by window capacity T
  [1:6]   log-level fractions (DEBUG/INFO/WARN/ERROR/FATAL — same 5-dim
          one-hot order as LogEmbedder's log_level_onehot input)
  [6]     error rate (fraction of lines with a non-zero error_code_id)
  [7]     anomaly rate (fraction of lines already flagged is_anomaly)
  [8]     mean latency (normalized by cfg.MAX_LATENCY_MS)
  [9]     max latency  (normalized by cfg.MAX_LATENCY_MS) — spike detector
  [10]    mean cpu usage (already 0-1)
  [11]    max cpu usage  (already 0-1) — spike detector
  [12]    mean memory usage (normalized by cfg.MAX_MEMORY_MB)
  [13]    max memory usage  (normalized by cfg.MAX_MEMORY_MB) — spike detector
"""

from __future__ import annotations

import torch
from torch import Tensor

from module3_deep_learning.config import defaults as cfg

QUANT_FEATURE_DIM = 14


def _masked_mean(values: Tensor, svc_mask: Tensor, counts: Tensor) -> Tensor:
    """values, svc_mask: (B, T). counts: (B,) already clamped to >= 1."""
    return (values * svc_mask.float()).sum(dim=1) / counts


def _masked_max(values: Tensor, svc_mask: Tensor, has_any: Tensor) -> Tensor:
    """Masked max over T, 0 for a (batch, service) with no real log lines
    (mirrors the rest of this module's all-zero-for-absent-service
    convention rather than propagating -inf)."""
    filled = values.masked_fill(~svc_mask, float("-inf"))
    max_val = filled.max(dim=1).values
    return torch.where(has_any, max_val, torch.zeros_like(max_val))


def compute_quantitative_features(
    service_ids: Tensor,        # (B, T) long
    log_level_onehot: Tensor,   # (B, T, 5) float
    error_code_ids: Tensor,     # (B, T) long, 0 = no error
    latency_ms: Tensor,         # (B, T) float
    cpu_usage: Tensor,          # (B, T) float, already 0-1
    memory_mb: Tensor,          # (B, T) float
    is_anomaly: Tensor,         # (B, T) float/bool
    attention_mask: Tensor,     # (B, T) 1 = real log line, 0 = padding
    num_services: int,
) -> Tensor:
    """Returns (B, num_services, QUANT_FEATURE_DIM)."""
    B, T = service_ids.shape
    mask = attention_mask.bool()
    is_anomaly_f = is_anomaly.float()
    has_error = (error_code_ids != 0).float()

    features = service_ids.new_zeros(B, num_services, QUANT_FEATURE_DIM, dtype=latency_ms.dtype)

    for s in range(num_services):
        svc_mask = (service_ids == s) & mask               # (B, T)
        has_any = svc_mask.any(dim=1)                        # (B,)
        counts = svc_mask.sum(dim=1).clamp(min=1).float()    # (B,)

        volume = svc_mask.sum(dim=1).float() / T
        level_fracs = (log_level_onehot * svc_mask.unsqueeze(-1).float()).sum(dim=1) / counts.unsqueeze(-1)  # (B, 5)
        error_rate = _masked_mean(has_error, svc_mask, counts)
        anomaly_rate = _masked_mean(is_anomaly_f, svc_mask, counts)

        mean_latency = _masked_mean(latency_ms, svc_mask, counts) / cfg.MAX_LATENCY_MS
        max_latency = _masked_max(latency_ms, svc_mask, has_any) / cfg.MAX_LATENCY_MS
        mean_cpu = _masked_mean(cpu_usage, svc_mask, counts)
        max_cpu = _masked_max(cpu_usage, svc_mask, has_any)
        mean_memory = _masked_mean(memory_mb, svc_mask, counts) / cfg.MAX_MEMORY_MB
        max_memory = _masked_max(memory_mb, svc_mask, has_any) / cfg.MAX_MEMORY_MB

        row = torch.cat([
            volume.unsqueeze(-1), level_fracs,
            error_rate.unsqueeze(-1), anomaly_rate.unsqueeze(-1),
            mean_latency.unsqueeze(-1), max_latency.unsqueeze(-1),
            mean_cpu.unsqueeze(-1), max_cpu.unsqueeze(-1),
            mean_memory.unsqueeze(-1), max_memory.unsqueeze(-1),
        ], dim=-1)  # (B, 14)
        features[:, s, :] = row

    return features
