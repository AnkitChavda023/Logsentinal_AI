"""
Attention-weighted per-service pooling (LogRobust, Zhang et al. ESEC/FSE'19, §3.3 Eq. 2-3).

LogRobust's attention-based Bi-LSTM learns a per-log-event importance weight
instead of treating every line in a sequence as equally informative, so
noisy/irrelevant lines are automatically down-weighted rather than diluting
the pooled representation. `models/full_model.py`'s previous per-service
pooling (`_pool_by_service`) was a plain masked mean — exactly the "every
line counts equally" gap LogRobust's attention layer exists to close. This
is the direct per-service analogue, adapted to a dependent multi-service
window: attention scores are computed once over the fused
(sequence+transformer) representation, then softmax-normalized independently
per service over that service's own masked log positions, so each service
node gets its own importance weighting over the lines that belong to it.

Strict generalization of the old behavior: forcing the scores to be uniform
reproduces the previous mean-pool exactly, so this can only add
expressiveness, never remove it.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPooling(nn.Module):
    """Pools per-log representations into one vector per service using a
    learned attention weight per log line (LogRobust §3.3 Eq. 2:
    alpha_t = tanh(W_a . h_t)), instead of a uniform average.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        x: Tensor,             # (B, T, D)
        service_ids: Tensor,   # (B, T) long
        num_services: int,
        mask: Tensor,          # (B, T) bool, True = real log line
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            pooled:  (B, num_services, D)
            weights: (B, num_services, T) — attention weight of every log
                position for every service (0 outside that service's own
                masked positions and for services with no real log lines in
                this window); returned for interpretability.
        """
        B, T, D = x.shape
        raw_scores = self.score(x).squeeze(-1)  # (B, T)

        pooled = x.new_zeros(B, num_services, D)
        weights = x.new_zeros(B, num_services, T)
        for s in range(num_services):
            svc_mask = (service_ids == s) & mask          # (B, T)
            has_any = svc_mask.any(dim=1)                   # (B,)

            # softmax(all -inf) is NaN, not 0 — a (batch, service) pair with
            # zero real log lines in this window needs its own guard, same
            # edge case _pool_by_service handled via counts.clamp(min=1).
            masked_scores = raw_scores.masked_fill(~svc_mask, float("-inf"))
            safe_scores = torch.where(
                has_any.unsqueeze(-1), masked_scores, torch.zeros_like(masked_scores)
            )
            attn = torch.softmax(safe_scores, dim=1)         # (B, T)
            attn = attn * svc_mask.float()                    # 0 outside the service's own
            # positions; for a service absent from this window svc_mask is
            # all-False so attn collapses to all-zero, matching the old
            # pool's 0/clamp(1)==0 behavior for the same case.

            weights[:, s, :] = attn
            pooled[:, s, :] = (x * attn.unsqueeze(-1)).sum(dim=1)

        return pooled, weights
