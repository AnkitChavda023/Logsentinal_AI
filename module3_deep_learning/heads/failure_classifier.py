"""
Task 1 — Binary failure classifier: will this service fail in the next 5 min?
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FailureClassifier(nn.Module):
    """
    Linear(256→64) → ReLU → Linear(64→1) → Sigmoid
    """

    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_services, 256) or (batch, 256)
        Returns:
            (batch, num_services, 1) failure probability in [0, 1].
        """
        return self.net(x)
