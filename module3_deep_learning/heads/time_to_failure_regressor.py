"""
Task 3 — Time-to-failure regression (0–300 seconds).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeToFailureRegressor(nn.Module):
    """
    Linear(256→64) → ReLU → Linear(64→1)
    Output range: [0, 300] seconds.
    """

    def __init__(self, input_dim: int = 256, max_seconds: float = 300.0) -> None:
        super().__init__()
        self.max_seconds = max_seconds
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # → [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (batch, num_services, 1) predicted seconds until failure.
        """
        return self.net(x) * self.max_seconds
