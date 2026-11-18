"""
Task 2 — 12-class failure type classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FailureTypeClassifier(nn.Module):
    """
    Linear(256→64) → ReLU → Linear(64→12) → Softmax
    """

    def __init__(self, input_dim: int = 256, num_classes: int = 12) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Logits (batch, num_services, 12).
            Apply softmax externally for probabilities.
        """
        return self.net(x)
