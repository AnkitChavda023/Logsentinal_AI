"""
Monte Carlo Dropout uncertainty estimation (§7.7).

Dropout stays ACTIVE at inference. Run T=50 forward passes;
mean = final prediction, std = uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to training mode (keep dropout active at inference)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_predict(
    model: nn.Module,
    batch,
    T: int = 50,
    task: str = "failure",
) -> dict[str, torch.Tensor]:
    """
    Run T stochastic forward passes and return mean + std for a single task.

    Args:
        model:  The LogSentinelModel (or any nn.Module with dropout).
        batch:  Input batch (passed directly to model.forward).
        T:      Number of MC samples.
        task:   Which output to aggregate ('failure', 'type', 'time').

    Returns:
        Dict with keys 'mean', 'std', 'lower_ci', 'upper_ci'.
    """
    all_tasks = mc_predict_all(model, batch, T=T)
    return all_tasks[task]


def mc_predict_all(
    model: nn.Module,
    batch,
    T: int = 50,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Run T stochastic forward passes ONCE and aggregate all three tasks
    (failure, type, time) from the same samples, instead of re-running
    the model T times per task — 3x cheaper than calling mc_predict()
    once per task.

    Returns:
        {'failure': {...}, 'type': {...}, 'time': {...}}, each with
        keys 'mean', 'std', 'lower_ci', 'upper_ci'.
    """
    model.eval()
    enable_dropout(model)

    failure_samples, type_samples, time_samples = [], [], []
    with torch.no_grad():
        for _ in range(T):
            output = model(batch)
            failure_samples.append(output.failure_probability)
            type_samples.append(torch.softmax(output.failure_type_logits, dim=-1))
            time_samples.append(output.time_to_failure)

    z = 1.96  # 95% CI

    def _summarize(samples: list[torch.Tensor], clamp: tuple[float, float] | None) -> dict[str, torch.Tensor]:
        stacked = torch.stack(samples, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        lower, upper = mean - z * std, mean + z * std
        if clamp is not None:
            lower, upper = lower.clamp(min=clamp[0]), upper.clamp(max=clamp[1])
        return {"mean": mean, "std": std, "lower_ci": lower, "upper_ci": upper}

    return {
        "failure": _summarize(failure_samples, clamp=(0.0, 1.0)),
        "type": _summarize(type_samples, clamp=(0.0, 1.0)),
        "time": _summarize(time_samples, clamp=None),
    }
