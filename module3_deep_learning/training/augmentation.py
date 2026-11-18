"""
Data Augmentation for Log Sequences

Implements augmentation techniques from Section 13 concept checklist:
1. Gaussian noise injection on continuous features
2. Time-warping / dropout of sequence positions

Operates on the per-sample dict LogSequenceDataset.__getitem__ produces
(named raw fields, not one packed tensor) — a named field can be perturbed
without knowing anything about any other field's column offset, which is
what a single flat "feature vector index" scheme couldn't do robustly.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class GaussianNoiseInjection:
    """Adds Gaussian noise to continuous (already 0-1-normalisable) fields."""

    _NORMALIZED_FIELDS = ("cpu_usage",)
    _UNBOUNDED_FIELDS = ("latency_ms", "memory_mb")

    def __init__(self, noise_std: float = 0.05, probability: float = 0.5) -> None:
        self.noise_std = noise_std
        self.probability = probability

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.probability:
            return sample

        mask = sample["attention_mask"].bool()
        for field in self._NORMALIZED_FIELDS:
            x = sample[field].clone()
            noise = torch.randn_like(x) * self.noise_std
            x[mask] = (x[mask] + noise[mask]).clamp(0.0, 1.0)
            sample[field] = x
        for field in self._UNBOUNDED_FIELDS:
            x = sample[field].clone()
            noise = torch.randn_like(x) * self.noise_std * x.clamp(min=1.0)
            x[mask] = (x[mask] + noise[mask]).clamp(min=0.0)
            sample[field] = x
        return sample


class SequenceDropout:
    """Randomly drops log entries, simulating missing/dropped logs."""

    _RAW_FIELDS = (
        "service_ids", "log_level_onehot", "error_code_ids", "latency_ms",
        "cpu_usage", "memory_mb", "time_since_error", "hour_of_day",
        "is_anomaly", "trace_depth",
    )

    def __init__(self, drop_rate: float = 0.1, probability: float = 0.5) -> None:
        self.drop_rate = drop_rate
        self.probability = probability

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.probability:
            return sample

        mask = sample["attention_mask"]
        valid_idx = torch.where(mask > 0)[0]
        num_valid = len(valid_idx)
        if num_valid < 10:  # too few log lines to safely thin out
            return sample

        num_keep = max(5, int(num_valid * (1 - self.drop_rate)))
        keep_idx = valid_idx[torch.randperm(num_valid)[:num_keep]].sort().values
        seq_len = mask.shape[0]

        new_sample = dict(sample)
        new_mask = torch.zeros_like(mask)
        new_mask[: len(keep_idx)] = 1.0
        new_sample["attention_mask"] = new_mask

        for field in self._RAW_FIELDS:
            x = sample[field]
            new_x = torch.zeros_like(x)
            new_x[: len(keep_idx)] = x[keep_idx]
            new_sample[field] = new_x

        return new_sample


class LogAugmentor:
    """Combined augmentation pipeline, applied to one dataset sample at a time."""

    def __init__(
        self,
        enable_noise: bool = True,
        enable_dropout: bool = True,
        noise_std: float = 0.05,
        drop_rate: float = 0.1,
        augment_probability: float = 0.5,
    ) -> None:
        self.steps = []
        if enable_noise:
            self.steps.append(GaussianNoiseInjection(noise_std=noise_std, probability=augment_probability))
        if enable_dropout:
            self.steps.append(SequenceDropout(drop_rate=drop_rate, probability=augment_probability))
        logger.info("LogAugmentor initialized with %d step(s)", len(self.steps))

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.steps:
            sample = step(sample)
        return sample
