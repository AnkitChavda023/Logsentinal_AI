
from __future__ import annotations

import math

import numpy as np


_rng: np.random.Generator = np.random.default_rng()


def seed_rng(seed: int) -> None:
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _rng


def sample_latency(low: float, high: float, cap_multiplier: float = 20.0) -> float:
    """
    Log-normal latency sample: right-skewed, long-tailed, effectively no
    hard ceiling — the well-documented real-world shape of production
    request latency (KS-test-verified against a truncated Gaussian in
    scripts/validate_realism.py's clean pre-failure-baseline check, which
    is what originally caught this as a disclosed gap: skew was ~0 and the
    Normal fit strictly beat the Log-normal fit — the previous
    `_rng.normal(...)` + `np.clip(low, high)` implementation this replaces).

    [low, high] is treated as the *typical* operating range (~87% of the
    distribution's mass, +/-1.5 log-space standard deviations around the
    geometric-mean median) rather than an absolute min/max — real latency
    has no absolute ceiling, occasional draws above `high` are the point,
    not a bug. `cap_multiplier` is only a generous backstop against
    pathological outliers a pure unbounded log-normal tail could otherwise
    produce (e.g. a value 1000x the typical range) — 20x the typical `high`
    still leaves a long, realistic tail intact while keeping downstream
    percentile-style features from being dominated by a single freak draw.
    """
    if low <= 0:
        low = 0.01  # log-normal requires a strictly positive scale; guards a misconfigured 0ms floor
    median = math.sqrt(low * high)
    mu = math.log(median)
    sigma = math.log(high / low) / 3.0
    value = float(_rng.lognormal(mean=mu, sigma=sigma))
    return min(value, high * cap_multiplier)


def sample_poisson(lam: float) -> int:
    return int(_rng.poisson(lam=max(lam, 0.0)))


def sample_cpu(base: float, noise: float = 0.05) -> float:
    
    value = _rng.normal(loc=base, scale=noise)
    return float(np.clip(value, 0.0, 1.0))


def sample_memory(base_mb: float, noise_mb: float = 20.0) -> float:

    value = _rng.normal(loc=base_mb, scale=noise_mb)
    return float(max(value, 0.0))


def choose_weighted(choices: list[str], weights: list[float]) -> str:
   
    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    return str(_rng.choice(choices, p=weights_arr))


def uniform(low: float = 0.0, high: float = 1.0) -> float:
    return float(_rng.uniform(low, high))


def randint(low: int, high: int) -> int:
    return int(_rng.integers(low, high))
