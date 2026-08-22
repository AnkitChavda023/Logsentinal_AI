
from __future__ import annotations

import numpy as np


_rng: np.random.Generator = np.random.default_rng()


def seed_rng(seed: int) -> None:
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _rng


def sample_latency(mean: float, std: float, low: float, high: float) -> float:
    
    value = _rng.normal(loc=mean, scale=std)
    return float(np.clip(value, low, high))


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
