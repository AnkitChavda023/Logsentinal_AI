from __future__ import annotations

from module1_log_generator.config.defaults import (
    DEFAULT_CPU_BASE,
    DEFAULT_MEMORY_BASE_FRACTION,
)
from module1_log_generator.models.schema import ServiceConfig
from module1_log_generator.utils.random_utils import sample_cpu, sample_latency, sample_memory


class MetricSampler:

    def sample_latency(self, service: ServiceConfig) -> float:

        low, high = service.normal_latency_ms
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # ±3σ within [low, high]
        return sample_latency(mean, std, low, high)

    def sample_cpu(self, service: ServiceConfig) -> float:
    
        low, high = service.normal_latency_ms
        latency_fraction = min((high / 1000.0) * 0.5, 0.8)
        base = max(DEFAULT_CPU_BASE, latency_fraction * 0.3)
        return sample_cpu(base, noise=0.05)

    def sample_memory(self, service: ServiceConfig) -> float:
        
        ceiling = service.resource_limits.memory_mb
        base_mb = ceiling * DEFAULT_MEMORY_BASE_FRACTION
        noise_mb = ceiling * 0.05
        return sample_memory(base_mb, noise_mb)
