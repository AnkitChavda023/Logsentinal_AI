from __future__ import annotations

from module1_log_generator.config.defaults import (
    DEFAULT_CPU_BASE,
    DEFAULT_MEMORY_BASE_FRACTION,
)
from module1_log_generator.models.schema import ServiceConfig
from module1_log_generator.utils.random_utils import sample_cpu, sample_latency, sample_memory


class MetricSampler:

    def sample_latency(self, service: ServiceConfig) -> float:
        """[low, high] is the service's *typical* latency band — see
        random_utils.sample_latency's docstring for why the actual sampled
        value is log-normal, not bounded by [low, high] as a hard clip."""
        low, high = service.normal_latency_ms
        return sample_latency(low, high)

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
