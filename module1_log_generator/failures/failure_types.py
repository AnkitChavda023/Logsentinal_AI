from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureType(str, Enum):
    LATENCY_SPIKE = "LATENCY_SPIKE"
    TIMEOUT = "TIMEOUT"
    CONNECTION_POOL_EXHAUSTION = "CONNECTION_POOL_EXHAUSTION"
    MEMORY_LEAK = "MEMORY_LEAK"
    CASCADING_FAILURE = "CASCADING_FAILURE"
    THUNDERING_HERD = "THUNDERING_HERD"
    NETWORK_PARTITION = "NETWORK_PARTITION"
    RATE_LIMITING = "RATE_LIMITING"
    DEADLOCK = "DEADLOCK"
    ZOMBIE_INSTANCE = "ZOMBIE_INSTANCE"
    CONFIGURATION_DRIFT = "CONFIGURATION_DRIFT"
    DEPENDENCY_API_BREAK = "DEPENDENCY_API_BREAK"



# Per-type parameter dataclasses



@dataclass
class LatencySpikeParams:
    multiplier_warn: float = 2.0
    multiplier_critical: float = 10.0
    warn_after_seconds: float = 10.0
    critical_after_seconds: float = 30.0
    failure_after_seconds: float = 60.0


@dataclass
class TimeoutParams:
    timeout_ms: float = 5_000.0
    failure_after_seconds: float = 15.0


@dataclass
class ConnectionPoolParams:
    pool_warn_pct: float = 85.0
    pool_critical_pct: float = 95.0
    warn_after_seconds: float = 20.0
    critical_after_seconds: float = 40.0
    failure_after_seconds: float = 60.0


@dataclass
class MemoryLeakParams:
    leak_rate_mb_per_sec: float = 5.0
    oom_threshold_mb: float = 3_000.0
    warn_fraction: float = 0.80
    critical_fraction: float = 0.95


@dataclass
class CascadingFailureParams:
    propagation_delay_seconds: float = 5.0
    affected_services: list[str] = field(default_factory=list)


@dataclass
class ThunderingHerdParams:
    restart_delay_seconds: float = 3.0


@dataclass
class NetworkPartitionParams:
    target_dependency: str = ""
    failure_after_seconds: float = 5.0


@dataclass
class RateLimitingParams:
    requests_per_second_limit: int = 100
    warn_after_seconds: float = 10.0
    failure_after_seconds: float = 20.0


@dataclass
class DeadlockParams:
    partner_service: str = ""
    timeout_seconds: float = 30.0


@dataclass
class ZombieInstanceParams:
    zombie_instance_id: str = ""


@dataclass
class ConfigurationDriftParams:
    drifted_instance_id: str = ""
    bad_config_key: str = "max_connections"


@dataclass
class DependencyApiBreakParams:
    broken_method: str = "GET /api/v2/resource"
    schema_field: str = "response.items"
