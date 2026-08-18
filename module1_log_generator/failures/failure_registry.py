
from __future__ import annotations

from typing import TYPE_CHECKING, Type

from module1_log_generator.failures.failure_types import FailureType

if TYPE_CHECKING:
    from module1_log_generator.failures.base_failure import BaseFailure
    from module1_log_generator.models.failure_models import FailureEvent


def get_failure_class(failure_type: FailureType) -> Type["BaseFailure"]:
    """
    Return the BaseFailure subclass that implements the given FailureType.

    Raises:
        KeyError: if the FailureType has no registered implementation.
    """
    # Import here to avoid circular imports at module load time
    from module1_log_generator.failures.latency_spike import LatencySpikeFailure
    from module1_log_generator.failures.timeout import TimeoutFailure
    from module1_log_generator.failures.connection_pool_exhaustion import (
        ConnectionPoolExhaustionFailure,
    )
    from module1_log_generator.failures.memory_leak import MemoryLeakFailure
    from module1_log_generator.failures.cascading_failure import CascadingFailure
    from module1_log_generator.failures.thundering_herd import ThunderingHerdFailure
    from module1_log_generator.failures.network_partition import NetworkPartitionFailure
    from module1_log_generator.failures.rate_limiting import RateLimitingFailure
    from module1_log_generator.failures.deadlock import DeadlockFailure
    from module1_log_generator.failures.zombie_instance import ZombieInstanceFailure
    from module1_log_generator.failures.configuration_drift import ConfigurationDriftFailure
    from module1_log_generator.failures.dependency_api_break import DependencyApiBreakFailure

    registry: dict[FailureType, Type["BaseFailure"]] = {
        FailureType.LATENCY_SPIKE: LatencySpikeFailure,
        FailureType.TIMEOUT: TimeoutFailure,
        FailureType.CONNECTION_POOL_EXHAUSTION: ConnectionPoolExhaustionFailure,
        FailureType.MEMORY_LEAK: MemoryLeakFailure,
        FailureType.CASCADING_FAILURE: CascadingFailure,
        FailureType.THUNDERING_HERD: ThunderingHerdFailure,
        FailureType.NETWORK_PARTITION: NetworkPartitionFailure,
        FailureType.RATE_LIMITING: RateLimitingFailure,
        FailureType.DEADLOCK: DeadlockFailure,
        FailureType.ZOMBIE_INSTANCE: ZombieInstanceFailure,
        FailureType.CONFIGURATION_DRIFT: ConfigurationDriftFailure,
        FailureType.DEPENDENCY_API_BREAK: DependencyApiBreakFailure,
    }
    return registry[failure_type]


def create_failure(event: "FailureEvent") -> "BaseFailure":
    """Instantiate a BaseFailure subclass from a FailureEvent."""
    cls = get_failure_class(FailureType(event.schedule.failure_type))
    return cls(event)
