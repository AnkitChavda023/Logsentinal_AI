

from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import ConnectionPoolParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class ConnectionPoolExhaustionFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = ConnectionPoolParams(**{k: v for k, v in raw.items()
                                               if k in ConnectionPoolParams.__dataclass_fields__})

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        p = self.params
        if elapsed_seconds < p.warn_after_seconds:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < p.critical_after_seconds:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < p.failure_after_seconds:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        phase = self.event.phase
        p = self.params

        # Linearly increase pool utilization over elapsed time
        max_elapsed = p.failure_after_seconds
        pool_pct = min(100.0, (elapsed_seconds / max_elapsed) * 100.0)

        if phase == FailurePhase.DEGRADED:
            entry.log_level = "WARN"
            entry.message = f"pool at {pool_pct:.0f}%"
            entry.is_anomaly = True
            entry.failure_type = "CONNECTION_POOL_EXHAUSTION"

        elif phase == FailurePhase.CRITICAL:
            entry.log_level = "ERROR"
            entry.message = "pool exhausted"
            entry.http_status = 500
            entry.error_code = "DB_CONN_POOL_FULL"
            entry.is_anomaly = True
            entry.failure_type = "CONNECTION_POOL_EXHAUSTION"
            entry.latency_ms *= 3.0

        elif phase == FailurePhase.RECOVERING:
            entry.log_level = "ERROR"
            entry.message = "Connection pool exhausted"
            entry.http_status = 500
            entry.error_code = "DB_CONN_POOL_FULL"
            entry.is_anomaly = True
            entry.failure_type = "CONNECTION_POOL_EXHAUSTION"
            entry.cpu_usage = min(entry.cpu_usage + 0.3, 1.0)
            entry.latency_ms *= 5.0

        return entry
