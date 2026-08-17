

from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import CascadingFailureParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class CascadingFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = CascadingFailureParams(
            **{k: v for k, v in raw.items()
               if k in CascadingFailureParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < 5.0:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < 30.0:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < 90.0:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        phase = self.event.phase

        if phase == FailurePhase.DEGRADED:
            entry.log_level = "WARN"
            entry.message = f"Upstream dependency unhealthy"
            entry.is_anomaly = True
            entry.failure_type = "CASCADING_FAILURE"

        elif phase == FailurePhase.CRITICAL:
            entry.log_level = "ERROR"
            entry.message = f"Cascading failure from upstream service"
            entry.http_status = 503
            entry.error_code = "UPSTREAM_FAILURE"
            entry.is_anomaly = True
            entry.failure_type = "CASCADING_FAILURE"
            entry.latency_ms *= 4.0

        elif phase == FailurePhase.RECOVERING:
            entry.log_level = "ERROR"
            entry.message = "Service unavailable due to dependency failure"
            entry.http_status = 503
            entry.error_code = "CASCADING_FAILURE"
            entry.is_anomaly = True
            entry.failure_type = "CASCADING_FAILURE"

        return entry
