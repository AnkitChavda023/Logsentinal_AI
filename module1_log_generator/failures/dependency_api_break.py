from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import DependencyApiBreakParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class DependencyApiBreakFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = DependencyApiBreakParams(
            **{k: v for k, v in raw.items()
               if k in DependencyApiBreakParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < 5.0:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < 30.0:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < 120.0:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        phase = self.event.phase
        method = self.params.broken_method
        field = self.params.schema_field

        if phase == FailurePhase.DEGRADED:
            entry.log_level = "WARN"
            entry.message = f"API contract mismatch detected for {method}"
            entry.is_anomaly = True
            entry.failure_type = "DEPENDENCY_API_BREAK"

        elif phase == FailurePhase.CRITICAL:
            entry.log_level = "ERROR"
            entry.message = f"Method not found: {method}"
            entry.http_status = 404
            entry.error_code = "METHOD_NOT_FOUND"
            entry.is_anomaly = True
            entry.failure_type = "DEPENDENCY_API_BREAK"

        elif phase == FailurePhase.RECOVERING:
            entry.log_level = "ERROR"
            entry.message = f"Schema validation failed for {field}"
            entry.http_status = 422
            entry.error_code = "SCHEMA_VALIDATION_FAILED"
            entry.is_anomaly = True
            entry.failure_type = "DEPENDENCY_API_BREAK"

        return entry
