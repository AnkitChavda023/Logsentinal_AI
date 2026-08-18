
from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import TimeoutParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class TimeoutFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = TimeoutParams(**{k: v for k, v in raw.items()
                                       if k in TimeoutParams.__dataclass_fields__})

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < 5.0:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < self.params.failure_after_seconds:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < self.params.failure_after_seconds + 30:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        phase = self.event.phase

        if phase == FailurePhase.DEGRADED:
            entry.latency_ms = self.params.timeout_ms * 0.7
            entry.log_level = "WARN"
            entry.message = f"Slow response: {entry.latency_ms:.0f}ms"
            entry.is_anomaly = True
            entry.failure_type = "TIMEOUT"

        elif phase in (FailurePhase.CRITICAL, FailurePhase.RECOVERING):
            entry.latency_ms = self.params.timeout_ms
            entry.log_level = "ERROR"
            entry.message = f"timeout after {self.params.timeout_ms:.0f}ms"
            entry.http_status = 504
            entry.error_code = "TIMEOUT"
            entry.is_anomaly = True
            entry.failure_type = "TIMEOUT"

        return entry
