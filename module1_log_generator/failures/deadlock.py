
from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import DeadlockParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class DeadlockFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = DeadlockParams(
            **{k: v for k, v in raw.items()
               if k in DeadlockParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < 5.0:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < self.params.timeout_seconds:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < self.params.timeout_seconds + 30:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        phase = self.event.phase
        partner = self.params.partner_service or "partner-service"

        if phase == FailurePhase.DEGRADED:
            entry.log_level = "WARN"
            entry.message = f"Waiting on lock held by {partner}"
            entry.latency_ms *= 3.0
            entry.is_anomaly = True
            entry.failure_type = "DEADLOCK"

        elif phase == FailurePhase.CRITICAL:
            entry.log_level = "ERROR"
            entry.message = f"Deadlock detected — both services stuck"
            entry.latency_ms = self.params.timeout_seconds * 1000
            entry.error_code = "DEADLOCK"
            entry.is_anomaly = True
            entry.failure_type = "DEADLOCK"

        elif phase == FailurePhase.RECOVERING:
            entry.log_level = "FATAL"
            entry.message = "Deadlock detected — terminating transaction"
            entry.http_status = 500
            entry.error_code = "DEADLOCK_TIMEOUT"
            entry.is_anomaly = True
            entry.failure_type = "DEADLOCK"

        return entry
