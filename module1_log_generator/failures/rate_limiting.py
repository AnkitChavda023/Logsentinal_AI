from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import RateLimitingParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class RateLimitingFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = RateLimitingParams(
            **{k: v for k, v in raw.items()
               if k in RateLimitingParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        p = self.params
        if elapsed_seconds < p.warn_after_seconds:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < p.failure_after_seconds:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < p.failure_after_seconds + 60:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        phase = self.event.phase
        limit = self.params.requests_per_second_limit

        if phase == FailurePhase.DEGRADED:
            entry.log_level = "WARN"
            entry.message = f"Rate limit approaching: {int(limit * 0.9)}/{limit} req/s"
            entry.is_anomaly = True
            entry.failure_type = "RATE_LIMITING"

        elif phase in (FailurePhase.CRITICAL, FailurePhase.RECOVERING):
            entry.log_level = "ERROR"
            entry.message = "Rate limit exceeded: HTTP 429"
            entry.http_status = 429
            entry.error_code = "RATE_LIMIT_EXCEEDED"
            entry.is_anomaly = True
            entry.failure_type = "RATE_LIMITING"

        return entry
