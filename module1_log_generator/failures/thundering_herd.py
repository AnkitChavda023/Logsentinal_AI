
from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import ThunderingHerdParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class ThunderingHerdFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = ThunderingHerdParams(
            **{k: v for k, v in raw.items()
               if k in ThunderingHerdParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < self.params.restart_delay_seconds:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < 20.0:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < 60.0:
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
            entry.message = "Restart imminent — health check failing"
            entry.is_anomaly = True
            entry.failure_type = "THUNDERING_HERD"

        elif phase == FailurePhase.CRITICAL:
            entry.log_level = "WARN"
            entry.message = f"Starting up — Listening on port 8080"
            entry.is_anomaly = True
            entry.failure_type = "THUNDERING_HERD"
            entry.latency_ms = 5000.0  # slow startup

        elif phase == FailurePhase.RECOVERING:
            entry.log_level = "ERROR"
            entry.message = "Service unavailable during restart storm"
            entry.http_status = 503
            entry.error_code = "THUNDERING_HERD"
            entry.is_anomaly = True
            entry.failure_type = "THUNDERING_HERD"

        return entry
