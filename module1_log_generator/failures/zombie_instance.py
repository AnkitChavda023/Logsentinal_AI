from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import ZombieInstanceParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class ZombieInstanceFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = ZombieInstanceParams(
            **{k: v for k, v in raw.items()
               if k in ZombieInstanceParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < 10.0:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < 60.0:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < 120.0:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        zombie_id = self.params.zombie_instance_id or f"{entry.service}-pod-zombie"

        # Only the zombie instance is affected
        if entry.instance_id == zombie_id:
            phase = self.event.phase

            if phase == FailurePhase.DEGRADED:
                entry.log_level = "WARN"
                entry.message = "Health check failing on this instance"
                entry.is_anomaly = True
                entry.failure_type = "ZOMBIE_INSTANCE"

            elif phase in (FailurePhase.CRITICAL, FailurePhase.RECOVERING):
                entry.log_level = "ERROR"
                entry.message = "Instance not responding — zombie state"
                entry.http_status = 500
                entry.error_code = "ZOMBIE_INSTANCE"
                entry.latency_ms *= 10.0
                entry.is_anomaly = True
                entry.failure_type = "ZOMBIE_INSTANCE"

        return entry
