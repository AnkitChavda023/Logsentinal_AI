from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import NetworkPartitionParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class NetworkPartitionFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = NetworkPartitionParams(
            **{k: v for k, v in raw.items()
               if k in NetworkPartitionParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < 5.0:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < self.params.failure_after_seconds:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < self.params.failure_after_seconds + 60:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        phase = self.event.phase
        dep = self.params.target_dependency or "dependency"

        if phase == FailurePhase.DEGRADED:
            entry.log_level = "WARN"
            entry.message = f"Intermittent connectivity to {dep}"
            entry.is_anomaly = True
            entry.failure_type = "NETWORK_PARTITION"

        elif phase == FailurePhase.CRITICAL:
            entry.log_level = "ERROR"
            entry.message = f"Connection refused to {dep}"
            entry.http_status = 503
            entry.error_code = "CONNECTION_REFUSED"
            entry.is_anomaly = True
            entry.failure_type = "NETWORK_PARTITION"

        elif phase == FailurePhase.RECOVERING:
            entry.log_level = "ERROR"
            entry.message = f"No route to host {dep}"
            entry.http_status = 503
            entry.error_code = "NO_ROUTE_TO_HOST"
            entry.is_anomaly = True
            entry.failure_type = "NETWORK_PARTITION"

        return entry
