
from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import ConfigurationDriftParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class ConfigurationDriftFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = ConfigurationDriftParams(
            **{k: v for k, v in raw.items()
               if k in ConfigurationDriftParams.__dataclass_fields__}
        )

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        if elapsed_seconds < 15.0:
            phase = FailurePhase.DEGRADED
        elif elapsed_seconds < 60.0:
            phase = FailurePhase.CRITICAL
        elif elapsed_seconds < 180.0:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        drifted_id = (
            self.params.drifted_instance_id or f"{entry.service}-pod-1"
        )

        if entry.instance_id == drifted_id:
            phase = self.event.phase
            key = self.params.bad_config_key

            if phase == FailurePhase.DEGRADED:
                entry.log_level = "WARN"
                entry.message = f"Config value for '{key}' differs from cluster default"
                entry.is_anomaly = True
                entry.failure_type = "CONFIGURATION_DRIFT"

            elif phase in (FailurePhase.CRITICAL, FailurePhase.RECOVERING):
                entry.log_level = "ERROR"
                entry.message = f"Invalid config '{key}' causing service errors"
                entry.http_status = 500
                entry.error_code = "CONFIG_DRIFT"
                entry.is_anomaly = True
                entry.failure_type = "CONFIGURATION_DRIFT"

        return entry
