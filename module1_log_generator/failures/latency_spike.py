from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import LatencySpikeParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class LatencySpikeFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = LatencySpikeParams(**{k: v for k, v in raw.items()
                                             if k in LatencySpikeParams.__dataclass_fields__})

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
        base_latency = entry.latency_ms

        if phase == FailurePhase.DEGRADED:
            entry.latency_ms = base_latency * self.params.multiplier_warn
            entry.log_level = "WARN"
            entry.message = (
                f"Latency elevated: {entry.latency_ms:.1f}ms "
                f"(threshold {base_latency * 1.5:.0f}ms)"
            )
            entry.is_anomaly = True
            entry.failure_type = "LATENCY_SPIKE"

        elif phase == FailurePhase.CRITICAL:
            entry.latency_ms = base_latency * self.params.multiplier_critical
            entry.log_level = "ERROR"
            entry.message = f"Latency spike: {entry.latency_ms:.1f}ms"
            entry.http_status = 503
            entry.error_code = "LATENCY_SPIKE_CRITICAL"
            entry.is_anomaly = True
            entry.failure_type = "LATENCY_SPIKE"

        elif phase == FailurePhase.RECOVERING:
            entry.latency_ms = self.params.timeout_ms if hasattr(self.params, "timeout_ms") else base_latency * 20
            entry.log_level = "ERROR"
            entry.message = f"Timeout after {entry.latency_ms:.0f}ms"
            entry.http_status = 504
            entry.error_code = "TIMEOUT"
            entry.is_anomaly = True
            entry.failure_type = "LATENCY_SPIKE"

        entry.cpu_usage = min(entry.cpu_usage * 1.3, 1.0)
        return entry
