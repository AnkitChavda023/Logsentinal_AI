from __future__ import annotations

from module1_log_generator.failures.base_failure import BaseFailure
from module1_log_generator.failures.failure_types import MemoryLeakParams
from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class MemoryLeakFailure(BaseFailure):
    def __init__(self, event: FailureEvent) -> None:
        super().__init__(event)
        raw = event.schedule.params or {}
        self.params = MemoryLeakParams(**{k: v for k, v in raw.items()
                                          if k in MemoryLeakParams.__dataclass_fields__})

    def tick(self, elapsed_seconds: float) -> FailurePhase:
        p = self.params
        projected_memory = elapsed_seconds * p.leak_rate_mb_per_sec
        fraction = projected_memory / p.oom_threshold_mb

        if fraction < p.warn_fraction:
            phase = FailurePhase.DEGRADED
        elif fraction < p.critical_fraction:
            phase = FailurePhase.CRITICAL
        elif fraction < 1.0:
            phase = FailurePhase.RECOVERING
        else:
            phase = FailurePhase.RESOLVED
            self.event.resolved = True

        self.event.phase = phase
        return phase

    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:
        p = self.params
        leaked = elapsed_seconds * p.leak_rate_mb_per_sec
        entry.memory_mb = entry.memory_mb + leaked
        phase = self.event.phase

        if phase == FailurePhase.DEGRADED:
            pct = (entry.memory_mb / p.oom_threshold_mb) * 100
            entry.log_level = "WARN"
            entry.message = f"Memory usage at {pct:.0f}% of limit"
            entry.is_anomaly = True
            entry.failure_type = "MEMORY_LEAK"

        elif phase == FailurePhase.CRITICAL:
            entry.log_level = "ERROR"
            entry.message = f"Memory leak detected: {entry.memory_mb:.0f} MB"
            entry.error_code = "MEMORY_LEAK"
            entry.is_anomaly = True
            entry.failure_type = "MEMORY_LEAK"

        elif phase == FailurePhase.RECOVERING:
            entry.memory_mb = p.oom_threshold_mb
            entry.log_level = "FATAL"
            entry.message = "Out of memory — process terminating"
            entry.error_code = "OOM_KILL"
            entry.http_status = 503
            entry.is_anomaly = True
            entry.failure_type = "MEMORY_LEAK"

        return entry
