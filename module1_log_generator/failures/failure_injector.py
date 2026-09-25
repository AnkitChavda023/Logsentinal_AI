from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from module1_log_generator.config import defaults as cfg
from module1_log_generator.failures.failure_registry import create_failure
from module1_log_generator.failures.failure_types import FailureType
from module1_log_generator.writers.manifest_writer import ManifestWriter
from module1_log_generator.models.failure_models import (
    FailureEvent,
    FailurePhase,
    FailureSchedule,
)
from module1_log_generator.models.log_entry import LogEntry
from module1_log_generator.utils.constants import FAILURE_SEVERITY, MANIFEST_HEADER
from module1_log_generator.utils.ids import generate_span_id

logger = logging.getLogger(__name__)

_PRECURSOR_EMPHASIS: dict[str, tuple[float, float, float]] = {
    "LATENCY_SPIKE": (1.0, 0.3, 0.1),
    "TIMEOUT": (1.0, 0.3, 0.1),
    "CONNECTION_POOL_EXHAUSTION": (0.4, 1.0, 0.2),
    "MEMORY_LEAK": (0.2, 0.3, 1.0),
    "THUNDERING_HERD": (0.5, 1.0, 0.3),
    "RATE_LIMITING": (0.3, 0.8, 0.1),
    "ZOMBIE_INSTANCE": (0.3, 0.7, 0.6),
    "DEADLOCK": (0.6, 0.4, 0.2),
    "CASCADING_FAILURE": (0.5, 0.5, 0.3),
    "NETWORK_PARTITION": (0.7, 0.2, 0.1),
    "CONFIGURATION_DRIFT": (0.3, 0.3, 0.3),
    "DEPENDENCY_API_BREAK": (0.6, 0.2, 0.1),
}
_PRECURSOR_EMPHASIS_DEFAULT = (0.5, 0.5, 0.5)


class FailureInjector:
    """
    Owns the list of FailureSchedules and all active FailureEvents.

    Must be called every simulated second via .tick().
    """

    def __init__(
        self,
        schedules: list[FailureSchedule],
        service_graph,  # core.service_graph.ServiceGraph
        manifest_path: Path,
    ) -> None:
        self._schedules = list(schedules)
        self._graph = service_graph
        self._active: list[tuple[FailureEvent, object]] = []
        # (FailureEvent, BaseFailure instance)
        self._manifest_path = manifest_path
        self._manifest_writer = ManifestWriter(manifest_path)
        self._event_counter = 0
        self._written_events: set[str] = set()
        self._last_sim_time: Optional[datetime] = None
        self._next_onset: dict[str, tuple[datetime, str]] = {}

    
    def open_manifest(self) -> None:
        """Open the ground-truth manifest CSV for writing."""
        self._manifest_writer.open()

    def close_manifest(self) -> None:
        
        for event, _ in self._active:
            self._write_manifest_row(event, self._last_sim_time or event.schedule.start_time)
        self._manifest_writer.close()

    def tick(self, sim_time: datetime) -> None:
        
        self._last_sim_time = sim_time

        still_pending = []
        next_onset: dict[str, tuple[datetime, str]] = {}
        for schedule in self._schedules:
            if sim_time >= schedule.start_time:
                self._activate(schedule, sim_time)
            else:
                still_pending.append(schedule)
                current = next_onset.get(schedule.service_name)
                if current is None or schedule.start_time < current[0]:
                    next_onset[schedule.service_name] = (schedule.start_time, schedule.failure_type)
        self._schedules = still_pending
        self._next_onset = next_onset

        still_active = []
        for event, failure_obj in self._active:
            elapsed = (sim_time - event.schedule.start_time).total_seconds()
            old_phase = event.phase
            was_resolved_before_tick = event.resolved
            new_phase = failure_obj.tick(elapsed)

            if new_phase != old_phase:
                event.phase_transitions.append((sim_time.isoformat(), new_phase.value))

            event.elapsed_seconds = elapsed

            if new_phase == FailurePhase.RESOLVED and not was_resolved_before_tick:
                event.resolved = True
                event.end_time = sim_time
                self._write_manifest_row(event, sim_time)
                logger.debug("Failure resolved: %s on %s", event.event_id, event.schedule.service_name)
            elif not event.resolved:
                still_active.append((event, failure_obj))

            if (
                event.schedule.failure_type == FailureType.CASCADING_FAILURE
                and new_phase == FailurePhase.CRITICAL
                and not event.spawned_events
            ):
                self._cascade(event, sim_time)

        self._active = still_active

    def apply_to_entry(self, entry: LogEntry) -> LogEntry:
        
        for event, failure_obj in self._active:
            if event.schedule.service_name == entry.service:
                entry = failure_obj.apply(entry, event.elapsed_seconds)
        return self._apply_precursor(entry)

    def _apply_precursor(self, entry: LogEntry) -> LogEntry:
        
        onset = self._next_onset.get(entry.service)
        if onset is None or self._last_sim_time is None:
            return entry

        start_time, failure_type = onset
        seconds_until_onset = (start_time - self._last_sim_time).total_seconds()
        if not (0 <= seconds_until_onset <= cfg.PRECURSOR_SECONDS):
            return entry

        progress = 1.0 - (seconds_until_onset / cfg.PRECURSOR_SECONDS)
        lat_w, cpu_w, mem_w = _PRECURSOR_EMPHASIS.get(failure_type, _PRECURSOR_EMPHASIS_DEFAULT)
        entry.latency_ms *= 1.0 + progress * (cfg.PRECURSOR_MAX_LATENCY_MULTIPLIER - 1.0) * lat_w
        entry.cpu_usage = min(1.0, entry.cpu_usage * (1.0 + progress * (cfg.PRECURSOR_MAX_CPU_MULTIPLIER - 1.0) * cpu_w))
        entry.memory_mb *= 1.0 + progress * (cfg.PRECURSOR_MAX_MEMORY_MULTIPLIER - 1.0) * mem_w
        return entry

    def has_active_failure(self, service_name: str) -> bool:
        return any(
            ev.schedule.service_name == service_name
            for ev, _ in self._active
        )

    def get_active_failures(self) -> list[FailureEvent]:
        return [ev for ev, _ in self._active]

    
    def _activate(self, schedule: FailureSchedule, sim_time: datetime) -> None:
        self._event_counter += 1
        event_id = f"F{self._event_counter:04d}"
        event = FailureEvent(
            event_id=event_id,
            schedule=schedule,
            phase=FailurePhase.DEGRADED,
            elapsed_seconds=0.0,
        )
        event.phase_transitions.append((sim_time.isoformat(), FailurePhase.DEGRADED.value))
        failure_obj = create_failure(event)
        self._active.append((event, failure_obj))
        logger.info(
            "Failure activated: %s (%s) on %s at %s",
            event_id,
            schedule.failure_type,
            schedule.service_name,
            sim_time.isoformat(),
        )

    def _cascade(self, source_event: FailureEvent, sim_time: datetime) -> None:
        dependents = self._graph.get_dependents(source_event.schedule.service_name)
        delay = source_event.schedule.params.get("propagation_delay_seconds", 5.0)

        for dep_service in dependents:
            self._event_counter += 1
            cascade_schedule = FailureSchedule(
                schedule_id=f"CASCADE-{self._event_counter:04d}",
                service_name=dep_service,
                failure_type=FailureType.CASCADING_FAILURE.value,
                start_time=sim_time,
                duration_seconds=60.0,
                params={"propagation_delay_seconds": delay},
            )
            child_event = FailureEvent(
                event_id=f"F{self._event_counter:04d}",
                schedule=cascade_schedule,
                phase=FailurePhase.DEGRADED,
            )
            child_event.phase_transitions.append((sim_time.isoformat(), FailurePhase.DEGRADED.value))
            child_failure = create_failure(child_event)
            self._active.append((child_event, child_failure))
            source_event.spawned_events.append(child_event)
            logger.info(
                "Cascading failure propagated to %s", dep_service
            )

    def _write_manifest_row(self, event: FailureEvent, sim_time: datetime) -> None:
        if self._manifest_writer is None:
            return
        if event.event_id in self._written_events:
            return
        self._written_events.add(event.event_id)

        failure_type = event.schedule.failure_type
        severity = FAILURE_SEVERITY.get(failure_type, "MEDIUM")

        self._manifest_writer.write_row(
            {
                "failure_id": event.event_id,
                "start_timestamp": event.schedule.start_time.isoformat(),
                "end_timestamp": sim_time.isoformat() if event.end_time is None else event.end_time.isoformat(),
                "affected_service": event.schedule.service_name,
                "failure_type": failure_type,
                "root_cause_service": event.schedule.service_name,
                "severity": severity,
                "phase_transitions": event.phase_transitions,
                "resolution_time": sim_time.isoformat() if event.end_time is None else event.end_time.isoformat(),
            }
        )
