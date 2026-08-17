from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def open_manifest(self) -> None:
        """Open the ground-truth manifest CSV for writing."""
        self._manifest_writer.open()

    def close_manifest(self) -> None:
        """
        Write manifest rows for any failures still active/unresolved
        when the simulation ends, then flush and close the manifest CSV.

        Without this, a failure whose duration extends past the end of
        the generation window would have log entries in the dataset but
        no corresponding ground-truth row — silently corrupting labels
        for Module 3 training.
        """
        for event, _ in self._active:
            self._write_manifest_row(event, self._last_sim_time or event.schedule.start_time)
        self._manifest_writer.close()

    def tick(self, sim_time: datetime) -> None:
        """
        Advance all active failures by one second.

        Args:
            sim_time: Current simulation wall-clock time.
        """
        self._last_sim_time = sim_time

        # Activate any schedules whose start_time has arrived
        still_pending = []
        for schedule in self._schedules:
            if sim_time >= schedule.start_time:
                self._activate(schedule, sim_time)
            else:
                still_pending.append(schedule)
        self._schedules = still_pending

        # Tick active events and remove resolved ones
        still_active = []
        for event, failure_obj in self._active:
            elapsed = (sim_time - event.schedule.start_time).total_seconds()
            old_phase = event.phase
            # Captured BEFORE failure_obj.tick(): every concrete failure class
            # (latency_spike.py, timeout.py, ...) sets event.resolved = True
            # itself the instant it computes RESOLVED, one line below. Reading
            # event.resolved *after* that call made this check permanently
            # false (already-true "not True"), so a resolved failure's
            # manifest row was never written and it was quietly dropped from
            # _active without ever appearing in the ground-truth CSV.
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

            # Cascade: for cascading failures, propagate to dependents
            if (
                event.schedule.failure_type == FailureType.CASCADING_FAILURE
                and new_phase == FailurePhase.CRITICAL
                and not event.spawned_events
            ):
                self._cascade(event, sim_time)

        self._active = still_active

    def apply_to_entry(self, entry: LogEntry) -> LogEntry:
        """
        Apply any currently active failure affecting entry.service
        to the given LogEntry.

        Returns the (possibly mutated) entry.
        """
        for event, failure_obj in self._active:
            if event.schedule.service_name == entry.service:
                entry = failure_obj.apply(entry, event.elapsed_seconds)
        return entry

    def has_active_failure(self, service_name: str) -> bool:
        """Return True if there is any active failure on service_name."""
        return any(
            ev.schedule.service_name == service_name
            for ev, _ in self._active
        )

    def get_active_failures(self) -> list[FailureEvent]:
        """Return all currently active FailureEvent objects."""
        return [ev for ev, _ in self._active]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        """Propagate a cascading failure to all dependent services."""
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
        """Write one row to the ground-truth manifest CSV."""
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
