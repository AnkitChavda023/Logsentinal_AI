"""
FailureSchedule and FailureEvent models.

FailureSchedule — created at simulation init when randomizing
    what failures will happen during the run.

FailureEvent — a live/active failure instantiated from a
    FailureSchedule once the simulation clock reaches its start
    time. This is the object that gets .tick()-ed each second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class FailurePhase(str, Enum):
    """
    Standard multi-phase progression for all failure types.

    Per §5.3: failures never switch states instantly — they always
    progress through WARN → CRITICAL → FAILURE.
    """
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    RECOVERING = "RECOVERING"
    RESOLVED = "RESOLVED"


@dataclass
class FailureSchedule:
    """
    Represents a planned failure that will be injected at a specific
    simulation time.

    Created in bulk at simulation init (§5.2 step 1).
    """

    schedule_id: str
    """Unique ID for this schedule entry (e.g. 'SCHED-001')."""

    service_name: str
    """Which service this failure is injected into."""

    failure_type: str
    """One of the 12 FailureType enum member names."""

    start_time: datetime
    """Simulation wall-clock time at which to activate this failure."""

    duration_seconds: float
    """How long the failure lasts before resolving."""

    params: dict = field(default_factory=dict)
    """Failure-type-specific parameters (e.g. multipliers for latency spikes)."""


@dataclass
class FailureEvent:
    """
    A live, ticking failure instance derived from a FailureSchedule.

    failure_injector.py creates one FailureEvent per FailureSchedule
    when the simulation clock passes its start_time, and calls .tick()
    on it every simulated second.
    """

    event_id: str
    """Globally unique ID, written to the ground-truth manifest."""

    schedule: FailureSchedule

    phase: FailurePhase = FailurePhase.DEGRADED
    elapsed_seconds: float = 0.0

    resolved: bool = False
    end_time: Optional[datetime] = None

    # Downstream events spawned by a cascading failure
    spawned_events: list["FailureEvent"] = field(default_factory=list)

    # Track phase changes for manifest
    phase_transitions: list[tuple[str, str]] = field(default_factory=list)
