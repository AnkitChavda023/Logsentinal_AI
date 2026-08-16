
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class FailurePhase(str, Enum):

    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    RECOVERING = "RECOVERING"
    RESOLVED = "RESOLVED"


@dataclass
class FailureSchedule:

    schedule_id: str

    service_name: str

    failure_type: str

    start_time: datetime

    duration_seconds: float

    params: dict = field(default_factory=dict)


@dataclass
class FailureEvent:


    event_id: str
    schedule: FailureSchedule

    phase: FailurePhase = FailurePhase.DEGRADED
    elapsed_seconds: float = 0.0

    resolved: bool = False
    end_time: Optional[datetime] = None

    spawned_events: list["FailureEvent"] = field(default_factory=list)

    phase_transitions: list[tuple[str, str]] = field(default_factory=list)
