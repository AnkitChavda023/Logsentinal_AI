from __future__ import annotations

from abc import ABC, abstractmethod

from module1_log_generator.models.failure_models import FailureEvent, FailurePhase
from module1_log_generator.models.log_entry import LogEntry


class BaseFailure(ABC):

    def __init__(self, event: FailureEvent) -> None:
        self.event = event

    @abstractmethod
    def tick(self, elapsed_seconds: float) -> FailurePhase:


    @abstractmethod
    def apply(self, entry: LogEntry, elapsed_seconds: float) -> LogEntry:

    def is_resolved(self) -> bool:
        return self.event.resolved