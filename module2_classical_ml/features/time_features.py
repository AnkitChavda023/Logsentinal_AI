from __future__ import annotations

from datetime import datetime
from typing import Optional

from module2_classical_ml.config.defaults import (
    MAX_SECONDS_SINCE_LAST_LOG,
    NO_PRIOR_EVENT_SECONDS,
)


class TimeFeatureExtractor:

    def __init__(self) -> None:
        # service_name → last_error_time
        self._last_error: dict[str, datetime] = {}
        # service_name → last_log_time
        self._last_log: dict[str, datetime] = {}

    def extract(
        self,
        timestamp: datetime,
        service: str,
        is_error: bool,
    ) -> dict[str, float]:

        last_error = self._last_error.get(service)
        last_log = self._last_log.get(service)

        seconds_since_last_error = (
            (timestamp - last_error).total_seconds()
            if last_error is not None
            else NO_PRIOR_EVENT_SECONDS
        )
        seconds_since_last_log = (
            (timestamp - last_log).total_seconds()
            if last_log is not None
            else 0.0
        )

        # Update state
        if is_error:
            self._last_error[service] = timestamp
        self._last_log[service] = timestamp

        return {
            "hour_of_day": float(timestamp.hour),
            "minute_of_hour": float(timestamp.minute),
            "day_of_week": float(timestamp.weekday()),
            "seconds_since_last_error": min(seconds_since_last_error, NO_PRIOR_EVENT_SECONDS),
            "seconds_since_last_log": min(seconds_since_last_log, MAX_SECONDS_SINCE_LAST_LOG),
        }

    def reset(self) -> None:
        self._last_error.clear()
        self._last_log.clear()
