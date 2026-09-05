from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

from module2_classical_ml.config.defaults import (
    ROLLING_WINDOW_MIN_HISTORY_SAMPLES,
    ROLLING_WINDOW_MINUTES,
    ROLLING_WINDOW_NEUTRAL_SCORE,
)


class RollingWindowDetector:

    def __init__(self, window_minutes: int = ROLLING_WINDOW_MINUTES) -> None:
        self._history: dict[str, dict[tuple[int, int], list[float]]] = {}
        self._current: dict[str, deque] = {}
        self._window_minutes = window_minutes

    def record(self, service: str, anomaly_signal: float) -> None:
        if service not in self._current:
            self._current[service] = deque(maxlen=60 * self._window_minutes)
        self._current[service].append(anomaly_signal)

    def commit_window(self, service: str, sim_time: datetime) -> None:
        if service not in self._history:
            self._history[service] = {}

        bucket = (sim_time.hour, sim_time.minute)
        current_mean = (
            np.mean(list(self._current.get(service, [])))
            if self._current.get(service)
            else 0.0
        )

        if bucket not in self._history[service]:
            self._history[service][bucket] = []
        self._history[service][bucket].append(float(current_mean))

    def clear_current(self, service: str) -> None:
        if service in self._current:
            self._current[service].clear()

    def score(self, service: str, sim_time: datetime) -> float:
        hist = self._history.get(service, {})
        bucket = (sim_time.hour, sim_time.minute)
        past_windows = hist.get(bucket, [])

        if len(past_windows) < ROLLING_WINDOW_MIN_HISTORY_SAMPLES:
            return ROLLING_WINDOW_NEUTRAL_SCORE  # Not enough history

        current = np.mean(list(self._current.get(service, [0.0])))
        pct_rank = float(np.mean(np.array(past_windows) <= current))
        return float(np.clip(pct_rank, 0.0, 1.0))
