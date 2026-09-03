from __future__ import annotations

from collections import deque
import math

from module2_classical_ml.config.defaults import (
    STATISTICAL_ROLLING_WINDOW_SIZE,
    STATISTICAL_SIGMA_CAP,
)


class StatisticalThresholdDetector:

    def __init__(self, window_size: int = STATISTICAL_ROLLING_WINDOW_SIZE) -> None:
        self._window = window_size
        # (service, metric) → deque of values
        self._history: dict[tuple[str, str], deque] = {}

    def score(
        self, service: str, features: dict[str, float]
    ) -> float:
        max_z = 0.0

        for metric, value in features.items():
            key = (service, metric)
            if key not in self._history:
                self._history[key] = deque(maxlen=self._window)

            q = self._history[key]
            if len(q) >= 2:
                mean = sum(q) / len(q)
                variance = sum((x - mean) ** 2 for x in q) / len(q)
                sigma = math.sqrt(variance) if variance > 0 else 1e-6
                z = abs(value - mean) / sigma
                max_z = max(max_z, z)

            q.append(value)

        return min(max_z / STATISTICAL_SIGMA_CAP, 1.0)

    def reset(self, service: str | None = None) -> None:
        if service is None:
            self._history.clear()
        else:
            keys_to_del = [k for k in self._history if k[0] == service]
            for k in keys_to_del:
                del self._history[k]
