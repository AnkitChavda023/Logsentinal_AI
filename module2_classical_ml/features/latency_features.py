from __future__ import annotations

from collections import deque

from module2_classical_ml.config.defaults import (
    LATENCY_WINDOW_1HR_SECONDS,
    LATENCY_WINDOW_5MIN_SECONDS,
)


class LatencyFeatureExtractor:

    def __init__(
        self,
        window_5min_size: int = LATENCY_WINDOW_5MIN_SECONDS,
        window_1hr_size: int = LATENCY_WINDOW_1HR_SECONDS,
    ) -> None:
        self._5min: dict[str, deque] = {}
        self._1hr: dict[str, deque] = {}
        self._window_5min = window_5min_size
        self._window_1hr = window_1hr_size

    def extract(self, service: str, latency_ms: float) -> dict[str, float]:
        if service not in self._5min:
            self._5min[service] = deque(maxlen=self._window_5min)
            self._1hr[service] = deque(maxlen=self._window_1hr)

        q5 = self._5min[service]
        q1h = self._1hr[service]

        mean_5 = sum(q5) / len(q5) if q5 else latency_ms
        std_5 = _std(q5) if len(q5) > 1 else 1.0
        mean_1h = sum(q1h) / len(q1h) if q1h else latency_ms

        zscore = (latency_ms - mean_5) / max(std_5, 1e-6)
        vs_5min = latency_ms / max(mean_5, 1e-6)
        vs_1hr = latency_ms / max(mean_1h, 1e-6)

        # Update windows
        q5.append(latency_ms)
        q1h.append(latency_ms)

        return {
            "latency_ms_raw": latency_ms,
            "latency_zscore": float(zscore),
            "latency_vs_5min_avg": float(vs_5min),
            "latency_vs_1hr_avg": float(vs_1hr),
        }


def _std(values: deque) -> float:
    if len(values) < 2:
        return 1.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return variance ** 0.5
