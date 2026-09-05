from __future__ import annotations

from collections import deque

from module2_classical_ml.config.defaults import (
    VOLUME_BASELINE_WINDOW_SECONDS,
    VOLUME_SPIKE_RATIO_THRESHOLD,
)


class VolumeFeatureExtractor:

    def __init__(
        self,
        baseline_window: int = VOLUME_BASELINE_WINDOW_SECONDS,
        volume_spike_threshold: float = VOLUME_SPIKE_RATIO_THRESHOLD,
    ) -> None:
        self._counts: dict[str, deque] = {}
        self._current_second_count: dict[str, int] = {}
        self._baseline_window = baseline_window
        self._volume_spike_threshold = volume_spike_threshold

    def record(self, service: str) -> None:
        if service not in self._current_second_count:
            self._current_second_count[service] = 0
        self._current_second_count[service] += 1

    def tick_second(self, service: str) -> None:
        if service not in self._counts:
            self._counts[service] = deque(maxlen=self._baseline_window)
        count = self._current_second_count.pop(service, 0)
        self._counts[service].append(float(count))

    def extract(self, service: str) -> dict[str, float]:
        if service not in self._counts:
            self._counts[service] = deque(maxlen=self._baseline_window)

        q = self._counts[service]
        current = float(self._current_second_count.get(service, 0))
        baseline = sum(q) / len(q) if q else current

        ratio = current / max(baseline, 1.0)
        is_spike = 1.0 if ratio >= self._volume_spike_threshold else 0.0

        return {
            "logs_per_second_current": current,
            "volume_vs_baseline_ratio": ratio,
            "is_volume_spike": is_spike,
        }