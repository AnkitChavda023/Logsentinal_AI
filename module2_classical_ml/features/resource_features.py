from __future__ import annotations

from collections import deque

from module2_classical_ml.config.defaults import RESOURCE_WINDOW_SECONDS


class ResourceFeatureExtractor:

    def __init__(self, window: int = RESOURCE_WINDOW_SECONDS) -> None:
        self._cpu: dict[str, deque] = {}
        self._mem: dict[str, deque] = {}
        self._window = window

    def extract(
        self, service: str, cpu_usage: float, memory_mb: float
    ) -> dict[str, float]:

        if service not in self._cpu:
            self._cpu[service] = deque(maxlen=self._window)
            self._mem[service] = deque(maxlen=self._window)

        cpu_q = self._cpu[service]
        mem_q = self._mem[service]

        cpu_avg = sum(cpu_q) / len(cpu_q) if cpu_q else cpu_usage
        cpu_vs_avg = cpu_usage / max(cpu_avg, 1e-6)

        # Memory trend: slope of memory over last window
        if len(mem_q) >= 2:
            diffs = [mem_q[i + 1] - mem_q[i] for i in range(len(mem_q) - 1)]
            mem_trend = sum(diffs) / len(diffs)
        else:
            mem_trend = 0.0

        cpu_q.append(cpu_usage)
        mem_q.append(memory_mb)

        return {
            "cpu_usage_raw": cpu_usage,
            "cpu_vs_rolling_avg": float(cpu_vs_avg),
            "memory_mb_raw": memory_mb,
            "memory_trend": float(mem_trend),
        }
