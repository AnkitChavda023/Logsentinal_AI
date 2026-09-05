from __future__ import annotations

from collections import deque
from typing import Optional

from module2_classical_ml.config.defaults import ERROR_WINDOW_SECONDS


_ERROR_CODE_MAP: dict[str, int] = {
    "": 0,
    "DB_CONN_POOL_FULL": 1,
    "TIMEOUT": 2,
    "OOM_KILL": 3,
    "CONNECTION_REFUSED": 4,
    "NO_ROUTE_TO_HOST": 5,
    "RATE_LIMIT_EXCEEDED": 6,
    "DEADLOCK": 7,
    "DEADLOCK_TIMEOUT": 8,
    "ZOMBIE_INSTANCE": 9,
    "CONFIG_DRIFT": 10,
    "METHOD_NOT_FOUND": 11,
    "SCHEMA_VALIDATION_FAILED": 12,
    "UPSTREAM_FAILURE": 13,
    "CASCADING_FAILURE": 14,
    "THUNDERING_HERD": 15,
    "LATENCY_SPIKE_CRITICAL": 16,
    "MEMORY_LEAK": 17,
    "ERR_GENERIC": 18,
}


class ErrorFeatureExtractor:

    def __init__(self, window_size: int = ERROR_WINDOW_SECONDS) -> None:
        # Per-service rolling error flags (1 = error, 0 = not)
        self._errors: dict[str, deque] = {}
        self._window_size = window_size

    def extract(
        self,
        service: str,
        log_level: str,
        error_code: Optional[str],
    ) -> dict[str, float]:

        if service not in self._errors:
            self._errors[service] = deque(maxlen=self._window_size)

        is_error = 1.0 if log_level in ("ERROR", "FATAL") else 0.0

        q = self._errors[service]
        error_count_60s = float(sum(q))
        error_rate_60s = error_count_60s / max(len(q), 1)

        q.append(is_error)

        code_str = (error_code or "").upper()
        encoded = _ERROR_CODE_MAP.get(code_str, len(_ERROR_CODE_MAP))

        return {
            "is_error": is_error,
            "error_code_encoded": float(encoded),
            "error_count_60s": error_count_60s,
            "error_rate_60s": error_rate_60s,
        }
