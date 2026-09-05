from __future__ import annotations

from typing import Optional

from module2_classical_ml.config.defaults import TRACE_STATE_MAX_ENTRIES


class TraceFeatureExtractor:

    def __init__(self) -> None:
        # trace_id → {has_error, depth, services_visited}
        self._traces: dict[str, dict] = {}

    def update(
        self,
        trace_id: str,
        service: str,
        is_error: bool,
        parent_span_id: Optional[str],
    ) -> None:
        if trace_id not in self._traces:
            self._traces[trace_id] = {
                "has_error": False,
                "depth": 0,
                "services": set(),
            }
        t = self._traces[trace_id]
        if is_error:
            t["has_error"] = True
        t["depth"] += 1
        t["services"].add(service)

    def extract(self, trace_id: str) -> dict[str, float]:
        t = self._traces.get(trace_id, {"has_error": False, "depth": 1, "services": set()})
        return {
            "trace_already_has_error": 1.0 if t["has_error"] else 0.0,
            "trace_depth": float(t["depth"]),
            "trace_services_visited": float(len(t["services"])),
        }

    def evict_old_traces(self, max_traces: int = TRACE_STATE_MAX_ENTRIES) -> None:
        if len(self._traces) > max_traces:
            keys = list(self._traces.keys())
            for k in keys[: max_traces // 2]:
                del self._traces[k]
