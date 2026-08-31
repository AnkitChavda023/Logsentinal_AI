from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from module2_classical_ml.features.time_features import TimeFeatureExtractor
from module2_classical_ml.features.latency_features import LatencyFeatureExtractor
from module2_classical_ml.features.error_features import ErrorFeatureExtractor
from module2_classical_ml.features.volume_features import VolumeFeatureExtractor
from module2_classical_ml.features.resource_features import ResourceFeatureExtractor
from module2_classical_ml.features.trace_features import TraceFeatureExtractor
from module2_classical_ml.features.topology_features import TopologyFeatureExtractor

FEATURE_NAMES = [
    # Group A — Time (5)
    "hour_of_day", "minute_of_hour", "day_of_week",
    "seconds_since_last_error", "seconds_since_last_log",
    # Group B — Latency (4)
    "latency_ms_raw", "latency_zscore",
    "latency_vs_5min_avg", "latency_vs_1hr_avg",
    # Group C — Error (4)
    "is_error", "error_code_encoded",
    "error_count_60s", "error_rate_60s",
    # Group D — Volume (3)
    "logs_per_second_current", "volume_vs_baseline_ratio", "is_volume_spike",
    # Group E — Resource (4)
    "cpu_usage_raw", "cpu_vs_rolling_avg",
    "memory_mb_raw", "memory_trend",
    # Group F — Trace (3)
    "trace_already_has_error", "trace_depth", "trace_services_visited",
    # Group G — Topology (3)
    "upstream_service_count", "downstream_service_count", "upstream_health_score",
]

FEATURE_DIM = len(FEATURE_NAMES)  # 26


class FeaturePipeline:

    def __init__(self, adjacency: dict[str, list[str]]) -> None:
        self._time = TimeFeatureExtractor()
        self._latency = LatencyFeatureExtractor()
        self._error = ErrorFeatureExtractor()
        self._volume = VolumeFeatureExtractor()
        self._resource = ResourceFeatureExtractor()
        self._trace = TraceFeatureExtractor()
        self._topology = TopologyFeatureExtractor(adjacency)

    def extract(
        self,
        timestamp: datetime,
        service: str,
        instance_id: str,
        trace_id: str,
        parent_span_id: Optional[str],
        log_level: str,
        latency_ms: float,
        cpu_usage: float,
        memory_mb: float,
        error_code: Optional[str],
    ) -> np.ndarray:
        is_error_bool = log_level in ("ERROR", "FATAL")

        feat_a = self._time.extract(timestamp, service, is_error_bool)
        feat_b = self._latency.extract(service, latency_ms)
        feat_c = self._error.extract(service, log_level, error_code)
        self._volume.record(service)
        feat_d = self._volume.extract(service)
        feat_e = self._resource.extract(service, cpu_usage, memory_mb)
        self._trace.update(trace_id, service, is_error_bool, parent_span_id)
        feat_f = self._trace.extract(trace_id)
        feat_g = self._topology.extract(service)
        self._trace.evict_old_traces()

        combined = {**feat_a, **feat_b, **feat_c, **feat_d, **feat_e, **feat_f, **feat_g}
        vector = np.array([combined[name] for name in FEATURE_NAMES], dtype=np.float32)
        return vector

    def extract_dict(self, **kwargs) -> dict[str, float]:
        """Same as extract() but returns a dict keyed by feature name."""
        vec = self.extract(**kwargs)
        return dict(zip(FEATURE_NAMES, vec.tolist()))

    def update_topology_health(self, service: str, anomaly_score: float) -> None:
        """Propagate an anomaly score update to the topology extractor."""
        self._topology.update_health(service, anomaly_score)


def build_feature_matrix(
    df,
    adjacency: Optional[dict[str, list[str]]] = None,
) -> tuple[np.ndarray, np.ndarray]:

    import pandas as pd  # local import: the rest of this module stays pandas-free

    pipeline = FeaturePipeline(adjacency=adjacency or {})
    timestamps = pd.to_datetime(df["timestamp"])

    rows = []
    for ts, row in zip(timestamps, df.itertuples(index=False)):
        vec = pipeline.extract(
            timestamp=ts.to_pydatetime(),
            service=row.service,
            instance_id=row.instance_id,
            trace_id=row.trace_id,
            parent_span_id=(getattr(row, "parent_span_id", None) or None),
            log_level=row.log_level,
            latency_ms=float(row.latency_ms),
            cpu_usage=float(row.cpu_usage),
            memory_mb=float(row.memory_mb),
            error_code=(getattr(row, "error_code", None) or None),
        )
        rows.append(vec)

    X = (
        np.stack(rows).astype(np.float32)
        if rows
        else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    )

    if "is_anomaly" in df.columns:
        y = df["is_anomaly"].astype(bool).astype(int).to_numpy()
    else:
        y = np.zeros(len(df), dtype=int)

    return X, y
