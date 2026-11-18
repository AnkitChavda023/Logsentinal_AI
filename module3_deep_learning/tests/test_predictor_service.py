"""
Tests for inference/predictor_service.py's log-buffering path
(add_logs_to_buffer / run_periodic_prediction), separate from the
request/response path already covered by the manual smoke test.
"""

import asyncio
from datetime import datetime, timezone

import networkx as nx
import pytest
import torch

from module3_deep_learning.inference.predictor_service import LogEntry, PredictorService
from module3_deep_learning.models.full_model import LogSentinelModel, ModelConfig

SERVICES = ["api-gateway", "database"]


def _make_service():
    model = LogSentinelModel(ModelConfig(num_services=len(SERVICES), num_error_codes=3))
    graph = nx.DiGraph()
    graph.add_edge("api-gateway", "database")
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return PredictorService(
        model=model, service_list=SERVICES, error_code_list=["ERR_A", "ERR_B"],
        edge_index=edge_index, graph=graph, device=torch.device("cpu"),
    )


def test_add_logs_to_buffer_and_periodic_prediction_do_not_crash_on_tz_aware_timestamps():
    """
    Regression test: add_logs_to_buffer() used to compute its cutoff with
    datetime.utcnow() (naive) and compare it against tz-aware timestamps
    parsed from real log entries ("...Z" -> "+00:00") — raising
    "can't compare offset-naive and offset-aware datetimes" on the very
    first buffered batch.
    """
    svc = _make_service()
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    logs = [
        LogEntry(
            timestamp=now_iso, service=s, instance_id="pod-1", trace_id="t", span_id="s",
            log_level="INFO", message="x", latency_ms=10.0, cpu_usage=0.1, memory_mb=100.0,
        )
        for s in SERVICES
    ]

    asyncio.run(svc.add_logs_to_buffer(logs))
    assert len(svc.log_buffer) == len(SERVICES)

    response = asyncio.run(svc.run_periodic_prediction())
    assert response is not None
    assert len(response.predictions) == len(SERVICES)


def test_run_periodic_prediction_returns_none_when_buffer_empty():
    svc = _make_service()
    assert asyncio.run(svc.run_periodic_prediction()) is None
