from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from module2_classical_ml.config.defaults import (
    SCORER_DEFAULT_MODEL_DIR,
    SCORER_MODEL_DIR_ENV_VAR,
    SCORER_SERVICE_GRAPH_ENV_VAR,
)
from module2_classical_ml.inference.scoring_engine import ScoringEngine
from shared.tracing.otel_setup import setup_tracing

logger = logging.getLogger(__name__)

app = FastAPI(title="LogSentinel — Classical ML Scorer", version="1.0.0")
setup_tracing(app, service_name="scorer-service")


class LogRecord(BaseModel):
    timestamp: str
    service: str
    instance_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    log_level: str
    message: str
    latency_ms: float
    cpu_usage: float
    memory_mb: float
    error_code: Optional[str] = None
    http_status: int = 200


class ScoreResponse(BaseModel):
    timestamp: str
    service: str
    instance_id: str
    anomaly_score: float
    classification: str
    method_scores: dict
    top_contributing_features: list
    shap_explanation: str


_engine: Optional[ScoringEngine] = None


@app.on_event("startup")
async def startup():
    global _engine
    model_dir = os.environ.get(SCORER_MODEL_DIR_ENV_VAR, SCORER_DEFAULT_MODEL_DIR)
    service_graph_config = os.environ.get(SCORER_SERVICE_GRAPH_ENV_VAR)
    _engine = ScoringEngine(model_dir=model_dir, service_graph_config=service_graph_config)


@app.post("/score", response_model=ScoreResponse)
async def score_log(record: LogRecord) -> ScoreResponse:
    if _engine is None:
        raise RuntimeError("ScoringEngine not initialised — startup() has not run.")

    ts = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))

    event = _engine.score(
        timestamp=ts,
        service=record.service,
        instance_id=record.instance_id,
        trace_id=record.trace_id,
        parent_span_id=record.parent_span_id,
        log_level=record.log_level,
        latency_ms=record.latency_ms,
        cpu_usage=record.cpu_usage,
        memory_mb=record.memory_mb,
        error_code=record.error_code,
    )

    payload = event.to_dict()
    return ScoreResponse(**payload)


@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": _engine is not None}
