"""
Predictor Service - FastAPI endpoint for real-time predictions

Analyzes the last 5 minutes of logs across all services and returns a
failure_probability / failure_type / time_to_failure prediction per
service, plus a root-cause causal chain for the most at-risk service
(§7, §7.8). Intended to be called on a fixed cadence (e.g. every 30s,
config/defaults.py's PREDICTOR_INTERVAL_SECONDS) by whatever's driving
the pipeline — a cron job, Module 4, or the /ingest + periodic-task path
below.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

from module3_deep_learning.causal_chain.root_cause_tracer import RootCauseTracer, ServiceHealthState
from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.models.full_model import LogBatch, LogSentinelModel, ModelOutput, model_config_from_yaml

logger = logging.getLogger(__name__)


class LogEntry(BaseModel):
    """Single log entry for prediction (mirrors module1's flat schema)."""
    timestamp: str
    service: str
    instance_id: str
    trace_id: str
    span_id: str
    log_level: str
    message: str
    latency_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_mb: float = 0.0
    error_code: Optional[str] = None
    is_anomaly: bool = False


class PredictionRequest(BaseModel):
    logs: List[LogEntry] = Field(..., description="Log entries from the analysis window")
    include_causal_chain: bool = Field(True, description="Include root cause analysis")
    include_uncertainty: bool = Field(False, description="Include MC-Dropout uncertainty estimates")


class ServicePrediction(BaseModel):
    """Matches shared.schemas.prediction_event.ServicePrediction's field names."""
    service: str
    failure_probability: float
    confidence_interval: Optional[Dict[str, float]] = None
    time_to_failure_seconds: float
    time_to_failure_uncertainty_seconds: Optional[float] = None
    failure_type: str
    failure_type_confidence: float
    is_root_cause: bool = False
    causal_chain: Optional[List[Dict[str, Any]]] = None


class PredictionResponse(BaseModel):
    prediction_timestamp: str
    window_analyzed_minutes: int
    system_health_score: float
    predictions: List[ServicePrediction]
    root_cause_service: Optional[str] = None
    processing_time_ms: float


class PredictorService:
    """Real-time prediction service for LogSentinel Module 3."""

    def __init__(
        self,
        model: LogSentinelModel,
        service_list: List[str],
        error_code_list: List[str],
        edge_index: torch.Tensor,
        graph: nx.DiGraph,
        device: Optional[torch.device] = None,
        window_minutes: int = cfg.WINDOW_SIZE_MINUTES,
        max_seq_length: int = cfg.MAX_SEQ_LENGTH,
    ):
        self.model = model
        self.service_list = service_list
        self.service_to_idx = {s: i for i, s in enumerate(service_list)}
        self.error_code_to_idx = {e: i + 1 for i, e in enumerate(error_code_list)}
        self.edge_index = edge_index
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_minutes = window_minutes
        self.max_seq_length = max_seq_length

        self.root_cause_tracer = RootCauseTracer(
            graph,
            anomaly_threshold=cfg.ANOMALY_THRESHOLD,
            critical_threshold=cfg.CRITICAL_ANOMALY_THRESHOLD,
            max_depth=cfg.ROOT_CAUSE_MAX_DEPTH,
            min_root_cause_confidence=cfg.ROOT_CAUSE_MIN_CONFIDENCE,
        )

        self.log_buffer: List[LogEntry] = []
        self.buffer_lock = asyncio.Lock()

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "PredictorService initialized: %d services, window=%dmin, device=%s",
            len(service_list), window_minutes, self.device,
        )

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        start_time = time.time()

        batch = self._build_batch(request.logs)
        with torch.no_grad():
            output = (
                self.model.predict_with_uncertainty(batch)
                if request.include_uncertainty
                else self.model(batch)
            )

        health_states = self._build_health_states(output, request.logs)
        predictions, root_cause_service = self._build_predictions(output, request, health_states)

        health_score = self._compute_system_health(output)
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            window_analyzed_minutes=self.window_minutes,
            system_health_score=health_score,
            predictions=predictions,
            root_cause_service=root_cause_service,
            processing_time_ms=processing_time,
        )

    def _build_predictions(
        self, output: ModelOutput, request: PredictionRequest, health_states: Dict[str, ServiceHealthState]
    ) -> tuple[List[ServicePrediction], Optional[str]]:
        # output tensors are (1, num_services, ...) — one window.
        failure_prob = output.failure_probability[0]
        type_probs = torch.softmax(output.failure_type_logits[0], dim=-1)
        time_pred = output.time_to_failure[0]

        predictions: List[ServicePrediction] = []
        for i, service in enumerate(self.service_list):
            type_idx = int(type_probs[i].argmax())
            pred = ServicePrediction(
                service=service,
                failure_probability=float(failure_prob[i].cpu()),
                time_to_failure_seconds=float(time_pred[i].cpu()),
                failure_type=cfg.FAILURE_TYPES[type_idx],
                failure_type_confidence=float(type_probs[i, type_idx].cpu()),
                is_root_cause=False,
            )
            if request.include_uncertainty and output.failure_prob_std is not None:
                std = float(output.failure_prob_std[0, i].cpu())
                pred.confidence_interval = {
                    "lower": max(0.0, pred.failure_probability - cfg.CONFIDENCE_Z_SCORE * std),
                    "upper": min(1.0, pred.failure_probability + cfg.CONFIDENCE_Z_SCORE * std),
                }
                pred.time_to_failure_uncertainty_seconds = float(output.time_to_failure_std[0, i].cpu())
            predictions.append(pred)

        predictions.sort(key=lambda p: -p.failure_probability)

        root_cause_service = None
        if request.include_causal_chain and predictions and predictions[0].failure_probability > cfg.CRITICAL_ANOMALY_THRESHOLD:
            top_service = predictions[0].service
            try:
                causal_result = self.root_cause_tracer.trace_root_cause(
                    symptom_service=top_service, service_health_states=health_states,
                )
                root_cause_service = causal_result.root_cause_service
                for pred in predictions:
                    if pred.service == root_cause_service:
                        pred.is_root_cause = True
                    if pred.service == top_service:
                        pred.causal_chain = [link.to_dict() for link in causal_result.chain]
            except (ValueError, KeyError) as e:
                logger.error("Root cause tracing failed for %s: %s", top_service, e)

        return predictions, root_cause_service

    def _build_batch(self, logs: List[LogEntry]) -> LogBatch:
        L = self.max_seq_length
        n = min(len(logs), L)

        service_ids = np.zeros(L, dtype=np.int64)
        log_level_onehot = np.zeros((L, cfg.LOG_LEVEL_DIM), dtype=np.float32)
        error_code_ids = np.zeros(L, dtype=np.int64)
        latency_ms = np.zeros(L, dtype=np.float32)
        cpu_usage = np.zeros(L, dtype=np.float32)
        memory_mb = np.zeros(L, dtype=np.float32)
        time_since_error = np.full(L, cfg.MAX_TIME_SINCE_ERROR_S, dtype=np.float32)
        hour_of_day = np.zeros(L, dtype=np.float32)
        is_anomaly = np.zeros(L, dtype=np.float32)
        trace_depth = np.ones(L, dtype=np.float32)
        attention_mask = np.zeros(L, dtype=np.float32)

        last_error_time: Dict[str, datetime] = {}
        for i, log in enumerate(logs[:n]):
            service_ids[i] = self.service_to_idx.get(log.service, 0)
            log_level_onehot[i, cfg.LOG_LEVEL_MAP.get(log.log_level, 1)] = 1.0
            error_code_ids[i] = self.error_code_to_idx.get(log.error_code or "", 0)
            latency_ms[i] = log.latency_ms
            cpu_usage[i] = log.cpu_usage
            memory_mb[i] = log.memory_mb

            try:
                ts = datetime.fromisoformat(log.timestamp.replace("Z", "+00:00"))
            except ValueError:
                # tz-aware fallback: fromisoformat() above always produces a
                # tz-aware value (the "+00:00" replace guarantees an offset),
                # and mixing it with a naive fallback would raise
                # "can't compare offset-naive and offset-aware datetimes"
                # the moment one malformed timestamp lands next to valid ones.
                ts = datetime.now(timezone.utc)

            if log.log_level in ("ERROR", "FATAL", "CRITICAL"):
                last_error_time[log.service] = ts
            if log.service in last_error_time:
                time_since_error[i] = (ts - last_error_time[log.service]).total_seconds()

            hour_of_day[i] = ts.hour + ts.minute / 60.0
            is_anomaly[i] = 1.0 if log.is_anomaly else 0.0
            attention_mask[i] = 1.0

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        return LogBatch(
            service_ids=_t(service_ids),
            log_level_onehot=_t(log_level_onehot),
            error_code_ids=_t(error_code_ids),
            latency_ms=_t(latency_ms),
            cpu_usage=_t(cpu_usage),
            memory_mb=_t(memory_mb),
            time_since_error=_t(time_since_error),
            hour_of_day=_t(hour_of_day),
            is_anomaly=_t(is_anomaly),
            trace_depth=_t(trace_depth),
            attention_mask=_t(attention_mask),
            edge_index=self.edge_index.to(self.device),
            num_services=len(self.service_list),
        )

    def _build_health_states(self, output: ModelOutput, logs: List[LogEntry]) -> Dict[str, ServiceHealthState]:
        service_metrics: Dict[str, Dict[str, Any]] = {
            s: {"latencies": [], "errors": 0, "total": 0, "cpu": [], "memory": []} for s in self.service_list
        }
        for log in logs:
            if log.service in service_metrics:
                m = service_metrics[log.service]
                m["latencies"].append(log.latency_ms)
                m["cpu"].append(log.cpu_usage)
                m["memory"].append(log.memory_mb)
                m["total"] += 1
                if log.log_level in ("ERROR", "FATAL", "CRITICAL"):
                    m["errors"] += 1

        failure_prob = output.failure_probability[0]
        health_states = {}
        for i, service in enumerate(self.service_list):
            m = service_metrics[service]
            anomaly_score = float(failure_prob[i].cpu())
            health_states[service] = ServiceHealthState(
                service_name=service,
                anomaly_score=anomaly_score,
                failure_probability=anomaly_score,
                latency_ms=float(np.mean(m["latencies"])) if m["latencies"] else 0.0,
                error_rate=m["errors"] / max(m["total"], 1),
                cpu_usage=float(np.mean(m["cpu"])) if m["cpu"] else 0.0,
                memory_mb=float(np.mean(m["memory"])) if m["memory"] else 0.0,
            )
        return health_states

    def _compute_system_health(self, output: ModelOutput) -> float:
        """0-100 score: average of (1 - failure_probability) across all services."""
        avg_health = float((1 - output.failure_probability[0]).mean().cpu())
        return round(avg_health * 100, 2)

    async def add_logs_to_buffer(self, logs: List[LogEntry]) -> None:
        async with self.buffer_lock:
            self.log_buffer.extend(logs)
            # tz-aware to match the tz-aware timestamps parsed below (a
            # naive utcnow() here raised "can't compare offset-naive and
            # offset-aware datetimes" against every real log line).
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.window_minutes)
            self.log_buffer = [
                log for log in self.log_buffer
                if datetime.fromisoformat(log.timestamp.replace("Z", "+00:00")) > cutoff
            ]

    async def run_periodic_prediction(self) -> Optional[PredictionResponse]:
        """Run prediction on buffered logs — call this every PREDICTOR_INTERVAL_SECONDS."""
        async with self.buffer_lock:
            if not self.log_buffer:
                logger.debug("No logs in buffer, skipping prediction")
                return None
            logs = list(self.log_buffer)

        request = PredictionRequest(logs=logs, include_causal_chain=True, include_uncertainty=True)
        return await self.predict(request)


def create_predictor_app(model_path, config_path) -> FastAPI:
    """Build the FastAPI app: load the checkpoint, build the service graph, wire routes."""
    import yaml

    app = FastAPI(
        title="LogSentinel Predictor Service",
        description="Real-time failure prediction for microservices",
        version="1.0.0",
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    service_list = config["data"]["service_list"]
    error_code_list = config["data"]["error_code_list"]

    # Must match train.py's ModelConfig construction exactly (same
    # model_config_from_yaml call) — see training/evaluate.py's identical
    # note for why (a checkpoint trained with any non-default model.*
    # value would otherwise silently fail to load here, or load into the
    # wrong shape).
    model_config = model_config_from_yaml(
        model_cfg_yaml=config.get("model", {}),
        num_services=len(service_list),
        num_error_codes=len(error_code_list),
        max_seq_length=config["data"].get("max_seq_length", cfg.MAX_SEQ_LENGTH),
    )
    model = LogSentinelModel(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    graph = nx.DiGraph()
    graph.add_nodes_from(service_list)
    service_to_idx = {s: i for i, s in enumerate(service_list)}
    edges = []
    for service, deps in config["data"]["service_dependencies"].items():
        if service not in service_to_idx:
            continue
        for dep in deps:
            if dep in service_to_idx:
                graph.add_edge(service, dep)
                edges.append([service_to_idx[service], service_to_idx[dep]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)

    predictor = PredictorService(
        model=model, service_list=service_list, error_code_list=error_code_list,
        edge_index=edge_index, graph=graph, device=device,
    )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        return await predictor.predict(request)

    @app.post("/ingest")
    async def ingest_logs(logs: List[LogEntry], background_tasks: BackgroundTasks) -> Dict[str, Any]:
        background_tasks.add_task(predictor.add_logs_to_buffer, logs)
        return {"status": "accepted", "count": len(logs)}

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        return {"status": "healthy"}

    return app
