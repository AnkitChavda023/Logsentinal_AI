
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional

from module1_log_generator.config.defaults import DEFAULT_MAX_TRACE_DEPTH, DEFAULT_LOG_LEVEL_DISTRIBUTION
from module1_log_generator.core.metric_sampler import MetricSampler
from module1_log_generator.core.service_graph import ServiceGraph
from module1_log_generator.models.log_entry import LogEntry
from module1_log_generator.models.schema import ServiceConfig, ServiceGraphConfig
from module1_log_generator.utils.constants import (
    HTTP_STATUS_NORMAL,
    HTTP_STATUS_SERVER_ERROR,
    LOG_MESSAGE_TEMPLATES,
    SEVERITY_NUMBER,
)
from module1_log_generator.utils.ids import generate_span_id, generate_trace_id
from module1_log_generator.utils.random_utils import choose_weighted, get_rng, uniform
from module1_log_generator.utils.time_utils import elapsed_to_datetime


class TraceSimulator:
    

    def __init__(
        self,
        graph: ServiceGraph,
        config: ServiceGraphConfig,
        metric_sampler: Optional[MetricSampler] = None,
    ) -> None:
        self._graph = graph
        self._config = config
        self._sampler = metric_sampler or MetricSampler()
        self._service_map: dict[str, ServiceConfig] = {
            svc.name: svc for svc in config.services
        }

    def simulate_trace(
        self,
        entry_service: str,
        sim_time: datetime,
        user_id: Optional[str] = None,
    ) -> list[LogEntry]:
        
        trace_id = generate_trace_id()
        entries: list[LogEntry] = []
        span_offset_ms = 0.0

        self._dfs(
            service_name=entry_service,
            trace_id=trace_id,
            parent_span_id=None,
            sim_time=sim_time,
            span_offset_ms=span_offset_ms,
            depth=0,
            entries=entries,
            user_id=user_id,
        )
        return entries


    def _dfs(
        self,
        service_name: str,
        trace_id: str,
        parent_span_id: Optional[str],
        sim_time: datetime,
        span_offset_ms: float,
        depth: int,
        entries: list[LogEntry],
        user_id: Optional[str],
    ) -> float:

        if depth > DEFAULT_MAX_TRACE_DEPTH:
            return 0.0

        svc_cfg = self._service_map.get(service_name)
        if svc_cfg is None:
            return 0.0

        span_id = generate_span_id()

        # Sample baseline metrics
        latency_ms = self._sampler.sample_latency(svc_cfg)
        cpu_usage = self._sampler.sample_cpu(svc_cfg)
        memory_mb = self._sampler.sample_memory(svc_cfg)

        # Determine log level using configured distribution
        level_dist = svc_cfg.log_level_distribution or DEFAULT_LOG_LEVEL_DISTRIBUTION
        choices = list(level_dist.keys())
        weights = list(level_dist.values())
        log_level = choose_weighted(choices, weights)

        # Determine if this request errors
        is_error = uniform() < svc_cfg.normal_error_rate
        if is_error:
            log_level = "ERROR"

        # Pick a message template
        message = self._pick_message(log_level, svc_cfg, latency_ms, is_error)

        # HTTP status
        http_status = (
            random.choice(HTTP_STATUS_SERVER_ERROR)
            if is_error
            else random.choice(HTTP_STATUS_NORMAL)
        )

        # Pick an instance_id
        instance_num = get_rng().integers(1, svc_cfg.instances + 1)
        instance_id = f"{service_name}-pod-{instance_num}"

        # Compute span timestamp
        span_time = sim_time + timedelta(milliseconds=span_offset_ms)

        entry = LogEntry(
            timestamp=span_time,
            service=service_name,
            instance_id=instance_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            log_level=log_level,
            message=message,
            latency_ms=latency_ms,
            cpu_usage=cpu_usage,
            memory_mb=memory_mb,
            http_status=http_status,
            error_code="ERR_GENERIC" if is_error else None,
            user_id=user_id,
            # NOT is_error: a baseline error at the configured
            # normal_error_rate is normal system noise, not a ground-truth
            # anomaly. Only failures/failure_injector.py — which knows
            # whether a real failure is actually active on this service —
            # is allowed to flip this to True (see FailureInjector.apply_to_entry).
            is_anomaly=False,
            failure_type=None,
        )
        entries.append(entry)

        # Recurse into dependencies
        for dep_name in self._graph.get_dependencies(service_name):
            # Check special_call_rate
            call_probability = svc_cfg.special_call_rate.get(dep_name, 1.0)
            if uniform() <= call_probability:
                child_latency = self._dfs(
                    service_name=dep_name,
                    trace_id=trace_id,
                    parent_span_id=span_id,
                    sim_time=sim_time,
                    span_offset_ms=span_offset_ms + latency_ms * 0.1,
                    depth=depth + 1,
                    entries=entries,
                    user_id=user_id,
                )
                latency_ms += child_latency

        entry.latency_ms = latency_ms
        return latency_ms

    def _pick_message(
        self,
        level: str,
        svc: ServiceConfig,
        latency_ms: float,
        is_error: bool,
    ) -> str:
        templates = LOG_MESSAGE_TEMPLATES.get(level, LOG_MESSAGE_TEMPLATES["INFO"])
        tmpl = random.choice(templates)

        # Fill template variables best-effort
        try:
            return tmpl.format(
                latency_ms=latency_ms,
                dependency=svc.dependencies[0] if svc.dependencies else "upstream",
                threshold=svc.normal_latency_ms[1] * 1.5,
                timeout_ms=svc.normal_latency_ms[1] * 10,
                pct=75.0,
                user_id="user_anon",
                trace_id="...",
                key="cache_key",
                resource="lock",
                query="SELECT",
                size=512,
                order_id="ORD-001",
                payment_id="PAY-001",
                ratio=0.95,
                job="daily-report",
                current=90,
                limit=100,
                attempt=1,
                operation="payment",
                http_status=500,
                reason="internal error",
                field="payload.amount",
                method="POST /pay",
                host="db.internal",
            )
        except (KeyError, IndexError):
            return tmpl
