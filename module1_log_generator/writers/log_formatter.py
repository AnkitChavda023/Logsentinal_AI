from __future__ import annotations

import json
from typing import Any

from module1_log_generator.models.log_entry import LogEntry
from module1_log_generator.utils.constants import SEVERITY_NUMBER
from module1_log_generator.utils.ids import build_traceparent
from module1_log_generator.utils.time_utils import compute_observed_timestamp, to_iso8601, to_flat_timestamp


def _compute_host_name(instance_id: str) -> str:
    try:
        instance_num = int(instance_id.split("-pod-")[-1])
        return f"node-{instance_num:02d}.internal"
    except (ValueError, IndexError):
        return "node-01.internal"


def to_otel_json(entry: LogEntry) -> dict[str, Any]:

    observed_dt = compute_observed_timestamp(entry.timestamp)

    return {
        "timestamp": to_iso8601(entry.timestamp),
        "observed_timestamp": to_iso8601(observed_dt),
        "severity_number": SEVERITY_NUMBER.get(entry.log_level, 9),
        "severity_text": entry.log_level,
        "trace_id": entry.trace_id,
        "span_id": entry.span_id,
        "parent_span_id": entry.parent_span_id,
        "trace_flags": "01",
        "traceparent": build_traceparent(entry.trace_id, entry.span_id),
        "body": entry.message,
        "resource": {
            "service.name": entry.service,
            "service.version": "1.0.0",
            "service.instance.id": entry.instance_id,
            "k8s.namespace.name": "prod",
            "host.name": _compute_host_name(entry.instance_id),
        },
        "attributes": {
            "latency_ms": entry.latency_ms,
            "cpu_usage": entry.cpu_usage,
            "memory_mb": entry.memory_mb,
            "error_code": entry.error_code,
            "http.status_code": entry.http_status,
            "user.id": entry.user_id,
            "is_anomaly": entry.is_anomaly,
            "failure_type": entry.failure_type,
        },
    }


def to_flat_row(entry: LogEntry) -> dict[str, Any]:

    return {
        "timestamp": to_flat_timestamp(entry.timestamp),
        "service": entry.service,
        "instance_id": entry.instance_id,
        "trace_id": entry.trace_id,
        "span_id": entry.span_id,
        "parent_span_id": entry.parent_span_id or "",
        "log_level": entry.log_level,
        "message": entry.message,
        "latency_ms": entry.latency_ms,
        "cpu_usage": entry.cpu_usage,
        "memory_mb": entry.memory_mb,
        "error_code": entry.error_code or "",
        "http_status": entry.http_status,
        "user_id": entry.user_id or "",
        "is_anomaly": entry.is_anomaly,
        "failure_type": entry.failure_type or "",
    }


def to_ndjson_line(entry: LogEntry) -> str:
    return json.dumps(to_otel_json(entry), default=str)


def to_syslog_line(entry: LogEntry) -> str:

    severity_map = {
        "DEBUG": 7,
        "INFO": 6,
        "WARN": 4,
        "ERROR": 3,
        "FATAL": 2,
    }
    facility = 1  # user-level
    severity = severity_map.get(entry.log_level, 6)
    priority = facility * 8 + severity

    ts = to_iso8601(entry.timestamp)
    hostname = _compute_host_name(entry.instance_id)
    app_name = entry.service
    proc_id = entry.instance_id
    msg_id = entry.trace_id

    return (
        f"<{priority}>1 {ts} {hostname} {app_name} {proc_id} {msg_id} - "
        f"[trace_id=\"{entry.trace_id}\" span_id=\"{entry.span_id}\"] "
        f"{entry.message}"
    )
