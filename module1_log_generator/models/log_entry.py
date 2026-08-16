"""
LogEntry — the in-memory representation of one log line.

All code in core/ and failures/ produces LogEntry objects.
Only writers/log_formatter.py converts a LogEntry into the
OTel JSON structure (§5.7) or the flat CSV row (§5.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LogEntry:
    """
    In-memory representation of a single synthetic log line.

    Fields mirror the original spec schema (§5.6) exactly.
    The OTel-aligned JSON output (§5.7) is produced by
    writers/log_formatter.py when serialising this object.
    """

    # --- Timing ---
    timestamp: datetime
    """When the event happened (UTC)."""

    # --- Identity ---
    service: str
    instance_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]

    # --- Classification ---
    log_level: str
    """One of DEBUG / INFO / WARN / ERROR / FATAL."""

    # --- Content ---
    message: str
    error_code: Optional[str] = None
    http_status: int = 200

    # --- Metrics ---
    latency_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_mb: float = 0.0

    # --- User context ---
    user_id: Optional[str] = None

    # --- Ground truth labels ---
    is_anomaly: bool = False
    failure_type: Optional[str] = None
