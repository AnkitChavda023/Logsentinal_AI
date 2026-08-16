

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LogEntry:

    timestamp: datetime

    service: str
    instance_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]


    log_level: str
    
    message: str
    error_code: Optional[str] = None
    http_status: int = 200

    latency_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_mb: float = 0.0

    user_id: Optional[str] = None

    is_anomaly: bool = False
    failure_type: Optional[str] = None
