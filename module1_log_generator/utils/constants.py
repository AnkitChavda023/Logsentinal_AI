from enum import IntEnum

SEVERITY_NUMBER: dict[str, int] = {
    "TRACE": 1,
    "TRACE2": 2,
    "TRACE3": 3,
    "TRACE4": 4,
    "DEBUG": 5,
    "DEBUG2": 6,
    "DEBUG3": 7,
    "DEBUG4": 8,
    "INFO": 9,
    "INFO2": 10,
    "INFO3": 11,
    "INFO4": 12,
    "WARN": 13,
    "WARN2": 14,
    "WARN3": 15,
    "WARN4": 16,
    "ERROR": 17,
    "ERROR2": 18,
    "ERROR3": 19,
    "ERROR4": 20,
    "FATAL": 21,
    "FATAL2": 22,
    "FATAL3": 23,
    "FATAL4": 24,
}

# Reverse mapping: number → canonical text
SEVERITY_TEXT: dict[int, str] = {v: k for k, v in SEVERITY_NUMBER.items()
                                  if not k[-1].isdigit()}

# Supported log level labels used throughout the generator
LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]

# Default log level distribution when a service YAML omits it
DEFAULT_LOG_LEVEL_DISTRIBUTION: dict[str, float] = {
    "DEBUG": 0.10,
    "INFO": 0.70,
    "WARN": 0.15,
    "ERROR": 0.04,
    "FATAL": 0.01,
}

TRACEPARENT_VERSION = "00"

HTTP_STATUS_NORMAL = [200, 201, 204]
HTTP_STATUS_CLIENT_ERROR = [400, 401, 403, 404, 422, 429]
HTTP_STATUS_SERVER_ERROR = [500, 502, 503, 504]

LOG_MESSAGE_TEMPLATES: dict[str, list[str]] = {
    "DEBUG": [
        "Processing request {trace_id}",
        "Cache lookup for key {key}",
        "Acquired lock on resource {resource}",
        "DB query planned: {query}",
        "Deserializing payload of {size} bytes",
    ],
    "INFO": [
        "Request handled successfully in {latency_ms:.1f}ms",
        "User {user_id} authenticated",
        "Order {order_id} created",
        "Payment {payment_id} processed",
        "Cache hit ratio: {ratio:.2f}",
        "Connected to {dependency}",
        "Health check passed",
        "Scheduled job {job} started",
    ],
    "WARN": [
        "Latency elevated: {latency_ms:.1f}ms (threshold {threshold}ms)",
        "Connection pool at {pct:.0f}%",
        "Retry attempt {attempt} for {operation}",
        "Rate limit approaching: {current}/{limit} req/s",
        "Memory usage at {pct:.0f}% of limit",
        "Slow query detected ({latency_ms:.0f}ms)",
        "Circuit breaker half-open",
    ],
    "ERROR": [
        "Connection pool exhausted",
        "Timeout after {timeout_ms}ms waiting for {dependency}",
        "HTTP {http_status} from {dependency}: {reason}",
        "Database query failed: {reason}",
        "Payment gateway timeout",
        "Schema validation failed for {field}",
        "Method not found: {method}",
        "Connection refused to {dependency}",
        "No route to host {host}",
        "Rate limit exceeded: HTTP 429",
    ],
    "FATAL": [
        "Out of memory — process terminating",
        "Unrecoverable database error: {reason}",
        "Service crash: {reason}",
        "Deadlock detected — terminating transaction",
    ],
}

MANIFEST_HEADER = [
    "failure_id",
    "start_timestamp",
    "end_timestamp",
    "affected_service",
    "failure_type",
    "root_cause_service",
    "severity",
    "phase_transitions",
    "resolution_time",
]

FAILURE_SEVERITY = {
    "LATENCY_SPIKE": "MEDIUM",
    "TIMEOUT": "HIGH",
    "CONNECTION_POOL_EXHAUSTION": "HIGH",
    "MEMORY_LEAK": "MEDIUM",
    "CASCADING_FAILURE": "CRITICAL",
    "THUNDERING_HERD": "HIGH",
    "NETWORK_PARTITION": "HIGH",
    "RATE_LIMITING": "MEDIUM",
    "DEADLOCK": "HIGH",
    "ZOMBIE_INSTANCE": "MEDIUM",
    "CONFIGURATION_DRIFT": "LOW",
    "DEPENDENCY_API_BREAK": "HIGH",
}
