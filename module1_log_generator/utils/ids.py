
import secrets


def generate_trace_id() -> str:

    return secrets.token_hex(16)


def generate_span_id() -> str:

    return secrets.token_hex(8)


def build_traceparent(trace_id: str, span_id: str, flags: str = "01") -> str:

    if len(trace_id) != 32:
        raise ValueError(f"trace_id must be 32 hex chars, got {len(trace_id)}")
    if len(span_id) != 16:
        raise ValueError(f"span_id must be 16 hex chars, got {len(span_id)}")
    version = "00"
    return f"{version}-{trace_id}-{span_id}-{flags}"
