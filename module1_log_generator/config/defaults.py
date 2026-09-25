from __future__ import annotations
DEFAULT_FAILURE_INJECTION_RATE: float = 1 / 10_000

DEFAULT_PARQUET_FLUSH_ROWS: int = 100_000


SCALE_PRESETS: dict[str, dict] = {
    "small": {
        "services_hint": 15,
        "instances_hint": 15,
        "logs_per_sec": 10,
        "duration_hours": 24,
        "description": "Demo scale - output",
    },
    "medium": {
        "services_hint": 50,
        "instances_hint": 30,
        "logs_per_sec": 50,
        "duration_hours": 120,
        "description": "Development scale - output",
    },
    "large": {
        "services_hint": 200,
        "instances_hint": 60,
        "logs_per_sec": 500,
        "duration_hours": 144,
        "description": "Production scale - output",
    },
}

DEFAULT_LOG_LEVEL_DISTRIBUTION: dict[str, float] = {
    "DEBUG": 0.10,
    "INFO": 0.70,
    "WARN": 0.15,
    "ERROR": 0.04,
    "FATAL": 0.01,
}

DEFAULT_PEAK_MULTIPLIER: float = 3.0

DEFAULT_OFF_PEAK_MULTIPLIER: float = 1.0

DEFAULT_CPU_CORES: int = 2
DEFAULT_MEMORY_MB: int = 2048

# Metric sampling baselines
DEFAULT_CPU_BASE: float = 0.25
DEFAULT_MEMORY_BASE_FRACTION: float = 0.40

# Trace simulation
DEFAULT_MAX_TRACE_DEPTH: int = 10

# Simulation clock
SECONDS_PER_HOUR: int = 3_600
SECONDS_PER_DAY: int = 86_400

# Failure precursor window (see FailureInjector._apply_precursor): every
# failure previously began at full DEGRADED-phase symptoms with zero
# warning — the window immediately preceding onset was statistically
# identical to any other normal window, so a downstream forecaster had
# nothing to learn from. These control a pre-onset drift in
# latency/cpu/memory that intensifies as start_time approaches — still not
# label leakage (log_level, error_code, and is_anomaly are untouched; they
# still flip only at the failure's own official onset), but deliberately
# stronger than a bare-minimum "technically learnable" signal: with a
# handful of hundred training windows total, Module 3 needs a clear signal
# to find, not just a theoretically-present one. Widen/soften these later
# once there's enough data volume that a subtler signal would still be
# learnable.
PRECURSOR_SECONDS: float = 600.0
PRECURSOR_MAX_LATENCY_MULTIPLIER: float = 1.8
PRECURSOR_MAX_CPU_MULTIPLIER: float = 1.6
PRECURSOR_MAX_MEMORY_MULTIPLIER: float = 1.5