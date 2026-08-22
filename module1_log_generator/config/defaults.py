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