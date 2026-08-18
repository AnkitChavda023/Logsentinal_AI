
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ResourceLimits(BaseModel):
    cpu_cores: int = Field(default=2, ge=1)
    memory_mb: int = Field(default=2048, ge=128)


class ServiceConfig(BaseModel):
    
    name: str = Field(..., description="Unique service identifier")
    instances: int = Field(default=1, ge=1, description="Number of pods/replicas")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Services this one calls",
    )
    normal_latency_ms: list[float] = Field(
        default=[10.0, 100.0],
        description="[min, max] normal response time range in ms",
    )
    normal_error_rate: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Fraction of requests that normally fail",
    )
    log_volume_per_sec: int = Field(
        default=100,
        ge=1,
        description="Average log output rate",
    )
    log_level_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Log level → probability mapping",
    )
    peak_hours: Optional[list[int]] = Field(
        default=None,
        description="[start_hour, end_hour] when traffic is 3x normal",
    )
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    special_call_rate: dict[str, float] = Field(
        default_factory=dict,
        description="Dependency name → fraction of requests that call it",
    )
    failure_injection_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Per-service override for failure injection probability",
    )

    @field_validator("normal_latency_ms")
    @classmethod
    def validate_latency_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("normal_latency_ms must be [min, max] — exactly 2 values")
        if v[0] >= v[1]:
            raise ValueError(
                f"normal_latency_ms min ({v[0]}) must be less than max ({v[1]})"
            )
        return v

    @field_validator("log_level_distribution")
    @classmethod
    def validate_level_distribution(cls, v: dict[str, float]) -> dict[str, float]:
        from module1_log_generator.utils.constants import LOG_LEVELS

        if not v:
            return v  # empty → filled by defaults later
        for level in v:
            if level not in LOG_LEVELS:
                raise ValueError(
                    f"Unknown log level '{level}'. Valid levels: {LOG_LEVELS}"
                )
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"log_level_distribution probabilities must sum to 1.0, got {total:.4f}"
            )
        return v

    @field_validator("peak_hours")
    @classmethod
    def validate_peak_hours(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("peak_hours must be [start, end] — exactly 2 values")
        start, end = v
        if not (0 <= start <= 23 and 0 <= end <= 23):
            raise ValueError("peak_hours values must be in [0, 23]")
        return v


class ServiceGraphConfig(BaseModel):

    services: list[ServiceConfig] = Field(..., min_length=1)
    allow_cycles: bool = Field(
        default=False,
        description="Set true to skip cycle detection (not recommended)",
    )
    simulation_seed: Optional[int] = Field(
        default=None,
        description="RNG seed for reproducible generation",
    )
    global_failure_injection_rate: float = Field(
        default=1 / 10_000,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def check_unique_names(self) -> "ServiceGraphConfig":
        names = [svc.name for svc in self.services]
        if len(names) != len(set(names)):
            seen: set[str] = set()
            for name in names:
                if name in seen:
                    raise ValueError(
                        f"Duplicate service name '{name}' in services list."
                    )
                seen.add(name)
        return self
