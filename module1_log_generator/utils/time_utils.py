
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random


def elapsed_to_datetime(start: datetime, elapsed_seconds: float) -> datetime:

    return start + timedelta(seconds=elapsed_seconds)


def to_iso8601(dt: datetime) -> str:

    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def to_flat_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def compute_observed_timestamp(event_dt: datetime) -> datetime:
    delay_us = random.randint(1_000, 20_000)  # 1–20 ms in microseconds
    return event_dt + timedelta(microseconds=delay_us)


def simulation_start(year: int = 2024, month: int = 1, day: int = 15) -> datetime:
    return datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
