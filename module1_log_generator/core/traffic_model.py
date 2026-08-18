
from __future__ import annotations

from datetime import datetime

from module1_log_generator.config.defaults import (
    DEFAULT_OFF_PEAK_MULTIPLIER,
    DEFAULT_PEAK_MULTIPLIER,
)
from module1_log_generator.models.schema import ServiceConfig
from module1_log_generator.utils.random_utils import sample_poisson


class TrafficModel:

    def __init__(self, peak_multiplier: float = DEFAULT_PEAK_MULTIPLIER) -> None:
        self._peak_multiplier = peak_multiplier

    def requests_this_second(
        self, service: ServiceConfig, sim_time: datetime
    ) -> int:

        base_rate = service.log_volume_per_sec
        multiplier = self._get_multiplier(service, sim_time)
        lam = base_rate * multiplier
        return sample_poisson(lam)


    def _get_multiplier(self, service: ServiceConfig, sim_time: datetime) -> float:
        if service.peak_hours is None:
            return DEFAULT_OFF_PEAK_MULTIPLIER
        start_h, end_h = service.peak_hours
        current_h = sim_time.hour
        if start_h <= current_h < end_h:
            return self._peak_multiplier
        return DEFAULT_OFF_PEAK_MULTIPLIER
