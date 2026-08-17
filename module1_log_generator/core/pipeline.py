from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from module1_log_generator.config.defaults import (
    DEFAULT_FAILURE_INJECTION_RATE,
    SCALE_PRESETS,
    SECONDS_PER_HOUR,
)
from module1_log_generator.core.metric_sampler import MetricSampler
from module1_log_generator.core.service_graph import ServiceGraph
from module1_log_generator.core.trace_simulator import TraceSimulator
from module1_log_generator.core.traffic_model import TrafficModel
from module1_log_generator.failures.failure_injector import FailureInjector
from module1_log_generator.failures.failure_types import FailureType
from module1_log_generator.models.failure_models import FailureSchedule
from module1_log_generator.models.schema import ServiceGraphConfig
from module1_log_generator.utils.random_utils import get_rng, seed_rng, uniform
from module1_log_generator.utils.time_utils import simulation_start
from module1_log_generator.writers.log_writer import get_writer

logger = logging.getLogger(__name__)


class GeneratorPipeline:

    def __init__(
        self,
        config: ServiceGraphConfig,
        scale: str = "medium",
        output_format: str = "parquet",
        output_dir: str = "data/generated",
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._scale = scale
        self._output_format = output_format
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Seed RNG
        effective_seed = seed if seed is not None else config.simulation_seed
        if effective_seed is not None:
            seed_rng(effective_seed)

        # Initialise components
        self._graph = ServiceGraph(config)
        self._traffic_model = TrafficModel()
        self._metric_sampler = MetricSampler()
        self._trace_simulator = TraceSimulator(
            self._graph, config, self._metric_sampler
        )

    def run(self, duration_hours: float) -> None:
 
        logger.info(
            "Starting generation: %.2f hours, scale=%s, format=%s",
            duration_hours,
            self._scale,
            self._output_format,
        )

        # STEP 1: INITIALIZE 
        self._write_graph_preview()

        start_dt = simulation_start()
        total_seconds = int(duration_hours * SECONDS_PER_HOUR)

        schedules = self._create_failure_schedules(start_dt, total_seconds)
        logger.info("Scheduled %d failure events", len(schedules))

        manifest_path = self._output_dir / "failure_manifest.csv"
        injector = FailureInjector(schedules, self._graph, manifest_path)
        injector.open_manifest()

        writer = get_writer(
            fmt=self._output_format,
            output_dir=self._output_dir,
        )

        # STEP 2: PER-SECOND LOOP 
        try:
            for elapsed in range(total_seconds):
                sim_time = start_dt + timedelta(seconds=elapsed)

                # Step 3: tick failure injector
                injector.tick(sim_time)

                # Generate log entries for this second
                entries = self._generate_second(sim_time, injector)

                # Buffer entries
                for entry in entries:
                    writer.write(entry)

                # Progress logging
                if elapsed % SECONDS_PER_HOUR == 0:
                    hour = elapsed // SECONDS_PER_HOUR
                    logger.info(
                        "Progress: hour %d/%.1f, total rows buffered: %d",
                        hour,
                        duration_hours,
                        writer.row_count,
                    )

        finally:
            #  STEP 4: OUTPUT 
            writer.flush()
            writer.close()
            injector.close_manifest()
            logger.info(
                "Generation complete. Rows written: %d. Manifest: %s",
                writer.row_count,
                manifest_path,
            )

    def _write_graph_preview(self) -> None:

        from module1_log_generator.viz.graph_visualizer import render_service_graph_html

        try:
            render_service_graph_html(
                self._config, self._graph, self._output_dir / "service_graph.html"
            )
        except Exception:
            logger.warning("Could not render service_graph.html preview", exc_info=True)

    def _generate_second(
        self, sim_time: datetime, injector: FailureInjector
    ) -> list:
        from module1_log_generator.models.log_entry import LogEntry

        all_entries: list[LogEntry] = []

        for svc in self._config.services:
            request_count = self._traffic_model.requests_this_second(svc, sim_time)

            for _ in range(request_count):
                user_id = f"user_{random.randint(10000, 99999)}"
                trace_entries = self._trace_simulator.simulate_trace(
                    entry_service=svc.name,
                    sim_time=sim_time,
                    user_id=user_id,
                )
                # Apply any active failures to each entry
                mutated = [injector.apply_to_entry(e) for e in trace_entries]
                all_entries.extend(mutated)

        return all_entries

    def _create_failure_schedules(
        self, start_dt: datetime, total_seconds: int
    ) -> list[FailureSchedule]:
        
        schedules: list[FailureSchedule] = []
        rng = get_rng()
        failure_types = list(FailureType)
        sched_counter = 0

        for svc in self._config.services:
            rate = (
                svc.failure_injection_rate
                if svc.failure_injection_rate is not None
                else self._config.global_failure_injection_rate
            )

            # Expected number of failures in the simulation window
            expected = int(total_seconds * svc.log_volume_per_sec * rate)
            n_failures = max(1, expected)

            # Spread them across the simulation
            for _ in range(n_failures):
                sched_counter += 1
                offset = int(rng.integers(60, max(total_seconds - 300, 120)))
                failure_type = failure_types[
                    int(rng.integers(0, len(failure_types)))
                ]
                duration = float(rng.integers(60, 300))

                schedule = FailureSchedule(
                    schedule_id=f"SCHED-{sched_counter:04d}",
                    service_name=svc.name,
                    failure_type=failure_type.value,
                    start_time=start_dt + timedelta(seconds=offset),
                    duration_seconds=duration,
                    params=self._build_failure_params(svc, failure_type, rng),
                )
                schedules.append(schedule)

        return schedules

    def _build_failure_params(
        self, svc, failure_type: FailureType, rng
    ) -> dict:
        
        if failure_type == FailureType.NETWORK_PARTITION and svc.dependencies:
            target = svc.dependencies[int(rng.integers(0, len(svc.dependencies)))]
            return {"target_dependency": target}

        if failure_type == FailureType.DEADLOCK:
            candidates = list(svc.dependencies) + self._graph.get_dependents(svc.name)
            if candidates:
                partner = candidates[int(rng.integers(0, len(candidates)))]
                return {"partner_service": partner}

        if failure_type == FailureType.ZOMBIE_INSTANCE:
            instance_num = int(rng.integers(1, svc.instances + 1))
            return {"zombie_instance_id": f"{svc.name}-pod-{instance_num}"}

        if failure_type == FailureType.CONFIGURATION_DRIFT:
            instance_num = int(rng.integers(1, svc.instances + 1))
            return {"drifted_instance_id": f"{svc.name}-pod-{instance_num}"}

        return {}
