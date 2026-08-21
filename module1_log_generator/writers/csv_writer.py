from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

from module1_log_generator.config.defaults import DEFAULT_PARQUET_FLUSH_ROWS
from module1_log_generator.writers.log_writer import BaseLogWriter

_CSV_COLUMNS = [
    "timestamp", "service", "instance_id", "trace_id", "span_id",
    "parent_span_id", "log_level", "message", "latency_ms", "cpu_usage",
    "memory_mb", "error_code", "http_status", "user_id", "is_anomaly",
    "failure_type",
]


class CsvWriter(BaseLogWriter):
    def __init__(
        self,
        output_dir: Path,
        flush_every: int = DEFAULT_PARQUET_FLUSH_ROWS,
    ) -> None:
        super().__init__(output_dir, flush_every)
        self._current_file: Optional[object] = None
        self._current_writer: Optional[csv.DictWriter] = None

    def _flush_buffer(self, rows: list[dict[str, Any]], file_num: int) -> None:
        path = self._output_dir / f"logs_part_{file_num:04d}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
