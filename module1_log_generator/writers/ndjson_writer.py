from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from module1_log_generator.config.defaults import DEFAULT_PARQUET_FLUSH_ROWS
from module1_log_generator.models.log_entry import LogEntry
from module1_log_generator.writers.log_formatter import to_otel_json
from module1_log_generator.writers.log_writer import BaseLogWriter


class NdjsonWriter(BaseLogWriter):

    def write(self, entry: LogEntry) -> None:
        self._buffer.append(to_otel_json(entry))
        self._row_count += 1
        if len(self._buffer) >= self._flush_every:
            self.flush()

    def _flush_buffer(self, rows: list[dict[str, Any]], file_num: int) -> None:
        path = self._output_dir / f"logs_part_{file_num:04d}.ndjson"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str))
                fh.write("\n")
