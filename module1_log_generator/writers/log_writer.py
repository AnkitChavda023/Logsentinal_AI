from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from module1_log_generator.config.defaults import DEFAULT_PARQUET_FLUSH_ROWS
from module1_log_generator.models.log_entry import LogEntry
from module1_log_generator.writers.log_formatter import to_flat_row

logger = logging.getLogger(__name__)


class BaseLogWriter(ABC):

    def __init__(
        self, output_dir: Path, flush_every: int = DEFAULT_PARQUET_FLUSH_ROWS
    ) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._buffer: list[dict[str, Any]] = []
        self._row_count: int = 0
        self._file_counter: int = 0

    @property
    def row_count(self) -> int:
        return self._row_count

    def write(self, entry: LogEntry) -> None:
        self._buffer.append(to_flat_row(entry))
        self._row_count += 1

        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self._file_counter += 1
        self._flush_buffer(self._buffer, self._file_counter)
        logger.debug(
            "Flushed %d rows to file #%d", len(self._buffer), self._file_counter
        )
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._close_resources()

    @abstractmethod
    def _flush_buffer(self, rows: list[dict], file_num: int) -> None:
        """Concrete writers implement this to persist rows."""

    def _close_resources(self) -> None:
        """Override to release file handles or connections."""


def get_writer(fmt: str, output_dir: Path, **kwargs) -> BaseLogWriter:
    
    from module1_log_generator.writers.parquet_writer import ParquetWriter
    from module1_log_generator.writers.csv_writer import CsvWriter
    from module1_log_generator.writers.ndjson_writer import NdjsonWriter

    writers = {
        "parquet": ParquetWriter,
        "csv": CsvWriter,
        "ndjson": NdjsonWriter,
    }
    if fmt not in writers:
        raise ValueError(
            f"Unknown output format '{fmt}'. Choose from: {list(writers.keys())}"
        )
    return writers[fmt](output_dir=output_dir, **kwargs)
