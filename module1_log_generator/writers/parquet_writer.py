from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from module1_log_generator.writers.log_writer import BaseLogWriter


class ParquetWriter(BaseLogWriter):

    def _flush_buffer(self, rows: list[dict[str, Any]], file_num: int) -> None:
        path = self._output_dir / f"logs_part_{file_num:04d}.parquet"
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False, compression="snappy")
