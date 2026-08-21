from __future__ import annotations

import csv
from pathlib import Path

from module1_log_generator.utils.constants import MANIFEST_HEADER


import csv
import json
from pathlib import Path
from typing import Any

from module1_log_generator.utils.constants import MANIFEST_HEADER


class ManifestWriter:

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.writer = None

    def open(self) -> None:
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=MANIFEST_HEADER)
        self.writer.writeheader()

    def write_row(self, row: dict[str, Any]) -> None:
        if not self.writer:
            return
            
        processed_row = dict(row)
        for k, v in processed_row.items():
            if isinstance(v, (list, dict, tuple)):
                processed_row[k] = json.dumps(v)
                
        self.writer.writerow(processed_row)

    def close(self) -> None:
        if self.file:
            self.file.flush()
            self.file.close()

