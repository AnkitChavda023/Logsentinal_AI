
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_parquet_dataset(
    parquet_dir: str | Path,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    dfs = []
    for f in files:
        part = pd.read_parquet(f)
        if sample_frac is not None:
            part = part.iloc[: int(len(part) * sample_frac)]
        dfs.append(part)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "Loaded %d rows from %d Parquet part(s) in %s%s",
        len(df), len(files), parquet_dir,
        f" (first {sample_frac:.0%} of each part, sequential)" if sample_frac is not None else "",
    )
    return df
