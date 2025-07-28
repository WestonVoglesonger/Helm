"""Feature store for logging interaction features.

This module writes structured records (dicts) to a Parquet file using Polars.
The schema is defined in `docs/arch.md` under *Evaluation Signals Schema*.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from ..config import get_settings


logger = logging.getLogger(__name__)


class FeatureStore:
    """Append‑only feature store using Parquet files."""

    def __init__(self, path: Optional[str] = None) -> None:
        settings = get_settings()
        # Default to writing into the `data/` folder
        if path is None:
            path = os.path.join(os.getcwd(), "data", "features.parquet")
        self.path = Path(path)
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, any]) -> None:
        """Append a single record to the Parquet file.

        If the file exists, the new record is concatenated.  Otherwise a new
        file is created.  Records should follow the evaluation signals schema.
        """
        df = pl.DataFrame([record])
        if self.path.exists():
            existing = pl.read_parquet(self.path)
            df = existing.vstack(df)
        df.write_parquet(self.path, compression="zstd")
        logger.debug("Appended record to feature store: %s", record)

    def load(self) -> pl.DataFrame:
        """Load all logged features into a Polars DataFrame."""
        if not self.path.exists():
            logger.warning("Feature store file does not exist: %s", self.path)
            return pl.DataFrame()
        return pl.read_parquet(self.path)