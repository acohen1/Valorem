#!/usr/bin/env python3
"""Fetch + preprocess macroeconomic data, then upsert into SQLite feature store.

Usage (from repo root):
    python scripts/update_macro.py --start 1990-01-01

The pipeline:
1. Pull default dashboard from FRED (or user-supplied list).
2. Align/trim so *all* columns have data from the same first date.
3. Persist the cleaned DataFrame into the feature store table **macro_daily**.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

# Internal imports
from valorem.data.ingestion.fred import fetch
from valorem.data.preprocessing import align_and_trim
from valorem.features.store import upsert

logger = logging.getLogger("update_macro")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def run(series: Sequence[str] | None, start: str | None, end: str | None) -> None:
    """Orchestrate pull -> preprocess -> upsert."""
    logger.info("Fetching FRED data…")
    raw = fetch(series=series, start=start, end=end)

    logger.info("Raw shape: %s", raw.shape)

    logger.info("Aligning & trimming…")
    clean = align_and_trim(raw)
    logger.info("Clean shape: %s (first date: %s)", clean.shape, clean.index[0].date())

    logger.info("Upserting to SQLite feature store (table=macro_daily)…")
    upsert(clean, table="macro_daily")
    logger.info("Done.")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Update macro feature table from FRED")
    parser.add_argument("--series", nargs="*", help="Override default series IDs (space-separated)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (passed through to fetch)")
    parser.add_argument("--end", help="End date YYYY-MM-DD")

    args = parser.parse_args()
    run(series=args.series or None, start=args.start, end=args.end)


if __name__ == "__main__":
    cli()
