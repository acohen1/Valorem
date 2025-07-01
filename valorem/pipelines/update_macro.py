# valorem/pipelines/update_macro.py
#!/usr/bin/env python3
"""
Pipeline: refresh macro-economic dashboard from FRED and store in SQLite.

Steps
-----
1. Fetch the default macro dashboard (or a user-supplied list) via `valorem.data.ingestion.fred.fetch`.
2. `align_and_trim` so every series starts on the same date.
3. Upsert into the `macro_daily` table **with REPLACE** so FRED revisions overwrite old values.

Usage
-----
python -m valorem.pipelines.update_macro             # full dashboard, all history  
python -m valorem.pipelines.update_macro --start 1990-01-01
python -m valorem.pipelines.update_macro --series CPIAUCSL UNRATE
"""
from __future__ import annotations

import argparse
import logging
from typing import Sequence

import pandas as pd
from valorem.data.ingestion.fred import fetch
from valorem.data.preprocessing import align_and_trim
from valorem.features.store import upsert

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(series: Sequence[str] | None, start: str | None, end: str | None) -> None:
    logger.info("Fetching FRED data …")
    raw = fetch(series=series, start=start, end=end)

    logger.info("Aligning & trimming …")
    clean = align_and_trim(raw)
    logger.info("Final shape %s (first date %s)", clean.shape, clean.index[0].date())

    logger.info("Upserting into macro_daily (replace=True) …")
    upsert(clean, table="macro_daily", replace=True)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli() -> None:
    p = argparse.ArgumentParser(description="Update macro_daily table from FRED")
    p.add_argument("--series", nargs="*", help="Override default FRED IDs")
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    args = p.parse_args()

    run(series=args.series or None, start=args.start, end=args.end)


if __name__ == "__main__":
    cli()
