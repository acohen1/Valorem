# valorem/pipelines/update_spy_market.py
#!/usr/bin/env python3
"""
Pipeline: load SPY market micro-structure data from Polygon into SQLite.

Endpoint -> Table map
--------------------
* bars   -> spy_bar_1m   (INSERT-IGNORE, skip if date already present)
* trades -> spy_trades   (INSERT-IGNORE, skip if date already present)
* quotes -> spy_quotes   (plan-dependent; INSERT-IGNORE, skip if date present)
* chain  -> spy_chain    (INSERT-REPLACE; open-interest, IV, other vars can revise daily)

Examples
--------
python -m valorem.pipelines.update_spy_market bars   --date 2024-06-05
python -m valorem.pipelines.update_spy_market trades --date 2024-06-05
python -m valorem.pipelines.update_spy_market quotes --date 2024-06-05
python -m valorem.pipelines.update_spy_market chain
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, time, timezone

import pandas as pd

from valorem.data.ingestion.polygon import (
    fetch_aggregates,
    fetch_quotes,
    fetch_trades,
    fetch_option_chain_snapshot,
)
from valorem.features.store import upsert, table_has_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _market_hours(day: datetime) -> tuple[datetime, datetime]:
    start = datetime.combine(day.date(), _MARKET_OPEN, tzinfo=timezone.utc)
    end = datetime.combine(day.date(), _MARKET_CLOSE, tzinfo=timezone.utc)
    return start, end


def handle_bars(date_str: str) -> None:
    if table_has_date("spy_bar_1m", date_str):
        logger.info("spy_bar_1m already has %s - skip", date_str)
        return
    day = pd.to_datetime(date_str).tz_localize("UTC")
    start, end = _market_hours(day)
    df = fetch_aggregates(start, end)
    if df.empty:
        logger.warning("No bars for %s", date_str)
        return
    upsert(df, "spy_bar_1m")


def handle_trades(date_str: str) -> None:
    if table_has_date("spy_trades", date_str):
        logger.info("spy_trades already has %s - skip", date_str)
        return
    df = fetch_trades(date_str)
    if df.empty:
        logger.warning("No trades for %s", date_str)
        return
    upsert(df, "spy_trades")


def handle_quotes(date_str: str) -> None:
    if table_has_date("spy_quotes", date_str):
        logger.info("spy_quotes already has %s - skip", date_str)
        return
    df = fetch_quotes(date_str)
    if df.empty:
        logger.warning("No quotes for %s", date_str)
        return
    upsert(df, "spy_quotes")


def handle_chain() -> None:
    df = fetch_option_chain_snapshot()
    if df.empty:
        logger.warning("Empty option-chain snapshot")
        return
    upsert(df, "spy_chain", replace=True)  # replace to capture OI/IV/Greeks revisions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli() -> None:
    p = argparse.ArgumentParser(description="Update SPY tables in SQLite from Polygon")
    p.add_argument("endpoint", choices=["bars", "quotes", "trades", "chain"])
    p.add_argument("--date", help="YYYY-MM-DD (required for bars/quotes/trades)")
    args = p.parse_args()

    if args.endpoint in {"bars", "quotes", "trades"} and not args.date:
        p.error("--date required for bars / quotes / trades")

    dispatch = {
        "bars":   lambda: handle_bars(args.date),
        "quotes": lambda: handle_quotes(args.date),
        "trades": lambda: handle_trades(args.date),
        "chain":  handle_chain,
    }
    dispatch[args.endpoint]()


if __name__ == "__main__":
    cli()
