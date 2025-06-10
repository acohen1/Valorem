#!/usr/bin/env python3
"""Polygon -> SQLite loader for SPY market micro-structure data.

Usage examples (run from repo root):

```bash
# 1-second bars for 2024-06-05
python scripts/update_spy_market.py bars --date 2024-06-05

# Tick quotes (NOTE: Currently not supported by our Polygon subscription)
python scripts/update_spy_market.py quotes --date 2024-06-05

# Tick trades
python scripts/update_spy_market.py trades --date 2024-06-05

# Current option chain snapshot
python scripts/update_spy_market.py chain
```

Tables written to **valorem.db**
-------------------------------
* `spy_bar_1s`   - 1-sec OHLCV (primary key = `timestamp`)
* `spy_quotes`   - tick NBBO quotes
* `spy_trades`   - tick trade prints
* `spy_l2`       - order-book snapshot rows
* `spy_chain`    - option chain snapshot rows (append-only for now)
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
from valorem.features.store import upsert

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("update_spy_market")

_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def _market_hours(date: datetime) -> tuple[datetime, datetime]:
    start = datetime.combine(date.date(), _MARKET_OPEN, tzinfo=timezone.utc)
    end = datetime.combine(date.date(), _MARKET_CLOSE, tzinfo=timezone.utc)
    return start, end

# OVLCH BAR HANDLER
def handle_bars(date_str: str):
    date = pd.to_datetime(date_str).tz_localize("UTC")
    start, end = _market_hours(date)
    df = fetch_aggregates(start, end)
    if df.empty:
        logger.warning("No bar data for %s", date_str)
        return
    upsert(df, table="spy_bar_1s")

# QUOTE HANDLER
def handle_quotes(date_str: str):
    df = fetch_quotes(date_str)
    if df.empty:
        logger.warning("No quotes for %s", date_str)
        return
    upsert(df, table="spy_quotes")

# TRADE HANDLER
def handle_trades(date_str: str):
    df = fetch_trades(date_str)
    if df.empty:
        logger.warning("No trades for %s", date_str)
        return
    upsert(df, table="spy_trades")

# OPTION CHAIN HANDLER
def handle_chain():
    df = fetch_option_chain_snapshot()
    if df.empty:
        logger.warning("Empty option chain snapshot")
        return
    upsert(df, table="spy_chain")

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Update SPY tables in SQLite from Polygon")
    parser.add_argument("endpoint", choices=["bars", "quotes", "trades", "chain"], help="Dataset to pull")
    parser.add_argument("--date", help="Date YYYY-MM-DD (required for bars/quotes/trades)")

    args = parser.parse_args()

    if args.endpoint in {"bars", "quotes", "trades"} and not args.date:
        parser.error("--date is required for bars/quotes/trades")

    if args.endpoint == "bars":
        handle_bars(args.date)
    elif args.endpoint == "quotes":
        handle_quotes(args.date)
    elif args.endpoint == "trades":
        handle_trades(args.date)
    elif args.endpoint == "chain":
        handle_chain()
    else:
        parser.error("Unknown endpoint")


if __name__ == "__main__":
    main()
