# valorem/pipelines/update_spy_market.py
#!/usr/bin/env python3
"""
Pipeline: load SPY micro-structure data from Polygon into SQLite.

Endpoint -> Table map
--------------------
* bars   -> spy_bar_1m   (INSERT-REPLACE; range-aware, idempotent)
* trades -> spy_trades   (INSERT-IGNORE; immutable ticks)
* quotes -> spy_quotes   (INSERT-IGNORE; plan-dependent)
* chain  -> spy_chain    (INSERT-REPLACE; OI / IV / greeks revise intraday)

Examples
--------
# Single trading day
python -m valorem.pipelines.update_spy_market bars   --date 2025-06-05

# Date span (overwrites any overlap)
python -m valorem.pipelines.update_spy_market bars   --start 2025-06-01 --end 2025-06-30

# Trades / quotes (still single-day only)
python -m valorem.pipelines.update_spy_market trades --date 2025-06-05

# Full option-chain snapshot
python -m valorem.pipelines.update_spy_market chain
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, time, timezone, date, timedelta
from typing import List

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
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(_h)
    logger.propagate = False

_MARKET_OPEN  = time(9, 30)
_MARKET_CLOSE = time(16, 0)


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _market_open_close(d: date) -> tuple[datetime, datetime]:
    """Return UTC datetimes delimiting regular-session minutes for *d*."""
    start = datetime.combine(d, _MARKET_OPEN,  tzinfo=timezone.utc)
    end   = datetime.combine(d, _MARKET_CLOSE, tzinfo=timezone.utc)
    return start, end


def _date_range(start: date, end: date) -> List[date]:
    """Inclusive list of calendar days."""
    step = timedelta(days=1)
    out: List[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += step
    return out


# ────────────────────────────────────────────────────────────────────────────
# bars
# ────────────────────────────────────────────────────────────────────────────
def handle_bars(
    *,
    start: date,
    end: date,
    force: bool = False,
) -> None:
    """
    Fetch all 1-minute bars between *start* and *end* (inclusive) and upsert.

    • Skips Polygon call altogether if every day in the span already exists
      (unless *force* is True).
    • Always uses REPLACE semantics so overlapping minutes are refreshed.
    """
    span_days = _date_range(start, end)

    if not force:
        missing = [d for d in span_days if not table_has_date("spy_bar_1m", d)]
        if not missing:
            logger.info("spy_bar_1m already contains %s -> %s (skip)", start, end)
            return
        start, end = min(missing), max(missing)  # shrink to the truly missing gap
        logger.info("Only %d/%d days missing - downloading %s -> %s",
                    len(missing), len(span_days), start, end)

    start_dt, _   = _market_open_close(start)
    _, end_dt     = _market_open_close(end)

    df = fetch_aggregates(start_dt, end_dt)  # UTC-indexed DataFrame
    if df.empty:
        logger.warning("Polygon returned zero rows for bars %s -> %s", start, end)
        return

    upsert(df, "spy_bar_1m", replace=True)
    logger.info("Upserted %d rows into spy_bar_1m (replace=True)", len(df))


# ────────────────────────────────────────────────────────────────────────────
# trades / quotes / chain
# ────────────────────────────────────────────────────────────────────────────
def handle_trades(day: str) -> None:
    if table_has_date("spy_trades", day):
        logger.info("spy_trades already has %s – skip", day)
        return
    df = fetch_trades(day)
    if df.empty:
        logger.warning("No trades for %s", day)
        return
    upsert(df, "spy_trades")                     # INSERT-IGNORE


def handle_quotes(day: str) -> None:
    if table_has_date("spy_quotes", day):
        logger.info("spy_quotes already has %s – skip", day)
        return
    df = fetch_quotes(day)
    if df.empty:
        logger.warning("No quotes for %s", day)
        return
    upsert(df, "spy_quotes")                    # INSERT-IGNORE


def handle_chain() -> None:
    df = fetch_option_chain_snapshot()
    if df.empty:
        logger.warning("Empty option-chain snapshot")
        return
    upsert(df, "spy_chain", replace=True)        # INSERT-REPLACE


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def cli() -> None:
    p = argparse.ArgumentParser(description="Update SPY tables in SQLite from Polygon")
    sub = p.add_subparsers(dest="endpoint", required=True)

    # ─ bars ─
    bars_p = sub.add_parser("bars", help="1-minute OHLCV")
    bars_grp = bars_p.add_mutually_exclusive_group(required=True)
    bars_grp.add_argument("--date",  help="single day YYYY-MM-DD")
    bars_grp.add_argument("--start", help="span start YYYY-MM-DD")
    bars_p.add_argument("--end",     help="span end   YYYY-MM-DD (required with --start)")
    bars_p.add_argument("--force", action="store_true",
                        help="always hit Polygon even if table already populated")

    # ─ trades / quotes ─
    for ep in ("trades", "quotes"):
        ep_p = sub.add_parser(ep, help=f"{ep} (tick-level, one day)")
        ep_p.add_argument("--date", required=True, help="YYYY-MM-DD")

    # ─ chain ─
    sub.add_parser("chain", help="full option-chain snapshot")

    args = p.parse_args()

    if args.endpoint == "bars":
        if args.date:
            day = date.fromisoformat(args.date)
            handle_bars(start=day, end=day, force=args.force)
        else:
            if not args.end:
                p.error("--end required when using --start")
            start_d = date.fromisoformat(args.start)
            end_d   = date.fromisoformat(args.end)
            if end_d < start_d:
                p.error("--end must be on/after --start")
            handle_bars(start=start_d, end=end_d, force=args.force)

    elif args.endpoint == "trades":
        handle_trades(args.date)

    elif args.endpoint == "quotes":
        handle_quotes(args.date)

    elif args.endpoint == "chain":
        handle_chain()


if __name__ == "__main__":
    cli()
