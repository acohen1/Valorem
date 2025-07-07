#!/usr/bin/env python3
"""
Build daily realised-volatility features (Garman-Klass variance and
Bipower variation) from 1-minute SPY bars.

CLI
----
python -m valorem.pipelines.update_secondary --missing
python -m valorem.pipelines.update_secondary --date 2025-06-10
python -m valorem.pipelines.update_secondary --start 2025-06-01 --end 2025-06-30
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from valorem.features.compute import garman_klass, bipower_variation
from valorem.features import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

BAR_TBL  = "spy_bar_1m"
FEAT_TBL = "secondary_features"
OHLC     = ["open", "high", "low", "close"]


# ────────────────────────────────────────────────────────────────────────────
# Helper: list distinct calendar days via raw SQL (fast, memory-light)
# ────────────────────────────────────────────────────────────────────────────
def _distinct_days(table: str) -> set[str]:
    sql = f"SELECT DISTINCT substr(date,1,10) AS day FROM {table};"
    try:
        eng = store.get_engine()
        days = pd.read_sql(sql, eng)["day"].tolist()
        return set(days)
    except Exception:          # table may not exist
        return set()


# ────────────────────────────────────────────────────────────────────────────
# Load minute bars for one calendar day
# ────────────────────────────────────────────────────────────────────────────
def _load_day_bars(d: date) -> pd.DataFrame:
    """All 1-minute bars for *d* (inclusive of that day only)."""
    start = d.isoformat()                       # e.g. '2025-06-10'
    end   = (d + timedelta(days=1)).isoformat() # '2025-06-11'
    df = store.load(BAR_TBL, start=start, end=end)
    if df.empty:
        return df
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ────────────────────────────────────────────────────────────────────────────
# Compute features for one day
# ────────────────────────────────────────────────────────────────────────────
def _compute_feats(bars: pd.DataFrame) -> pd.DataFrame:
    daily_ohlc = (
        bars[OHLC].astype(float)
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    gk = garman_klass(daily_ohlc)
    bv = bipower_variation(np.log(bars["close"].astype(float)).diff().dropna())
    out = pd.concat([gk, bv], axis=1)
    out.columns = ["gk_var", "bipower_var"]
    return out.dropna()


def _process_day(d: date) -> None:
    bars = _load_day_bars(d)
    if bars.empty:
        log.warning("No minute data for %s – skip", d)
        return
    feats = _compute_feats(bars)
    if feats.empty:
        log.warning("Cannot compute features for %s – skip", d)
        return
    store.upsert(feats, FEAT_TBL, replace=True)
    log.info("%s  (rows=%d)", d, len(feats))


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Populate secondary_features table.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="single day YYYY-MM-DD")
    g.add_argument("--start", help="span start YYYY-MM-DD")
    g.add_argument("--missing", action="store_true",
                   help="process bar-days not yet in secondary_features")
    p.add_argument("--end", help="span end YYYY-MM-DD (required with --start)")
    args = p.parse_args()

    # single day
    if args.date:
        _process_day(date.fromisoformat(args.date))
        return

    # explicit span
    if args.start:
        if not args.end:
            p.error("--end required with --start")
        s, e = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if e < s:
            p.error("--end must not precede --start")
        cur = s
        while cur <= e:
            _process_day(cur)
            cur += timedelta(days=1)
        return

    # --missing
    bar_days   = _distinct_days(BAR_TBL)
    feat_days  = _distinct_days(FEAT_TBL)
    todo_days  = sorted(bar_days - feat_days)
    if not todo_days:
        log.info("Everything already processed - nothing to do.")
        return
    for dstr in todo_days:
        _process_day(date.fromisoformat(dstr))


if __name__ == "__main__":
    main()
