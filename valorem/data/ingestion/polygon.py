from __future__ import annotations

"""
Polygon.io data ingestion utilities.

This module centralizes all pulls from the Polygon API for SPY-centric market data
and returns tidy pandas DataFrames ready for persistence in Valorem's feature store.

Key design points
-----------------
* Reads the POLYGON_API_KEY from `.env` (via python-dotenv).
* Provides five public helpers:
  - `fetch_aggregates` for 1-second OHLCV bars (Aggregates v3).
  - `fetch_quotes`  for NBBO quote ticks (Quotes v3).
  - `fetch_trades`  for trade prints (Trades v3).
  - `fetch_l2_snapshots` for L2 order-book depth snapshots (Snapshot v2).
  - `fetch_option_chain_snapshot` for full option chain with greeks (Snapshot v3).
* Handles cursor-based pagination automatically.
* Returns DataFrames with a `DatetimeIndex` (UTC) and consistent column naming.

Example
-------
>>> from valorem.data.ingestion.polygon import fetch_aggregates
>>> df = fetch_aggregates(start, end, symbol="SPY")
>>> df.head()

Dependencies
------------
pip install requests python-dotenv pandas tqdm
"""

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
import logging
import os
from typing import Any, Dict, Generator, List

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

__all__ = [
    "fetch_aggregates",
    "fetch_quotes",
    "fetch_trades",
    "fetch_l2_snapshots",
    "fetch_option_chain_snapshot",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_URL = "https://api.polygon.io"
_CACHE_DIR = Path.home() / ".valorem_cache" / "polygon"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    load_dotenv()
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        raise EnvironmentError("POLYGON_API_KEY missing - add it to your .env")

    sess = requests.Session()
    sess.params.update({"apiKey": key})
    return sess

def _cache_path(endpoint: str, key: str) -> Path:
    """
    Build a cache filename like:
        ~/.valorem_cache/polygon/aggregates/SPY_2024-06-05_0930-1600.csv
    """
    sub = _CACHE_DIR / endpoint
    sub.mkdir(exist_ok=True)
    hashed = key.replace("/", "_")
    return sub / f"{hashed}.csv"

def _load_cache(endpoint: str, key: str) -> pd.DataFrame | None:
    p = _cache_path(endpoint, key)
    if p.exists() and p.stat().st_size:
        try:
            return pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception as exc:
            logger.warning("Bad cache %s (%s) – rebuilding", p.name, exc)
    return None

def _save_cache(endpoint: str, key: str, df: pd.DataFrame) -> None:
    p = _cache_path(endpoint, key)
    df.to_csv(p)


# ---------------------------------------------------------------------------
# Pagination helper (cursor-based endpoints)
# ---------------------------------------------------------------------------

def _paginate(
    sess: requests.Session,
    url: str,
    params: Dict[str, Any],
) -> Generator[Dict[str, Any], None, None]:
    """Yield JSON result chunks across ?cursor pagination."""
    while True:
        r = sess.get(url, params=params, timeout=30)
        if r.status_code != 200:
            logger.error("Polygon %s - %d %s", url, r.status_code, r.text[:256])
            r.raise_for_status()
        data = r.json()
        for result in data.get("results", []):
            yield result
        cursor = data.get("next_url") or data.get("next_cursor") or data.get("next")
        if cursor:
            params = {"cursor": cursor, "apiKey": sess.params["apiKey"]}
            url = BASE_URL + cursor if cursor.startswith("/") else cursor
        else:
            break


# ---------------------------------------------------------------------------
# 1. Aggregates v3 - 1-second OHLCV
# ---------------------------------------------------------------------------

def fetch_aggregates(
    start: datetime,
    end: datetime,
    *,
    symbol: str = "SPY",
    multiplier: int = 1,
    timespan: str = "second",
    limit: int = 50000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for a given time range.

    Parameters
    ----------
    start : datetime
        UTC start timestamp for the bar data.
    end : datetime
        UTC end timestamp for the bar data.
    symbol : str, default "SPY"
        Ticker symbol to retrieve bars for.
    multiplier : int, default 1
        Bar size multiplier (e.g. 1 for 1-minute bars).
    timespan : str, default "second"
        Timespan unit ("minute", "hour", etc.).
    limit : int, default 50000
        Maximum rows per API call.
    use_cache : bool, default True
        Whether to load from / save to local CSV cache.

    Returns
    -------
    pd.DataFrame
        Indexed by UTC timestamp with columns:
        "open", "high", "low", "close", "volume", "vwap", "trades".
    """
    cache_key = f"{symbol}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"
    if use_cache and (cached := _load_cache("aggregates", cache_key)) is not None:
        logger.debug("Aggregates cache hit %s", cache_key)
        return cached

    # ------- existing API call logic unchanged -------
    sess = _get_session()
    url  = (f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
            f"{int(start.timestamp()*1000)}/{int(end.timestamp()*1000)}")
    params = {"adjusted": "true", "sort": "asc", "limit": limit}
    rows = list(_paginate(sess, url, params))
    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .assign(timestamp=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True))
            .set_index("timestamp")
            .rename(columns={"o":"open","h":"high","l":"low","c":"close",
                             "v":"volume","vw":"vwap","n":"trades"})
            .drop(columns=["t"])
            .sort_index())

    if use_cache:
        _save_cache("aggregates", cache_key, df)

    return df


# ---------------------------------------------------------------------------
# 2. Quotes v3 - NBBO quotes
# ---------------------------------------------------------------------------

def fetch_quotes(
    date: str | datetime,
    *,
    symbol: str = "SPY",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch tick-level NBBO quotes for a single date.

    Parameters
    ----------
    date : str or datetime
        Date string "YYYY-MM-DD" or datetime to fetch quotes for.
    symbol : str, default "SPY"
        Ticker symbol to retrieve quotes for.
    use_cache : bool, default True
        Whether to load from / save to local CSV cache.

    Returns
    -------
    pd.DataFrame
        Indexed by UTC timestamp with columns:
        "bid", "ask", "bid_size", "ask_size", plus any raw Polygon fields.
    """
    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
    cache_key = f"{symbol}_{date_str}"
    if use_cache and (hit := _load_cache("quotes", cache_key)) is not None:
        logger.debug("Quotes cache hit %s", cache_key)
        return hit

    url   = f"{BASE_URL}/v3/quotes/{symbol}/{date_str}"
    rows  = list(tqdm(_paginate(_get_session(), url, {}), desc=f"quotes {date_str}"))
    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .assign(timestamp=lambda d: pd.to_datetime(d["sip_timestamp"], unit="ns", utc=True))
            .set_index("timestamp")
            .rename(columns={"bid_price":"bid","ask_price":"ask",
                             "bid_size":"bid_size","ask_size":"ask_size"})
            .sort_index())

    if use_cache:
        _save_cache("quotes", cache_key, df)
    return df


# ---------------------------------------------------------------------------
# 3. Trades v3 - tick prints
# ---------------------------------------------------------------------------

def fetch_trades(
    date: str | datetime,
    *,
    symbol: str = "SPY",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch tick-level trade prints for a single date.

    Parameters
    ----------
    date : str or datetime
        Date string "YYYY-MM-DD" or datetime to fetch trades for.
    symbol : str, default "SPY"
        Ticker symbol to retrieve trade prints for.
    use_cache : bool, default True
        Whether to load from / save to local CSV cache.

    Returns
    -------
    pd.DataFrame
        Indexed by UTC timestamp with columns:
        "price", "size", "conditions", "trade_id", plus any raw Polygon fields.
    """
    date_str  = pd.to_datetime(date).strftime("%Y-%m-%d")
    cache_key = f"{symbol}_{date_str}"
    if use_cache and (hit := _load_cache("trades", cache_key)) is not None:
        logger.debug("Trades cache hit %s", cache_key)
        return hit

    url  = f"{BASE_URL}/v3/trades/{symbol}/{date_str}"
    rows = list(tqdm(_paginate(_get_session(), url, {}), desc=f"trades {date_str}"))
    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .assign(timestamp=lambda d: pd.to_datetime(d["sip_timestamp"], unit="ns", utc=True))
            .set_index("timestamp")
            .rename(columns={"p":"price","s":"size","c":"conditions","t":"trade_id"})
            .sort_index())

    if use_cache:
        _save_cache("trades", cache_key, df)
    return df



# ---------------------------------------------------------------------------
# 4. L2 Book snapshots v2
# ---------------------------------------------------------------------------

def fetch_l2_snapshots(*, symbol: str = "SPY", use_cache: bool = False) -> pd.DataFrame:
    """
    Fetch a single Level-2 order-book snapshot.

    Parameters
    ----------
    symbol : str, default "SPY"
        Ticker symbol to retrieve the L2 snapshot for.
    use_cache : bool, default False
        Whether to load from / save to local CSV cache.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame indexed by UTC timestamp with columns:
        "bid_px_1", "bid_sz_1", …, "ask_px_5", "ask_sz_5".
    """
    cache_key = f"{symbol}_{pd.Timestamp.utcnow():%Y%m%d%H%M}"
    if use_cache and (hit := _load_cache("l2", cache_key)) is not None:
        return hit

    url  = f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
    book = _get_session().get(url, timeout=30).json().get("ticker", {})
    ts   = pd.to_datetime(book.get("lastUpdate", 0), unit="ms", utc=True)

    def _side(levels: List[List[Any]], s: str) -> Dict[str, Any]:
        return {f"{s}_{k}_{i+1}": v for i,(v,k) in enumerate([(px,"px"),(sz,"sz")] for px,sz in levels[:5])}

    flat = {**_side(book.get("bids", []), "bid"), **_side(book.get("asks", []), "ask")}
    df   = pd.DataFrame([flat], index=[ts])

    if use_cache:
        _save_cache("l2", cache_key, df)
    return df



# ---------------------------------------------------------------------------
# 5. Options chain snapshot with greeks
# ---------------------------------------------------------------------------

def fetch_option_chain_snapshot(*, underlying: str = "SPY", use_cache: bool = False) -> pd.DataFrame:
    """
    Fetch a full options chain snapshot with greeks.

    Parameters
    ----------
    underlying : str, default "SPY"
        Underlying ticker symbol for the options chain.
    use_cache : bool, default False
        Whether to load from / save to local CSV cache.

    Returns
    -------
    pd.DataFrame
        Multi-row DataFrame indexed by UTC timestamp with all
        option contracts and greek columns (delta, gamma, theta, vega, etc.).
    """
    cache_key = f"{underlying}_{pd.Timestamp.utcnow():%Y%m%d%H%M}"
    if use_cache and (hit := _load_cache("chain", cache_key)) is not None:
        return hit

    url  = f"{BASE_URL}/v3/snapshot/options/{underlying}?greeks=true"
    chain = _get_session().get(url, timeout=60).json().get("results", [])
    if not chain:
        return pd.DataFrame()

    df = pd.json_normalize(chain)
    df["timestamp"] = pd.to_datetime(df["last_quote.p_T"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    if use_cache:
        _save_cache("chain", cache_key, df)
    return df



# ---------------------------------------------------------------------------
# CLI helper - python -m valorem.data.ingestion.polygon
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Download SPY Polygon data to CSV")
    parser.add_argument(
        "endpoint",
        choices=["aggregates", "quotes", "trades", "l2", "chain"],
        help="Which endpoint to pull",
    )
    parser.add_argument("--out", default="poly_data.csv", help="Output CSV path")
    parser.add_argument("--no-cache", action="store_true", help="Ignore local cache")
    parser.add_argument("--date", help="Date YYYY-MM-DD (quotes,trades,l2,chain)")
    parser.add_argument(
        "--start", help="Start datetime YYYY-MM-DDTHH:MM:SS (aggregates only)"
    )
    parser.add_argument(
        "--end", help="End datetime YYYY-MM-DDTHH:MM:SS (aggregates only)"
    )

    args = parser.parse_args()

    try:
        if args.endpoint == "aggregates":
            if not (args.start and args.end):
                parser.error("aggregates requires --start and --end")
            df_out = fetch_aggregates(
                start=pd.to_datetime(args.start).tz_localize("UTC"),
                end=pd.to_datetime(args.end).tz_localize("UTC"),
            )
        elif args.endpoint == "quotes":
            if not args.date:
                parser.error("quotes requires --date")
            df_out = fetch_quotes(args.date)
        elif args.endpoint == "trades":
            if not args.date:
                parser.error("trades requires --date")
            df_out = fetch_trades(args.date)
        elif args.endpoint == "l2":
            df_out = fetch_l2_snapshots()
        elif args.endpoint == "chain":
            df_out = fetch_option_chain_snapshot()
        else:
            parser.error("Unknown endpoint")
    except Exception as exc:
        logger.error("Error fetching Polygon data: %s", exc)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=True)
    logger.info("Saved → %s (%d rows, %d cols)", out_path, *df_out.shape)
