from __future__ import annotations
"""
Polygon.io data ingestion utilities.

This module centralizes all Polygon REST pulls for SPY-centric market data and
returns tidy pandas DataFrames that can be written straight into Valorem's
SQLite feature store.

Key design points
-----------------
* Reads *POLYGON_API_KEY* from `.env` (via *python-dotenv*).
* Four public helpers:

  - `fetch_aggregates`  -> 1-minute OHLCV bars (Aggregates v3)  
  - `fetch_trades`      -> tick-level trade prints (Trades v3)  
  - `fetch_quotes`      -> NBBO quote ticks **(call only if plan permits)**  
  - `fetch_option_chain_snapshot` -> full option chain snapshot with greeks

* Cursor-based pagination handled automatically.
* Each helper returns a DataFrame with a UTC `DatetimeIndex` and consistent
  column names ready for `upsert()`.

Example
-------
>>> from valorem.data.ingestion.polygon import fetch_aggregates
>>> df = fetch_aggregates(start, end, symbol="SPY")
>>> df.head()

Dependencies
------------
pip install requests python-dotenv pandas tqdm
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, Generator, List

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

__all__ = [
    "fetch_aggregates",
    "fetch_quotes",
    "fetch_trades",
    "fetch_option_chain_snapshot",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

BASE_URL = "https://api.polygon.io"


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------
def _get_session() -> requests.Session:
    load_dotenv()
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        raise EnvironmentError("POLYGON_API_KEY missing – add it to your .env")
    s = requests.Session()
    s.params.update({"apiKey": key})
    return s


# ---------------------------------------------------------------------------
# Cursor-pagination helper
# ---------------------------------------------------------------------------
def _paginate(
    sess: requests.Session,
    url: str,
    params: Dict[str, Any],
) -> Generator[Dict[str, Any], None, None]:
    """Yield successive result chunks following Polygon's cursor scheme."""
    while True:
        r = sess.get(url, params=params, timeout=30)
        if r.status_code != 200:
            logger.error("Polygon %s – %d %s", url, r.status_code, r.text[:256])
            r.raise_for_status()
        payload = r.json()
        yield from payload.get("results", [])
        cursor = payload.get("next_url") or payload.get("next_cursor") or payload.get("next")
        if not cursor:
            break
        params = {"cursor": cursor, "apiKey": sess.params["apiKey"]}
        url = BASE_URL + cursor if cursor.startswith("/") else cursor


# ---------------------------------------------------------------------------
# Aggregates (1-minute OHLCV) – date-range aware
# ---------------------------------------------------------------------------
from datetime import datetime, date  # <- add 'date' import at top

def fetch_aggregates(                     # signature unchanged for callers
    start: datetime | date,
    end:   datetime | date,
    *,
    symbol: str = "SPY",
    multiplier: int = 1,
    timespan: str = "minute",
    limit: int = 50_000,
) -> pd.DataFrame:
    """
    Fetch *all* 1-minute bars for `symbol` between *start* and *end* (inclusive).

    Accepts either `datetime` (with tzinfo) or `date`.  Internally we make one
    Polygon “/range” request and follow its cursor pagination, so there is no
    manual day-by-day loop in client code.

    Returns
    -------
    DataFrame indexed by UTC timestamp with
        ['open','high','low','close','volume','vwap','trades'].
    """
    # --- flexible input -----------------------------------------------------
    if isinstance(start, date) and not isinstance(start, datetime):
        start = datetime.combine(start, datetime.min.time()).astimezone(tz=pd.Timestamp.utcnow().tz)
    if isinstance(end,   date) and not isinstance(end,   datetime):
        end   = datetime.combine(end,   datetime.max.time()).astimezone(tz=pd.Timestamp.utcnow().tz)

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware (UTC preferred)")

    url = (
        f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/"
        f"{multiplier}/{timespan}/{int(start.timestamp()*1000)}/{int(end.timestamp()*1000)}"
    )

    sess  = _get_session()
    rows  = list(_paginate(sess, url, {"adjusted": "true", "sort": "asc", "limit": limit}))
    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .assign(timestamp=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True))
        .set_index("timestamp")
        .rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close",
                     "v": "volume", "vw": "vwap", "n": "trades"}
        )
        .drop(columns="t")
        .sort_index()
    )
    return df

# ---------------------------------------------------------------------------
# NBBO quotes (plan-dependent)
# ---------------------------------------------------------------------------
def fetch_quotes(
    date: str | datetime,
    *,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Fetch tick-level NBBO quotes for a single date (UTC).

    Note
    ----
    Requires Polygon **Quotes** entitlement ( >= $200/mo )
    """
    day = pd.to_datetime(date).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/v3/quotes/{symbol}?timestamp={day}"
    rows = list(tqdm(_paginate(_get_session(), url, {}), desc=f"quotes {day}"))
    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .assign(timestamp=lambda d: pd.to_datetime(d["sip_timestamp"], unit="ns", utc=True))
        .set_index("timestamp")
        .rename(
            columns={
                "bid_price": "bid",
                "ask_price": "ask",
                "bid_size": "bid_size",
                "ask_size": "ask_size",
            }
        )
        .sort_index()
    )
    return df


# ---------------------------------------------------------------------------
# Trades (tick prints)
# ---------------------------------------------------------------------------
def fetch_trades(
    date: str | datetime,
    *,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """Fetch tick-level trade prints for a single date (UTC)."""
    day = pd.to_datetime(date).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/v3/trades/{symbol}?timestamp={day}"
    rows = list(tqdm(_paginate(_get_session(), url, {}), desc=f"trades {day}"))
    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .assign(timestamp=lambda d: pd.to_datetime(d["sip_timestamp"], unit="ns", utc=True))
        .set_index("timestamp")
        .rename(columns={"p": "price", "s": "size", "c": "conditions", "t": "trade_id"})
        .sort_index()
    )
    return df


# ---------------------------------------------------------------------------
# Option-chain snapshot (robust flatten)
# ---------------------------------------------------------------------------
def fetch_option_chain_snapshot(
    *,
    underlying: str = "SPY",
    expiration: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a snapshot of all option contracts for *underlying*.

    Parameters
    ----------
    expiration : str | None, optional
        Expiration date ``YYYY-MM-DD``. If omitted, Polygon returns contracts
        for the soonest available expiration.

    Timestamp rule (per-row)
    ------------------------
    First non-null of:
        - last_quote.last_updated      (ns)
        - last_trade.sip_timestamp     (ns)
        - underlying_asset.last_updated(ns)
    

    Returns
    -------
    pd.DataFrame
        One row per contract, indexed by UTC timestamp, flat columns.
    """
    params = {"limit": 250}
    if expiration:
        params["expiration_date"] = expiration
    url  = f"{BASE_URL}/v3/snapshot/options/{underlying}"
    resp = _get_session().get(url, params=params, timeout=60)
    results: list[dict[str, Any]] = resp.json().get("results", [])
    if not results:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for r in results:
        quote = r.get("last_quote", {}) or {}
        trade = r.get("last_trade", {}) or {}
        ua    = r.get("underlying_asset", {}) or {}

        ts_ns = (
            quote.get("last_updated")
            or trade.get("sip_timestamp")
            or ua.get("last_updated")
        )
        if ts_ns is None:
            # very rare; skip contract that has no time reference
            continue
        ts = pd.to_datetime(ts_ns, unit="ns", utc=True)

        rows.append(
            {
                "timestamp": ts,
                "option_symbol":   r["details"]["ticker"],
                "contract_type":   r["details"]["contract_type"],
                "expiration_date": r["details"]["expiration_date"],
                "strike_price":    r["details"]["strike_price"],
                "break_even_price": r.get("break_even_price"),
                "open_interest":    r.get("open_interest"),
                "implied_volatility": r.get("implied_volatility"),
                # greeks
                **{f"greeks_{k}": v for k, v in r.get("greeks", {}).items()},
                # day stats
                **{f"day_{k}": v for k, v in r.get("day", {}).items()},
                # quote (may be empty)
                "bid":       quote.get("bid"),
                "ask":       quote.get("ask"),
                "bid_size":  quote.get("bid_size"),
                "ask_size":  quote.get("ask_size"),
                # trade (may be empty)
                "last_trade_price": trade.get("price"),
                "last_trade_size":  trade.get("size"),
                # underlying
                "underlying_price": ua.get("price"),
            }
        )

    return pd.DataFrame(rows).set_index("timestamp").sort_index()


# ---------------------------------------------------------------------------
# CLI helper (python -m valorem.data.ingestion.polygon)
# ---------------------------------------------------------------------------
def cli() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Download SPY Polygon data -> CSV")
    parser.add_argument("endpoint", choices=["aggregates", "quotes", "trades", "chain"])
    parser.add_argument("--out", default="poly_data.csv")
    parser.add_argument("--date", help="YYYY-MM-DD for quotes/trades")
    parser.add_argument("--start", help="YYYY-MM-DDTHH:MM:SS for aggregates")
    parser.add_argument("--end",   help="YYYY-MM-DDTHH:MM:SS for aggregates")
    args = parser.parse_args()

    if args.endpoint == "aggregates":
        if not (args.start and args.end):
            parser.error("--start and --end required for aggregates")
        df_out = fetch_aggregates(
            start=pd.to_datetime(args.start).tz_localize("UTC"),
            end=pd.to_datetime(args.end).tz_localize("UTC"),
        )
    elif args.endpoint == "quotes":
        if not args.date:
            parser.error("--date required for quotes")
        df_out = fetch_quotes(args.date)
    elif args.endpoint == "trades":
        if not args.date:
            parser.error("--date required for trades")
        df_out = fetch_trades(args.date)
    elif args.endpoint == "chain":
        df_out = fetch_option_chain_snapshot()
    else:
        parser.error("Unknown endpoint")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=True)
    logger.info("Saved -> %s (%d rows, %d cols)", out_path, *df_out.shape)

if __name__ == "__main__":
    cli()

