from __future__ import annotations

"""FRED data ingestion utilities.

This module centralizes all pulls from the Federal Reserve Economic Data (FRED)
API and returns tidy pandas DataFrames that downstream pipelines can write into
Valorem's SQLite feature store.

Key design points
-----------------
- Reads the *FRED_API_KEY* from ``.env`` (using *python-dotenv*).
- Uses **fredapi** under the hood for convenience and reliability.
- A single public helper, :func:`fetch`, pulls either the default macro
  dashboard (see :data:`DEFAULT_SERIES`) or a user-supplied list of series IDs.
- Returns a **wide-form** DataFrame indexed by ``pd.DatetimeIndex`` (daily
  frequency, forward-filled) with column names either the raw FRED IDs or the
  prettier aliases declared in :data:`DEFAULT_SERIES`.
- Minimal local caching: one CSV per series in ``~/.valorem_cache/fred`` to
  accelerate repeat development-time runs without hammering the API.

Example
~~~~~~~
>>> from valorem.data.ingestion.fred import fetch
>>> df = fetch()
>>> df.head()

Dependencies
~~~~~~~~~~~~
``pip install fredapi python-dotenv pandas``
"""

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

__all__ = ["DEFAULT_SERIES", "fetch"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Default macro dashboard — IDs → human-readable aliases
# ---------------------------------------------------------------------------
DEFAULT_SERIES: dict[str, str] = {
    "FEDFUNDS": "fed_funds_rate",
    "DTB3": "t_bill_3m",
    "DGS2": "yield_2y",
    "DGS10": "yield_10y",
    "T10Y2Y": "yield_curve_10y_2y_spread",
    "CPIAUCSL": "cpi_all_items_sa",
    "PCEPILFE": "core_pce_price_index",
    "T5YIFR": "breakeven_5y5y",
    "MICH": "umich_infl_exp_1y",
    "PAYEMS": "nonfarm_payrolls",
    "UNRATE": "unemployment_rate",
    "ICSA": "initial_claims_sa",
    "JTSJOL": "job_openings",
    "INDPRO": "industrial_production",
    "ADPWINDMANNERSA": "ism_pmi_mfg",
    "USSLIND": "conference_leading_index",
    "UMCSENT": "umich_sentiment",
    "BAA10Y": "baa_10y_spread",
    "BAMLC0A4CBBB": "bbb_oas",
    "STLFSI4": "st_louis_fci",
    "WALCL": "fed_balance_sheet",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_fred() -> Fred:
    """Instantiate a *fredapi.Fred* client with API key from environment."""
    load_dotenv()  # idempotent; no-op if .env absent
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not found. Populate it in your .env (see .env.example)."
        )
    return Fred(api_key=api_key)


_CACHE_DIR = Path.home() / ".valorem_cache" / "fred"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _series_cache_path(series_id: str) -> Path:
    """Return path for on-disk CSV cache of the given series ID."""
    return _CACHE_DIR / f"{series_id}.csv"


def _load_from_cache(series_id: str) -> pd.Series | None:
    """Load a series from local CSV cache if it exists and is non-empty."""
    path = _series_cache_path(series_id)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        series = df.iloc[:, 0]
        series.name = series_id
        return series
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read cache for %s: %s", series_id, exc)
        return None


def _save_to_cache(series_id: str, series: pd.Series) -> None:
    path = _series_cache_path(series_id)
    series.to_csv(path, header=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch(
    series: Iterable[str] | None = None,
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    rename: bool = True,
    forward_fill: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch one or multiple FRED series and return a *wide-form* DataFrame.

    Parameters
    ----------
    series
        Iterable of FRED series IDs. Defaults to :data:`DEFAULT_SERIES` keys.
    start, end
        Optional inclusive date filter (string ``YYYY-MM-DD`` or *datetime*).
    rename
        If *True*, rename columns with human-readable aliases from
        :data:`DEFAULT_SERIES` when available.
    forward_fill
        Whether to forward-fill missing observations (important when mixing
        daily series with lower-frequency releases).
    use_cache
        Skip API calls for any series already cached on disk.

    Returns
    -------
    pandas.DataFrame
        Index = daily ``pd.DatetimeIndex``. Columns = series. Values = floats.
    """

    fred = _get_fred()

    if series is None:
        series = list(DEFAULT_SERIES.keys())
    else:
        series = list(series)

    dfs: list[pd.Series] = []

    for sid in series:
        # Try local cache first
        cached = _load_from_cache(sid) if use_cache else None
        if cached is not None:
            s = cached
            logger.debug("Loaded %s from cache", sid)
        else:
            logger.info("Requesting %s from FRED", sid)
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            if not isinstance(s, pd.Series):
                s = pd.Series(s)
            s.name = sid
            if use_cache:
                _save_to_cache(sid, s)

        dfs.append(s)

    # Align + merge
    df = pd.concat(dfs, axis=1).sort_index()

    if start is not None:
        df = df.loc[pd.to_datetime(start) :]
    if end is not None:
        df = df.loc[: pd.to_datetime(end)]

    if forward_fill:
        df = df.ffill()

    if rename:
        df = df.rename(columns={sid: DEFAULT_SERIES.get(sid, sid) for sid in df.columns})

    return df


# ---------------------------------------------------------------------------
# Simple CLI helper (python -m valorem.data.ingestion.fred)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Download FRED series to CSV")
    parser.add_argument("--out", default="fred_data.csv", help="Output CSV path")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore on-disk cache and refetch"
    )

    args = parser.parse_args()

    try:
        df_out = fetch(start=args.start, end=args.end, use_cache=not args.no_cache)
    except Exception as exc:
        logger.error("Error fetching FRED data: %s", exc)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=True)
    logger.info("Saved FRED data → %s (%d rows, %d cols)", out_path, *df_out.shape)
