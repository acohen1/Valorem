from __future__ import annotations
"""
FRED data ingestion utilities.

This module centralizes all pulls from the Federal Reserve Economic Data (FRED)
API and returns tidy pandas DataFrames ready for Valorem's feature store.

Key design points
-----------------
* Reads *FRED_API_KEY* from `.env` (via *python-dotenv*).
* Single public helper `fetch()` pulls either:
  - the default macro dashboard (`DEFAULT_SERIES`), or
  - a user-supplied list of series IDs.
* Returns a **wide** DataFrame indexed by `DatetimeIndex` (daily), forward-filled
  as needed.
* SQLite is the single durable store.  Scripts decide
  whether data are missing before calling `fetch()`.

Example
-------
>>> from valorem.data.ingestion.fred import fetch
>>> df = fetch(start="2000-01-01")
>>> df.head()

Dependencies
------------
pip install fredapi python-dotenv pandas
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

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

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
    "SP500": "sp500_index",
    "DJIA": "dow_jones_index",
    "NASDAQCOM": "nasdaq_index",
    "VIXCLS": "vix_index",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_fred() -> Fred:
    """Instantiate a `fredapi.Fred` client using the API key in the environment."""
    load_dotenv()
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise EnvironmentError("FRED_API_KEY not found. Add it to your .env.")
    return Fred(api_key=key)


# ---------------------------------------------------------------------------
# Public fetcher
# ---------------------------------------------------------------------------
def fetch(
    series: Iterable[str] | None = None,
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    rename: bool = True,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """
    Pull one or more FRED time-series and return a wide DataFrame.

    Parameters
    ----------
    series : iterable[str] or None
        FRED series IDs to fetch.  Defaults to `DEFAULT_SERIES.keys()`.
    start, end : str | datetime | None
        Optional inclusive date bounds ("YYYY-MM-DD" or datetime).
    rename : bool, default True
        Replace raw FRED IDs with human aliases from `DEFAULT_SERIES`.
    forward_fill : bool, default True
        Forward-fill missing observations (for mix of daily / monthly series).

    Returns
    -------
    pd.DataFrame
        Index = daily `DatetimeIndex`, columns = series, values = floats.
    """
    fred = _get_fred()
    ids = list(series) if series is not None else list(DEFAULT_SERIES.keys())

    frames: list[pd.Series] = []
    for sid in ids:
        logger.info("Fetching %s from FRED", sid)
        s = fred.get_series(sid, observation_start=start, observation_end=end)
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
        s.name = sid
        frames.append(s)

    df = pd.concat(frames, axis=1).sort_index()

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
# Simple CLI helper  (python -m valorem.data.ingestion.fred)
# ---------------------------------------------------------------------------
def cli() -> None:
    import argparse
    from pathlib import Path
    import sys

    parser = argparse.ArgumentParser(description="Download FRED series to CSV")
    parser.add_argument("--out", default="fred_data.csv", help="Output CSV path")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    try:
        df_out = fetch(start=args.start, end=args.end)
    except Exception as exc:
        logger.error("Error fetching FRED data: %s", exc)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=True)
    logger.info("Saved FRED data → %s (%d rows, %d cols)", out_path, *df_out.shape)

if __name__ == "__main__":
    logger.info("Running FRED data ingestion CLI")
    cli()
