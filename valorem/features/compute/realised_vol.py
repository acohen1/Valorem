"""
Realised-volatility estimators.

Public API
----------
garman_klass(df)   → pd.Series      # needs ['open','high','low','close']
bipower_variation(r, window=1) → pd.Series   # log-returns Series

Both return a *variance* estimate; take **np.sqrt** if you need σ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["garman_klass", "bipower_variation"]

LN2 = np.log(2.0)
MU1 = np.sqrt(2.0 / np.pi)          # E[|Z|] for Z~N(0,1)


def garman_klass(df: pd.DataFrame) -> pd.Series:
    """
    Daily Garman-Klass variance estimator.

    Parameters
    ----------
    df : DataFrame
        Must have columns 'open', 'high', 'low', 'close'. Index = date.

    Returns
    -------
    pd.Series
        σ²_GK per day (same index as *df*).
    """
    o, h, l, c = (np.log(df[col].astype(float)) for col in ("open", "high", "low", "close"))
    r  = c - o
    u  = h - o
    d  = l - o
    return 0.5 * (u - d) ** 2 - (2 * LN2 - 1) * r ** 2


def bipower_variation(r: pd.Series, window: int = 1) -> pd.Series:
    """
    Bipower variation using intraday log-returns.

    Parameters
    ----------
    r : Series
        High-frequency log-returns within each day. MultiIndex or DatetimeIndex
        must be sortable so `.groupby(r.index.date)` works.
    window : int
        Gap between |r_t| and |r_{t-window}| (default = 1).

    Returns
    -------
    pd.Series
        BV_t (variance) indexed by date.
    """
    abs_r = r.abs()
    # align |r_t| with |r_{t-window}|
    prod = abs_r * abs_r.shift(window)
    bv   = (prod.groupby(r.index.date).sum()) / (MU1 ** 2)
    bv.index = pd.to_datetime(bv.index)
    return bv
