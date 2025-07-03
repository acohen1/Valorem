# valorem/data/patchtst_loader.py
"""
PatchTST-compatible DataLoader from valorem.db.

Features per calendar day
-------------------------
spy_bar_1m        ▶  open, high, low, close, vwap, volume, trades
spy_trades        ▶  share_volume, trade_count, vwap_trade
spy_chain         ▶  avg_iv, put_call_ratio, underlying_price
secondary_features▶  gk_var, bipower_var
macro_daily       ▶  (all 26 macro columns)

Output
------
DataLoader yielding (past_window, future_horizon) tensors:
    past  : [B, T, D]
    future: [B, H]   (future path of *close* price)
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import timedelta
import numpy as np

from valorem.features import store

# ─────────────────────────────────────────────────────────────
# 1.  Daily aggregations
# ─────────────────────────────────────────────────────────────
def agg_spy_bar_daily() -> pd.DataFrame:
    bar = store.load("spy_bar_1m").reset_index(names="ts")
    bar["date"] = bar["ts"].dt.date
    return (bar
        .groupby("date")
        .agg(open   =("open",   "first"),
             high   =("high",   "max"),
             low    =("low",    "min"),
             close  =("close",  "last"),
             vwap   =("vwap",   "mean"),
             volume =("volume", "sum"),
             trades =("trades", "sum"))
        .rename_axis("date"))

def agg_spy_trades_daily() -> pd.DataFrame:
    ticks = store.load("spy_trades").reset_index(names="ts")
    ticks["date"] = ticks["ts"].dt.date

    # Σ(size) and trade count per day
    daily = (ticks
        .groupby("date")
        .agg(share_volume=("size",  "sum"),
             trade_count =("id",    "count")))

    # VWAP per day = Σ(price·size) / Σ(size)
    vwap_trade = (
        (ticks["price"] * ticks["size"]).groupby(ticks["date"]).sum()
        / ticks.groupby("date")["size"].sum()
    ).rename("vwap_trade")

    return daily.join(vwap_trade).rename_axis("date")

def agg_spy_chain_daily() -> pd.DataFrame:
    chain = store.load("spy_chain").reset_index(names="ts")
    chain["date"] = chain["ts"].dt.date

    daily = (chain
        .groupby("date")
        .agg(avg_iv          =("implied_volatility", "mean"),
             underlying_price=("underlying_price",   "mean")))

    counts = (
        chain.groupby(["date", "contract_type"])
        .size()
        .unstack(fill_value=0)
    )

    puts  = counts.get("put",  pd.Series(0, index=counts.index))
    calls = counts.get("call", pd.Series(1, index=counts.index))  # avoid div/0

    put_call_ratio = (puts / calls).rename("put_call_ratio")

    return daily.join(put_call_ratio).rename_axis("date")


def daily_dataframe() -> pd.DataFrame:
    frames = [
        agg_spy_bar_daily(),
        agg_spy_trades_daily(),
        agg_spy_chain_daily(),
        store.load("secondary_features").rename_axis("date"),
        store.load("macro_daily").rename_axis("date"),
    ]
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer")
    return df.sort_index().ffill()   # forward-fill macro / IV gaps

# ─────────────────────────────────────────────────────────────
# 2.  PatchTST Dataset
# ─────────────────────────────────────────────────────────────
class PatchTSTDataset(Dataset):
    def __init__(self, df, past_window=60, horizon=5, target_col="close"):
        df = df.dropna(subset=[target_col]).ffill()      # keep only rows w/ target
        self.df = df
        self.x  = df.values.astype("float32")

        need = past_window + horizon
        if len(self.x) < need:
            raise ValueError(f"Need ≥ {need} rows after dropna, got {len(self.x)}")

        self.past = past_window
        self.hrz  = horizon
        self.target_idx = self.df.columns.get_loc(target_col)
        
        self.past = past_window
        self.hrz  = horizon

    def __len__(self) -> int:
        return len(self.x) - self.past - self.hrz + 1

    def __getitem__(self, idx: int):
        s  = idx
        e  = s + self.past
        fh = e + self.hrz
        past   = self.x[s:e]                              # [T,D]
        future = self.x[e:fh, self.target_idx]           # [H]
        return (torch.from_numpy(past),
                torch.from_numpy(future))

# ─────────────────────────────────────────────────────────────
# 3.  Public factory
# ─────────────────────────────────────────────────────────────
def make_dataloader(
    batch_size: int = 32,
    past_window: int = 60,
    horizon: int = 5,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a PyTorch DataLoader that feeds PatchTST with Valorem's daily
    feature matrix.

    Parameters
    ----------
    batch_size : int, default 32
        Number of windows per mini-batch.
    past_window : int, default 60
        Length of the historical context fed to the model (T).
    horizon : int, default 5
        Forecast horizon in days (H).  The loader returns the *close*
        column for the next ``horizon`` days as the target.
    shuffle : bool, default True
        Whether to shuffle the dataset at every epoch.  Disable (`False`)
        when creating a validation / test loader.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields tuples ``(past, future)``:

        * **past**   - tensor of shape ``[B, past_window, D]``  
          (all feature columns for the past window)

        * **future** - tensor of shape ``[B, horizon]``  
          (future path of the daily *close* price)

    Notes
    -----
    * Feature set *D* currently combines:

      - 7 OHLCV metrics from ``spy_bar_1m``  
      - 3 trade-flow features from ``spy_trades``  
      - 3 option-surface summaries from ``spy_chain``  
      - 2 realised-vol columns from ``secondary_features``  
      - 26 macro indicators from ``macro_daily``

    * Missing macro / IV values are forward-filled; rows with missing
      *close* price are dropped.

    """
    df = daily_dataframe()
    ds = PatchTSTDataset(df, past_window, horizon)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, drop_last=True)

# ─────────────────────────────────────────────────────────────
# Example run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dl = make_dataloader(batch_size=4)
    x, y = next(iter(dl))
    print("past:",   x.shape)  # [4, 60, D]
    print("future:", y.shape)  # [4, 5]
