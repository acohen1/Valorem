# valorem/data/preprocessing/align.py
import pandas as pd

def align_and_trim(df: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    """Trim to the first row where no column is NA, then resample to *freq*."""
    # 1. locate the earliest row with every column populated
    first_valid = df.dropna().index.min()
    df = df.loc[first_valid:]

    # 2. (optional) daily/business-day grid and forward-fill
    df = (
        df.asfreq(freq)         # create empty rows on missing days
          .ffill()              # carry last obs forward
    )
    return df
