import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from valorem.data.preprocessing import align_and_trim


def _toy_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        # Introduce NaNs in first two rows so the earliest full row is 2020‑01‑03
        "b": [float("nan"), float("nan"), 10, 11, 12],
    }, index=idx)
    return df


def test_align_and_trim_intersection():
    raw = _toy_df()
    trimmed = align_and_trim(raw, freq="D")

    expected_idx = pd.date_range("2020-01-03", periods=3, freq="D")
    expected = pd.DataFrame({
        "a": [3, 4, 5],
        "b": [10, 11, 12],
    }, index=expected_idx)

    assert_frame_equal(trimmed, expected, check_dtype=False)


@pytest.mark.parametrize("freq", ["B", "D"])
def test_align_and_trim_freq(freq):
    raw = _toy_df()
    trimmed = align_and_trim(raw, freq=freq)
    # Ensure index frequency matches requested freq
    assert trimmed.index.freqstr == freq
