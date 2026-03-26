"""Unit tests for greeks_from_row utility."""

import math

import pandas as pd
import pytest

from src.strategy.types import Greeks
from src.utils.calculations import greeks_from_row


class TestGreeksFromRow:
    """Tests for greeks_from_row function."""

    @pytest.fixture
    def fallback(self) -> Greeks:
        """Default fallback Greeks used across tests."""
        return Greeks(delta=0.50, gamma=0.02, vega=0.30, theta=-0.05)

    def test_all_present(self, fallback):
        """Surface values used when all Greek keys are present and valid."""
        row = pd.Series(
            {
                "option_symbol": "SPY240315C00450000",
                "bid": 5.50,
                "ask": 5.80,
                "delta": 0.55,
                "gamma": 0.03,
                "vega": 0.35,
                "theta": -0.06,
            }
        )

        result = greeks_from_row(row, fallback)

        assert result.delta == pytest.approx(0.55)
        assert result.gamma == pytest.approx(0.03)
        assert result.vega == pytest.approx(0.35)
        assert result.theta == pytest.approx(-0.06)

    def test_all_nan(self, fallback):
        """Fallback values used when all Greek keys exist but are NaN."""
        row = pd.Series(
            {
                "option_symbol": "SPY240315C00450000",
                "bid": 5.50,
                "ask": 5.80,
                "delta": float("nan"),
                "gamma": float("nan"),
                "vega": float("nan"),
                "theta": float("nan"),
            }
        )

        result = greeks_from_row(row, fallback)

        assert result.delta == pytest.approx(0.50)
        assert result.gamma == pytest.approx(0.02)
        assert result.vega == pytest.approx(0.30)
        assert result.theta == pytest.approx(-0.05)

    def test_mixed_nan(self, fallback):
        """Per-field fallback when some Greeks are NaN and some are valid."""
        row = pd.Series(
            {
                "option_symbol": "SPY240315C00450000",
                "bid": 5.50,
                "ask": 5.80,
                "delta": 0.55,
                "gamma": float("nan"),
                "vega": 0.35,
                "theta": float("nan"),
            }
        )

        result = greeks_from_row(row, fallback)

        assert result.delta == pytest.approx(0.55)
        assert result.gamma == pytest.approx(0.02)  # fallback
        assert result.vega == pytest.approx(0.35)
        assert result.theta == pytest.approx(-0.05)  # fallback

    def test_missing_keys(self, fallback):
        """Fallback values used when Greek keys are absent from the Series."""
        row = pd.Series(
            {
                "option_symbol": "SPY240315C00450000",
                "bid": 5.50,
                "ask": 5.80,
            }
        )

        result = greeks_from_row(row, fallback)

        assert result.delta == pytest.approx(0.50)
        assert result.gamma == pytest.approx(0.02)
        assert result.vega == pytest.approx(0.30)
        assert result.theta == pytest.approx(-0.05)
