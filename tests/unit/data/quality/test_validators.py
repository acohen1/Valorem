"""Unit tests for DataQualityValidator."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.quality.validators import (
    DataQualityValidator,
    IssueSeverity,
    ValidationIssue,
    ValidationResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def validator():
    """Create a default validator instance."""
    return DataQualityValidator()


@pytest.fixture
def lenient_validator():
    """Create a validator with higher thresholds."""
    return DataQualityValidator(null_threshold=0.1, max_duplicate_ratio=0.05)


@pytest.fixture
def valid_underlying_bars():
    """Create valid underlying bars DataFrame."""
    n = 100
    base_time = datetime(2024, 1, 2, 9, 30)

    df = pd.DataFrame({
        "ts_utc": [base_time + timedelta(minutes=i) for i in range(n)],
        "open": np.random.uniform(400, 410, n),
        "high": np.random.uniform(410, 420, n),
        "low": np.random.uniform(390, 400, n),
        "close": np.random.uniform(400, 410, n),
        "volume": np.random.randint(1000, 10000, n),
    })

    # Ensure valid OHLC relationships
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1) - 0.01

    return df


@pytest.fixture
def valid_option_quotes():
    """Create valid option quotes DataFrame."""
    n = 100
    base_time = datetime(2024, 1, 2, 9, 30)

    return pd.DataFrame({
        "ts_utc": [base_time + timedelta(minutes=i) for i in range(n)],
        "option_symbol": ["SPY240119C00450000"] * n,
        "bid": np.random.uniform(4.0, 5.0, n),
        "ask": np.random.uniform(5.0, 6.0, n),
        "bid_size": np.random.randint(10, 100, n),
        "ask_size": np.random.randint(10, 100, n),
    })


@pytest.fixture
def valid_fred_series():
    """Create valid FRED series DataFrame."""
    n = 30
    base_date = datetime(2024, 1, 1)

    return pd.DataFrame({
        "series_id": ["DGS10"] * n,
        "obs_date": [base_date + timedelta(days=i) for i in range(n)],
        "value": np.random.uniform(4.0, 5.0, n),
        "release_datetime_utc": [
            base_date + timedelta(days=i, hours=15) for i in range(n)
        ],
    })


# ============================================================================
# ValidationResult Tests
# ============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_passed_result(self):
        """Test passed validation result."""
        result = ValidationResult(passed=True, total_rows=100)

        assert result.passed
        assert result.total_rows == 100
        assert not result.has_errors
        assert not result.has_warnings
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_failed_result_with_errors(self):
        """Test failed validation result with errors."""
        issues = [
            ValidationIssue(
                check_name="test",
                message="Test error",
                severity=IssueSeverity.ERROR,
                row_count=5,
            )
        ]
        result = ValidationResult(passed=False, issues=issues, total_rows=100)

        assert not result.passed
        assert result.has_errors
        assert result.error_count == 1

    def test_result_with_warnings_only(self):
        """Test result with warnings but no errors."""
        issues = [
            ValidationIssue(
                check_name="test",
                message="Test warning",
                severity=IssueSeverity.WARNING,
                row_count=2,
            )
        ]
        result = ValidationResult(passed=True, issues=issues, total_rows=100)

        assert result.passed
        assert not result.has_errors
        assert result.has_warnings
        assert result.warning_count == 1

    def test_summary_passed(self):
        """Test summary for passed result."""
        result = ValidationResult(passed=True, total_rows=100)
        summary = result.summary()

        assert "passed" in summary
        assert "100 rows" in summary

    def test_summary_failed(self):
        """Test summary for failed result."""
        issues = [
            ValidationIssue("test1", "Error 1", IssueSeverity.ERROR),
            ValidationIssue("test2", "Warning 1", IssueSeverity.WARNING),
        ]
        result = ValidationResult(passed=False, issues=issues, total_rows=100)
        summary = result.summary()

        assert "failed" in summary
        assert "1 errors" in summary
        assert "1 warnings" in summary


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_issue_str_representation(self):
        """Test string representation of issue."""
        issue = ValidationIssue(
            check_name="null_values",
            message="Column 'ts_utc' has 5 null values",
            severity=IssueSeverity.ERROR,
            row_count=5,
        )

        str_repr = str(issue)
        assert "[ERROR]" in str_repr
        assert "null_values" in str_repr
        assert "5 null values" in str_repr

    def test_issue_with_sample_rows(self):
        """Test issue with sample rows."""
        issue = ValidationIssue(
            check_name="duplicates",
            message="Found duplicates",
            severity=IssueSeverity.WARNING,
            row_count=10,
            sample_rows=[0, 5, 10, 15, 20],
        )

        assert len(issue.sample_rows) == 5
        assert issue.row_count == 10


# ============================================================================
# Underlying Bars Validation Tests
# ============================================================================


class TestUnderlyingBarsValidation:
    """Tests for underlying bars validation."""

    def test_valid_bars_pass(self, validator, valid_underlying_bars):
        """Test valid underlying bars pass validation."""
        result = validator.check_underlying_bars(valid_underlying_bars)

        assert result.passed
        assert result.total_rows == len(valid_underlying_bars)
        assert len(result.issues) == 0

    def test_empty_dataframe_passes(self, validator):
        """Test empty DataFrame passes validation."""
        df = pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close"])
        result = validator.check_underlying_bars(df)

        assert result.passed
        assert result.total_rows == 0

    def test_missing_required_columns_fails(self, validator):
        """Test missing required columns fails validation."""
        df = pd.DataFrame({
            "ts_utc": [datetime.now()],
            "open": [100.0],
            # Missing high, low, close
        })
        result = validator.check_underlying_bars(df)

        assert not result.passed
        assert any(i.check_name == "required_columns" for i in result.issues)

    def test_null_values_detected(self, validator, valid_underlying_bars):
        """Test null values are detected."""
        df = valid_underlying_bars.copy()
        df.loc[0, "close"] = None
        df.loc[1, "close"] = None

        result = validator.check_underlying_bars(df)

        assert not result.passed
        null_issues = [i for i in result.issues if i.check_name == "null_values"]
        assert len(null_issues) == 1
        assert null_issues[0].row_count == 2

    def test_timestamp_ordering_warning(self, validator, valid_underlying_bars):
        """Test out-of-order timestamps generate warning."""
        df = valid_underlying_bars.copy()
        # Swap two rows to break ordering
        df.iloc[10], df.iloc[11] = df.iloc[11].copy(), df.iloc[10].copy()

        result = validator.check_underlying_bars(df)

        # Should still pass (warning only) but have warning
        assert result.has_warnings
        order_issues = [i for i in result.issues if i.check_name == "timestamp_ordering"]
        assert len(order_issues) == 1

    def test_duplicate_bars_detected(self, validator, valid_underlying_bars):
        """Test duplicate bars are detected."""
        df = valid_underlying_bars.copy()
        # Duplicate first row
        df = pd.concat([df.iloc[:1], df], ignore_index=True)

        result = validator.check_underlying_bars(df)

        assert not result.passed
        dup_issues = [i for i in result.issues if i.check_name == "duplicates"]
        assert len(dup_issues) == 1
        assert dup_issues[0].row_count == 2  # Original + duplicate

    def test_invalid_ohlc_high_less_than_low(self, validator, valid_underlying_bars):
        """Test high < low is detected."""
        df = valid_underlying_bars.copy()
        df.loc[5, "high"] = df.loc[5, "low"] - 1  # high < low

        result = validator.check_underlying_bars(df)

        assert not result.passed
        ohlc_issues = [i for i in result.issues if i.check_name == "ohlc_relationships"]
        assert len(ohlc_issues) == 1
        assert ohlc_issues[0].row_count == 1

    def test_invalid_ohlc_high_less_than_open(self, validator, valid_underlying_bars):
        """Test high < open is detected."""
        df = valid_underlying_bars.copy()
        df.loc[5, "high"] = df.loc[5, "open"] - 1

        result = validator.check_underlying_bars(df)

        assert not result.passed
        ohlc_issues = [i for i in result.issues if i.check_name == "ohlc_relationships"]
        assert len(ohlc_issues) == 1

    def test_invalid_ohlc_high_less_than_close(self, validator, valid_underlying_bars):
        """Test high < close is detected."""
        df = valid_underlying_bars.copy()
        df.loc[5, "high"] = df.loc[5, "close"] - 1

        result = validator.check_underlying_bars(df)

        assert not result.passed

    def test_invalid_ohlc_low_greater_than_open(self, validator, valid_underlying_bars):
        """Test low > open is detected."""
        df = valid_underlying_bars.copy()
        df.loc[5, "low"] = df.loc[5, "open"] + 1

        result = validator.check_underlying_bars(df)

        assert not result.passed

    def test_invalid_ohlc_low_greater_than_close(self, validator, valid_underlying_bars):
        """Test low > close is detected."""
        df = valid_underlying_bars.copy()
        df.loc[5, "low"] = df.loc[5, "close"] + 1

        result = validator.check_underlying_bars(df)

        assert not result.passed

    def test_multiple_ohlc_violations(self, validator, valid_underlying_bars):
        """Test multiple OHLC violations counted correctly."""
        df = valid_underlying_bars.copy()
        df.loc[0, "high"] = df.loc[0, "low"] - 1
        df.loc[1, "high"] = df.loc[1, "low"] - 1
        df.loc[2, "high"] = df.loc[2, "low"] - 1

        result = validator.check_underlying_bars(df)

        ohlc_issues = [i for i in result.issues if i.check_name == "ohlc_relationships"]
        assert ohlc_issues[0].row_count == 3


# ============================================================================
# Option Quotes Validation Tests
# ============================================================================


class TestOptionQuotesValidation:
    """Tests for option quotes validation."""

    def test_valid_quotes_pass(self, validator, valid_option_quotes):
        """Test valid option quotes pass validation."""
        result = validator.check_option_quotes(valid_option_quotes)

        assert result.passed
        assert result.total_rows == len(valid_option_quotes)

    def test_empty_dataframe_passes(self, validator):
        """Test empty DataFrame passes validation."""
        df = pd.DataFrame(columns=["ts_utc", "option_symbol", "bid", "ask"])
        result = validator.check_option_quotes(df)

        assert result.passed
        assert result.total_rows == 0

    def test_missing_required_columns_fails(self, validator):
        """Test missing required columns fails validation."""
        df = pd.DataFrame({
            "ts_utc": [datetime.now()],
            # Missing option_symbol, bid, ask
        })
        result = validator.check_option_quotes(df)

        assert not result.passed
        assert any(i.check_name == "required_columns" for i in result.issues)

    def test_null_timestamp_detected(self, validator, valid_option_quotes):
        """Test null timestamps are detected."""
        df = valid_option_quotes.copy()
        df.loc[0, "ts_utc"] = None

        result = validator.check_option_quotes(df)

        assert not result.passed
        null_issues = [i for i in result.issues if i.check_name == "null_values"]
        assert len(null_issues) == 1

    def test_null_bid_ask_allowed(self, validator, valid_option_quotes):
        """Test null bid/ask is allowed (some schemas may have nulls)."""
        df = valid_option_quotes.copy()
        df.loc[0, "bid"] = None
        df.loc[0, "ask"] = None

        result = validator.check_option_quotes(df)

        # Should still pass - bid/ask nulls are not checked as errors
        # Only core columns (ts_utc, option_symbol) are required non-null
        assert result.passed

    def test_duplicate_quotes_detected(self, validator, valid_option_quotes):
        """Test duplicate quotes are detected."""
        df = valid_option_quotes.copy()
        df = pd.concat([df.iloc[:1], df], ignore_index=True)

        result = validator.check_option_quotes(df)

        assert not result.passed
        dup_issues = [i for i in result.issues if i.check_name == "duplicates"]
        assert len(dup_issues) == 1

    def test_crossed_markets_warning(self, validator, valid_option_quotes):
        """Test crossed markets (bid > ask) generate warning."""
        df = valid_option_quotes.copy()
        df.loc[5, "bid"] = 10.0
        df.loc[5, "ask"] = 8.0  # bid > ask

        result = validator.check_option_quotes(df)

        # Crossed markets is a warning, not error
        assert result.passed
        crossed_issues = [i for i in result.issues if i.check_name == "crossed_markets"]
        assert len(crossed_issues) == 1
        assert crossed_issues[0].row_count == 1

    def test_negative_prices_error(self, validator, valid_option_quotes):
        """Test negative prices generate error."""
        df = valid_option_quotes.copy()
        df.loc[5, "bid"] = -1.0

        result = validator.check_option_quotes(df)

        assert not result.passed
        neg_issues = [i for i in result.issues if i.check_name == "negative_prices"]
        assert len(neg_issues) == 1

    def test_negative_ask_error(self, validator, valid_option_quotes):
        """Test negative ask generates error."""
        df = valid_option_quotes.copy()
        df.loc[5, "ask"] = -1.0

        result = validator.check_option_quotes(df)

        assert not result.passed


# ============================================================================
# FRED Series Validation Tests
# ============================================================================


class TestFredSeriesValidation:
    """Tests for FRED series validation."""

    def test_valid_series_pass(self, validator, valid_fred_series):
        """Test valid FRED series passes validation."""
        result = validator.check_fred_series(valid_fred_series)

        assert result.passed
        assert result.total_rows == len(valid_fred_series)

    def test_empty_dataframe_passes(self, validator):
        """Test empty DataFrame passes validation."""
        df = pd.DataFrame(columns=["obs_date", "value", "release_datetime_utc"])
        result = validator.check_fred_series(df)

        assert result.passed
        assert result.total_rows == 0

    def test_missing_required_columns_fails(self, validator):
        """Test missing required columns fails validation."""
        df = pd.DataFrame({
            "obs_date": [datetime.now()],
            # Missing value
        })
        result = validator.check_fred_series(df)

        assert not result.passed
        assert any(i.check_name == "required_columns" for i in result.issues)

    def test_null_values_detected(self, validator, valid_fred_series):
        """Test null values are detected."""
        df = valid_fred_series.copy()
        df.loc[0, "value"] = None

        result = validator.check_fred_series(df)

        assert not result.passed
        null_issues = [i for i in result.issues if i.check_name == "null_values"]
        assert len(null_issues) == 1

    def test_timestamp_ordering_warning(self, validator, valid_fred_series):
        """Test out-of-order dates generate warning."""
        df = valid_fred_series.copy()
        df.iloc[10], df.iloc[11] = df.iloc[11].copy(), df.iloc[10].copy()

        result = validator.check_fred_series(df)

        order_issues = [i for i in result.issues if i.check_name == "timestamp_ordering"]
        assert len(order_issues) == 1

    def test_duplicate_series_detected(self, validator, valid_fred_series):
        """Test duplicate observations are detected."""
        df = valid_fred_series.copy()
        df = pd.concat([df.iloc[:1], df], ignore_index=True)

        result = validator.check_fred_series(df)

        assert not result.passed
        dup_issues = [i for i in result.issues if i.check_name == "duplicates"]
        assert len(dup_issues) == 1


# ============================================================================
# Threshold Configuration Tests
# ============================================================================


class TestThresholdConfiguration:
    """Tests for configurable thresholds."""

    def test_null_threshold_warning(self, lenient_validator, valid_underlying_bars):
        """Test null values within threshold generate warning only."""
        df = valid_underlying_bars.copy()
        # Set 5% nulls (within 10% threshold)
        null_count = int(len(df) * 0.05)
        df.loc[:null_count - 1, "close"] = None

        result = lenient_validator.check_underlying_bars(df)

        # Should pass because within threshold
        null_issues = [i for i in result.issues if i.check_name == "null_values"]
        assert len(null_issues) == 1
        assert null_issues[0].severity == IssueSeverity.WARNING

    def test_null_threshold_error(self, lenient_validator, valid_underlying_bars):
        """Test null values exceeding threshold generate error."""
        df = valid_underlying_bars.copy()
        # Set 15% nulls (exceeds 10% threshold)
        null_count = int(len(df) * 0.15)
        df.loc[:null_count - 1, "close"] = None

        result = lenient_validator.check_underlying_bars(df)

        assert not result.passed
        null_issues = [i for i in result.issues if i.check_name == "null_values"]
        assert null_issues[0].severity == IssueSeverity.ERROR

    def test_duplicate_threshold_warning(self, lenient_validator, valid_underlying_bars):
        """Test duplicates within threshold generate warning only."""
        df = valid_underlying_bars.copy()
        # Add 2% duplicates (within 5% threshold, accounting for both original and dup being counted)
        dup_count = int(len(df) * 0.02)
        duplicates = df.iloc[:dup_count].copy()
        df = pd.concat([df, duplicates], ignore_index=True)

        result = lenient_validator.check_underlying_bars(df)

        dup_issues = [i for i in result.issues if i.check_name == "duplicates"]
        if dup_issues:
            # With 2% duplicates, total duplicate count is ~4% (both original and dup)
            # which is within the 5% threshold
            assert dup_issues[0].severity == IssueSeverity.WARNING


# ============================================================================
# validate_all Tests
# ============================================================================


class TestValidateAll:
    """Tests for validate_all method."""

    def test_validate_all_with_all_data(
        self, validator, valid_underlying_bars, valid_option_quotes, valid_fred_series
    ):
        """Test validate_all with all data types."""
        results = validator.validate_all(
            underlying_df=valid_underlying_bars,
            options_df=valid_option_quotes,
            macro_dfs={"DGS10": valid_fred_series},
        )

        assert "underlying_bars" in results
        assert "option_quotes" in results
        assert "fred_DGS10" in results
        assert all(r.passed for r in results.values())

    def test_validate_all_with_none_data(self, validator):
        """Test validate_all with None data types."""
        results = validator.validate_all(
            underlying_df=None,
            options_df=None,
            macro_dfs=None,
        )

        assert len(results) == 0

    def test_validate_all_partial_data(self, validator, valid_underlying_bars):
        """Test validate_all with only underlying data."""
        results = validator.validate_all(
            underlying_df=valid_underlying_bars,
            options_df=None,
            macro_dfs=None,
        )

        assert "underlying_bars" in results
        assert "option_quotes" not in results
        assert results["underlying_bars"].passed

    def test_validate_all_multiple_fred_series(
        self, validator, valid_fred_series
    ):
        """Test validate_all with multiple FRED series."""
        fred_dfs = {
            "DGS10": valid_fred_series.copy(),
            "VIXCLS": valid_fred_series.copy(),
        }

        results = validator.validate_all(macro_dfs=fred_dfs)

        assert "fred_DGS10" in results
        assert "fred_VIXCLS" in results


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_dataframe(self, validator):
        """Test validation with single row DataFrame."""
        df = pd.DataFrame({
            "ts_utc": [datetime.now()],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        })

        result = validator.check_underlying_bars(df)
        assert result.passed

    def test_large_dataframe_performance(self, validator):
        """Test validation performance with large DataFrame."""
        import time

        n = 100000
        base_time = datetime(2024, 1, 2, 9, 30)

        df = pd.DataFrame({
            "ts_utc": [base_time + timedelta(minutes=i) for i in range(n)],
            "open": np.random.uniform(400, 410, n),
            "high": np.random.uniform(410, 420, n),
            "low": np.random.uniform(390, 400, n),
            "close": np.random.uniform(400, 410, n),
            "volume": np.random.randint(1000, 10000, n),
        })

        # Ensure valid OHLC
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1) + 0.01
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1) - 0.01

        start = time.time()
        result = validator.check_underlying_bars(df)
        elapsed = time.time() - start

        assert result.passed
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_all_null_column(self, validator):
        """Test DataFrame with entirely null column."""
        df = pd.DataFrame({
            "ts_utc": [datetime.now() + timedelta(minutes=i) for i in range(10)],
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [None] * 10,  # All nulls
            "volume": [1000] * 10,
        })

        result = validator.check_underlying_bars(df)

        assert not result.passed
        null_issues = [i for i in result.issues if i.check_name == "null_values"]
        assert null_issues[0].row_count == 10

    def test_missing_volume_column(self, validator):
        """Test validation passes without optional volume column."""
        df = pd.DataFrame({
            "ts_utc": [datetime.now() + timedelta(minutes=i) for i in range(10)],
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            # No volume column
        })

        result = validator.check_underlying_bars(df)

        # Volume is not in required columns, so should pass
        assert result.passed
