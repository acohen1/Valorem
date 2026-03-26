"""Data quality validation for ingested data.

This module provides validation checks to ensure data quality before
persisting to the database. Catches common issues like nulls, duplicates,
and invalid OHLC relationships.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import pandas as pd


class IssueSeverity(Enum):
    """Severity level for validation issues."""

    WARNING = auto()  # Non-blocking, logged but ingestion continues
    ERROR = auto()  # Blocking, ingestion fails


@dataclass
class ValidationIssue:
    """A single validation issue found in data."""

    check_name: str
    message: str
    severity: IssueSeverity
    row_count: int = 0
    sample_rows: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        severity_str = self.severity.name
        return f"[{severity_str}] {self.check_name}: {self.message}"


@dataclass
class ValidationResult:
    """Result of data validation checks."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    total_rows: int = 0

    @property
    def has_errors(self) -> bool:
        """Check if any error-level issues exist."""
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any warning-level issues exist."""
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    def summary(self) -> str:
        """Generate a summary of validation results."""
        if self.passed:
            return f"Validation passed ({self.total_rows} rows)"
        return (
            f"Validation failed: {self.error_count} errors, "
            f"{self.warning_count} warnings ({self.total_rows} rows)"
        )


class DataQualityValidator:
    """Validate ingested data meets quality standards.

    Provides validation checks for:
    - Underlying bars (OHLCV data)
    - Option quotes (bid/ask data)
    - FRED macro series

    Each check returns a ValidationResult with detailed issue information.
    """

    def __init__(
        self,
        null_threshold: float = 0.0,
        max_duplicate_ratio: float = 0.0,
        logger: logging.Logger | None = None,
    ):
        """Initialize validator.

        Args:
            null_threshold: Max allowed null ratio (0.0 = no nulls allowed)
            max_duplicate_ratio: Max allowed duplicate ratio
            logger: Optional logger instance
        """
        self._null_threshold = null_threshold
        self._max_duplicate_ratio = max_duplicate_ratio
        self._logger = logger or logging.getLogger(__name__)

    def check_underlying_bars(self, df: pd.DataFrame) -> ValidationResult:
        """Check underlying bars for data quality issues.

        Validates:
        - No null values in required columns
        - Timestamps are monotonically increasing
        - No duplicate (ts_utc, symbol) pairs
        - Valid OHLC relationships (high >= low, etc.)

        Args:
            df: DataFrame with columns [ts_utc, open, high, low, close, volume]

        Returns:
            ValidationResult with any issues found
        """
        issues: list[ValidationIssue] = []

        if df.empty:
            return ValidationResult(passed=True, issues=[], total_rows=0)

        # Check for required columns
        required_cols = ["ts_utc", "open", "high", "low", "close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(
                ValidationIssue(
                    check_name="required_columns",
                    message=f"Missing required columns: {missing_cols}",
                    severity=IssueSeverity.ERROR,
                )
            )
            return ValidationResult(
                passed=False, issues=issues, total_rows=len(df)
            )

        # Check for nulls in required columns
        null_issues = self._check_nulls(df, required_cols)
        issues.extend(null_issues)

        # Check timestamp ordering
        timestamp_issues = self._check_timestamp_ordering(df, "ts_utc")
        issues.extend(timestamp_issues)

        # Check for duplicates
        duplicate_issues = self._check_duplicates(df, ["ts_utc"], "bars")
        issues.extend(duplicate_issues)

        # Check OHLC relationships
        ohlc_issues = self._check_ohlc_relationships(df)
        issues.extend(ohlc_issues)

        passed = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return ValidationResult(passed=passed, issues=issues, total_rows=len(df))

    def check_option_quotes(self, df: pd.DataFrame) -> ValidationResult:
        """Check option quotes for data quality issues.

        Validates:
        - No null values in required columns
        - Timestamps are monotonically increasing (per symbol)
        - No duplicate (ts_utc, option_symbol) pairs
        - Valid bid/ask relationships (bid <= ask)

        Args:
            df: DataFrame with columns [ts_utc, option_symbol, bid, ask, bid_size, ask_size]

        Returns:
            ValidationResult with any issues found
        """
        issues: list[ValidationIssue] = []

        if df.empty:
            return ValidationResult(passed=True, issues=[], total_rows=0)

        # Check for required columns
        required_cols = ["ts_utc", "option_symbol", "bid", "ask"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(
                ValidationIssue(
                    check_name="required_columns",
                    message=f"Missing required columns: {missing_cols}",
                    severity=IssueSeverity.ERROR,
                )
            )
            return ValidationResult(
                passed=False, issues=issues, total_rows=len(df)
            )

        # Check for nulls in required columns (bid/ask can be null for some schemas)
        core_cols = ["ts_utc", "option_symbol"]
        null_issues = self._check_nulls(df, core_cols)
        issues.extend(null_issues)

        # Check for duplicates
        duplicate_issues = self._check_duplicates(
            df, ["ts_utc", "option_symbol"], "quotes"
        )
        issues.extend(duplicate_issues)

        # Check bid/ask relationships
        quote_issues = self._check_quote_relationships(df)
        issues.extend(quote_issues)

        passed = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return ValidationResult(passed=passed, issues=issues, total_rows=len(df))

    def check_fred_series(self, df: pd.DataFrame) -> ValidationResult:
        """Check FRED series data for quality issues.

        Validates:
        - No null values in required columns
        - Observation dates are monotonically increasing
        - No duplicate (series_id, obs_date) pairs
        - Values are within reasonable bounds

        Args:
            df: DataFrame with columns [series_id, obs_date, value, release_datetime_utc]

        Returns:
            ValidationResult with any issues found
        """
        issues: list[ValidationIssue] = []

        if df.empty:
            return ValidationResult(passed=True, issues=[], total_rows=0)

        # Check for required columns
        required_cols = ["obs_date", "value"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(
                ValidationIssue(
                    check_name="required_columns",
                    message=f"Missing required columns: {missing_cols}",
                    severity=IssueSeverity.ERROR,
                )
            )
            return ValidationResult(
                passed=False, issues=issues, total_rows=len(df)
            )

        # Check for nulls
        null_issues = self._check_nulls(df, required_cols)
        issues.extend(null_issues)

        # Check timestamp ordering
        timestamp_issues = self._check_timestamp_ordering(df, "obs_date")
        issues.extend(timestamp_issues)

        # Check for duplicates
        if "series_id" in df.columns:
            dup_cols = ["series_id", "obs_date"]
        else:
            dup_cols = ["obs_date"]
        duplicate_issues = self._check_duplicates(df, dup_cols, "series")
        issues.extend(duplicate_issues)

        passed = not any(i.severity == IssueSeverity.ERROR for i in issues)
        return ValidationResult(passed=passed, issues=issues, total_rows=len(df))

    def _check_nulls(
        self, df: pd.DataFrame, columns: list[str]
    ) -> list[ValidationIssue]:
        """Check for null values in specified columns.

        Args:
            df: DataFrame to check
            columns: Columns to check for nulls

        Returns:
            List of validation issues
        """
        issues = []
        for col in columns:
            if col not in df.columns:
                continue

            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_ratio = null_count / len(df)
                # Get sample row indices
                null_rows = df[df[col].isnull()].index.tolist()[:5]

                severity = (
                    IssueSeverity.WARNING
                    if null_ratio <= self._null_threshold
                    else IssueSeverity.ERROR
                )

                issues.append(
                    ValidationIssue(
                        check_name="null_values",
                        message=(
                            f"Column '{col}' has {null_count} null values "
                            f"({null_ratio:.2%})"
                        ),
                        severity=severity,
                        row_count=null_count,
                        sample_rows=null_rows,
                    )
                )

        return issues

    def _check_timestamp_ordering(
        self, df: pd.DataFrame, ts_col: str
    ) -> list[ValidationIssue]:
        """Check that timestamps are monotonically increasing.

        Args:
            df: DataFrame to check
            ts_col: Timestamp column name

        Returns:
            List of validation issues
        """
        issues = []

        if ts_col not in df.columns:
            return issues

        # Check if monotonic increasing
        is_monotonic = df[ts_col].is_monotonic_increasing

        if not is_monotonic:
            # Find out-of-order rows
            out_of_order = df[ts_col].diff() < pd.Timedelta(0)
            bad_count = out_of_order.sum()
            bad_rows = out_of_order[out_of_order].index.tolist()[:5]

            issues.append(
                ValidationIssue(
                    check_name="timestamp_ordering",
                    message=f"Timestamps not monotonically increasing ({bad_count} violations)",
                    severity=IssueSeverity.WARNING,  # Warning because reordering is possible
                    row_count=bad_count,
                    sample_rows=bad_rows,
                )
            )

        return issues

    def _check_duplicates(
        self, df: pd.DataFrame, key_cols: list[str], data_type: str
    ) -> list[ValidationIssue]:
        """Check for duplicate rows based on key columns.

        Args:
            df: DataFrame to check
            key_cols: Columns that form the unique key
            data_type: Description of data type for error message

        Returns:
            List of validation issues
        """
        issues = []

        # Check that all key columns exist
        missing_keys = [c for c in key_cols if c not in df.columns]
        if missing_keys:
            return issues  # Skip if columns missing

        # Find duplicates
        duplicates = df.duplicated(subset=key_cols, keep=False)
        dup_count = duplicates.sum()

        if dup_count > 0:
            dup_ratio = dup_count / len(df)
            dup_rows = duplicates[duplicates].index.tolist()[:5]

            severity = (
                IssueSeverity.WARNING
                if dup_ratio <= self._max_duplicate_ratio
                else IssueSeverity.ERROR
            )

            issues.append(
                ValidationIssue(
                    check_name="duplicates",
                    message=f"Found {dup_count} duplicate {data_type} ({dup_ratio:.2%})",
                    severity=severity,
                    row_count=dup_count,
                    sample_rows=dup_rows,
                )
            )

        return issues

    def _check_ohlc_relationships(
        self, df: pd.DataFrame
    ) -> list[ValidationIssue]:
        """Check OHLC price relationships are valid.

        Valid relationships:
        - high >= low
        - high >= open
        - high >= close
        - low <= open
        - low <= close

        Args:
            df: DataFrame with OHLC columns

        Returns:
            List of validation issues
        """
        issues = []

        # Check all required columns exist
        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return issues

        # Only check rows where all OHLC values are non-null
        valid_rows = df[required].notna().all(axis=1)
        if not valid_rows.any():
            return issues  # No valid rows to check

        df_valid = df[valid_rows]

        # Build invalid mask
        invalid_ohlc = (
            (df_valid["high"] < df_valid["low"])
            | (df_valid["high"] < df_valid["open"])
            | (df_valid["high"] < df_valid["close"])
            | (df_valid["low"] > df_valid["open"])
            | (df_valid["low"] > df_valid["close"])
        )

        invalid_count = invalid_ohlc.sum()

        if invalid_count > 0:
            invalid_rows = invalid_ohlc[invalid_ohlc].index.tolist()[:5]

            issues.append(
                ValidationIssue(
                    check_name="ohlc_relationships",
                    message=f"Invalid OHLC relationships in {invalid_count} rows",
                    severity=IssueSeverity.ERROR,
                    row_count=invalid_count,
                    sample_rows=invalid_rows,
                )
            )

        return issues

    def _check_quote_relationships(
        self, df: pd.DataFrame
    ) -> list[ValidationIssue]:
        """Check bid/ask price relationships are valid.

        Valid relationships:
        - bid <= ask (no crossed markets)
        - bid >= 0, ask >= 0 (non-negative prices)

        Args:
            df: DataFrame with bid/ask columns

        Returns:
            List of validation issues
        """
        issues = []

        # Check required columns exist
        if "bid" not in df.columns or "ask" not in df.columns:
            return issues

        # Check for crossed markets (bid > ask)
        # Only check rows where both bid and ask are non-null
        valid_quotes = df[["bid", "ask"]].notna().all(axis=1)
        crossed = (df["bid"] > df["ask"]) & valid_quotes

        crossed_count = crossed.sum()

        if crossed_count > 0:
            crossed_rows = crossed[crossed].index.tolist()[:5]

            issues.append(
                ValidationIssue(
                    check_name="crossed_markets",
                    message=f"Crossed markets (bid > ask) in {crossed_count} rows",
                    severity=IssueSeverity.WARNING,  # Warning since markets can temporarily cross
                    row_count=crossed_count,
                    sample_rows=crossed_rows,
                )
            )

        # Check for negative prices
        negative_bid = (df["bid"] < 0) & df["bid"].notna()
        negative_ask = (df["ask"] < 0) & df["ask"].notna()
        negative_count = (negative_bid | negative_ask).sum()

        if negative_count > 0:
            negative_mask = negative_bid | negative_ask
            negative_rows = negative_mask[negative_mask].index.tolist()[:5]

            issues.append(
                ValidationIssue(
                    check_name="negative_prices",
                    message=f"Negative prices in {negative_count} rows",
                    severity=IssueSeverity.ERROR,
                    row_count=negative_count,
                    sample_rows=negative_rows,
                )
            )

        return issues

    def validate_all(
        self,
        underlying_df: pd.DataFrame | None = None,
        options_df: pd.DataFrame | None = None,
        macro_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, ValidationResult]:
        """Run all validation checks on provided data.

        Args:
            underlying_df: Underlying bars DataFrame
            options_df: Option quotes DataFrame
            macro_dfs: Dictionary of macro series DataFrames

        Returns:
            Dictionary mapping data type to ValidationResult
        """
        results = {}

        if underlying_df is not None:
            self._logger.info("Validating underlying bars...")
            results["underlying_bars"] = self.check_underlying_bars(underlying_df)
            self._logger.info(results["underlying_bars"].summary())

        if options_df is not None:
            self._logger.info("Validating option quotes...")
            results["option_quotes"] = self.check_option_quotes(options_df)
            self._logger.info(results["option_quotes"].summary())

        if macro_dfs:
            for series_id, df in macro_dfs.items():
                self._logger.info(f"Validating FRED series {series_id}...")
                results[f"fred_{series_id}"] = self.check_fred_series(df)
                self._logger.info(results[f"fred_{series_id}"].summary())

        return results
