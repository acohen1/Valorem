"""Feature validation for anti-leakage checks.

This module provides validation utilities to ensure features
don't leak future information into the model.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd


class IssueSeverity(Enum):
    """Severity level for validation issues."""

    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue.

    Attributes:
        severity: Issue severity level
        message: Human-readable description
        column: Column name (if applicable)
        details: Additional details
    """

    severity: IssueSeverity
    message: str
    column: str | None = None
    details: dict | None = None


@dataclass
class ValidationResult:
    """Result of feature validation.

    Attributes:
        passed: Whether all checks passed
        issues: List of validation issues found
        checked_at: Timestamp of validation
    """

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureValidator:
    """Validate features for data leakage.

    This class performs anti-leakage validation to ensure:
    1. Rolling window features have appropriate NaN values at the start
    2. Macro features respect release timestamps
    3. No future data is used in feature computation

    Example:
        validator = FeatureValidator()
        result = validator.validate_no_future_leakage(panel_df)
        if not result.passed:
            for issue in result.issues:
                print(f"{issue.severity.value}: {issue.message}")
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the validator.

        Args:
            logger: Optional logger instance
        """
        self._logger = logger or logging.getLogger(__name__)

    def validate_no_future_leakage(
        self,
        panel_df: pd.DataFrame,
        reference_df: pd.DataFrame | None = None,
        lookback_buffered: bool = False,
    ) -> ValidationResult:
        """Validate that features don't leak future data.

        Performs several checks:
        1. Rolling window features should have NaN in early rows
        2. Change features should have NaN for rows < period
        3. If reference_df is provided, validates macro release times

        Args:
            panel_df: Feature panel DataFrame
            reference_df: Optional reference DataFrame with release timestamps
            lookback_buffered: If True, skip rolling/change NaN checks because
                a lookback buffer already warmed up the rolling windows before
                the output was trimmed.

        Returns:
            ValidationResult with pass/fail and issues
        """
        issues = []

        if panel_df.empty:
            return ValidationResult(passed=True, issues=[])

        # Checks 1-2 only apply when data starts from scratch (no lookback buffer).
        # When a lookback buffer is used, rolling windows are pre-warmed and the
        # trimmed output legitimately has no early NaNs.
        if not lookback_buffered:
            # Check 1: Rolling window features should have NaN in early rows
            rolling_issues = self._check_rolling_window_nans(panel_df)
            issues.extend(rolling_issues)

            # Check 2: Change features should have appropriate NaN counts
            change_issues = self._check_change_feature_nans(panel_df)
            issues.extend(change_issues)

        # Check 3: Check for any completely non-NaN columns that should have NaNs
        unexpected_issues = self._check_unexpected_complete_columns(panel_df)
        issues.extend(unexpected_issues)

        # Check 4: Validate macro release time alignment if reference provided
        if reference_df is not None:
            macro_issues = self._check_macro_release_alignment(panel_df, reference_df)
            issues.extend(macro_issues)

        # Determine pass/fail based on error-level issues
        has_errors = any(i.severity == IssueSeverity.ERROR for i in issues)

        if issues:
            error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
            warning_count = len(issues) - error_count
            if error_count:
                self._logger.warning(f"Validation: {error_count} errors, {warning_count} warnings")
            else:
                self._logger.info(f"Validation passed with {warning_count} warnings")

        return ValidationResult(
            passed=not has_errors,
            issues=issues,
        )

    def _check_rolling_window_nans(
        self,
        df: pd.DataFrame,
    ) -> list[ValidationIssue]:
        """Check that rolling window features have NaN in early rows.

        Args:
            df: Feature panel DataFrame

        Returns:
            List of validation issues
        """
        issues = []

        # Pattern to detect rolling window features with window size
        # e.g., iv_vol_5d, volume_ratio_5d, rv_21d
        rolling_pattern = re.compile(r"^(\w+)_(\d+)d$")

        if self._is_node_level(df):
            node_groups = self._iter_node_groups(df)
            for col in df.columns:
                match = rolling_pattern.match(col)
                if not match:
                    continue
                window_days = int(match.group(2))
                flagged_nodes = 0
                for node_df in node_groups:
                    steps_per_day = self._infer_steps_per_day(node_df.get("ts_utc"))
                    window = window_days * steps_per_day
                    if window <= 2 or len(node_df) < window:
                        continue
                    first_few = node_df[col].iloc[:max(1, window // 2)]
                    if first_few.isna().sum() == 0:
                        flagged_nodes += 1
                if flagged_nodes > 0:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"Rolling feature {col} has no early NaNs in "
                            f"{flagged_nodes} node series"
                        ),
                        column=col,
                        details={"flagged_nodes": flagged_nodes, "window_days": window_days},
                    ))
            return issues

        steps_per_day = self._infer_steps_per_day(df.get("ts_utc"))

        for col in df.columns:
            match = rolling_pattern.match(col)
            if match:
                window = int(match.group(2)) * steps_per_day

                if len(df) >= window:
                    first_few = df[col].iloc[:max(1, window // 2)]
                    nan_count = first_few.isna().sum()
                    if nan_count == 0 and window > 2:
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            message=f"Rolling feature {col} (window={window}) has no NaN in first {len(first_few)} rows",
                            column=col,
                            details={"window": window, "first_nan_count": nan_count},
                        ))

        return issues

    def _check_change_feature_nans(
        self,
        df: pd.DataFrame,
    ) -> list[ValidationIssue]:
        """Check that change features have appropriate NaN counts.

        Args:
            df: Feature panel DataFrame

        Returns:
            List of validation issues
        """
        issues = []

        # Pattern for change features
        # e.g., iv_change_1d, iv_change_5d, oi_change_5d
        change_pattern = re.compile(r"^(\w+)_change_(\d+)([dwm])$")

        if self._is_node_level(df):
            node_groups = self._iter_node_groups(df)
            for col in df.columns:
                match = change_pattern.match(col)
                if not match:
                    continue
                period = int(match.group(2))
                unit = match.group(3)
                period_days = self._period_to_days(period, unit)
                flagged_nodes = 0
                for node_df in node_groups:
                    steps_per_day = self._infer_steps_per_day(node_df.get("ts_utc"))
                    period_steps = period_days * steps_per_day
                    if period_steps <= 0 or len(node_df) < period_steps:
                        continue
                    first_vals = node_df[col].iloc[:period_steps]
                    nan_ratio = first_vals.isna().sum() / len(first_vals)
                    if nan_ratio < 0.5:
                        flagged_nodes += 1
                if flagged_nodes > 0:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"Change feature {col} has low early NaN ratio in "
                            f"{flagged_nodes} node series"
                        ),
                        column=col,
                        details={"flagged_nodes": flagged_nodes, "period_days": period_days},
                    ))
            return issues

        steps_per_day = self._infer_steps_per_day(df.get("ts_utc"))

        for col in df.columns:
            match = change_pattern.match(col)
            if match:
                period = int(match.group(2))
                unit = match.group(3)
                period_days = self._period_to_days(period, unit)
                period_steps = period_days * steps_per_day

                if len(df) >= period_steps:
                    first_vals = df[col].iloc[:period_steps]
                    nan_ratio = first_vals.isna().sum() / len(first_vals)
                    if nan_ratio < 0.5:  # At least half should be NaN
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            message=f"Change feature {col} (period={period_days}d) has {nan_ratio:.0%} NaN in first {period_steps} rows",
                            column=col,
                            details={
                                "period_days": period_days,
                                "period_steps": period_steps,
                                "nan_ratio": nan_ratio,
                            },
                        ))

        return issues

    @staticmethod
    def _is_node_level(df: pd.DataFrame) -> bool:
        """Return True if DataFrame contains per-node panel columns."""
        return "tenor_days" in df.columns and "delta_bucket" in df.columns

    @staticmethod
    def _iter_node_groups(df: pd.DataFrame) -> list[pd.DataFrame]:
        """Return node-level groups sorted in time order."""
        grouped = (
            df.sort_values(["tenor_days", "delta_bucket", "ts_utc"])
            .groupby(["tenor_days", "delta_bucket"], sort=False)
        )
        return [group for _, group in grouped]

    @staticmethod
    def _period_to_days(period: int, unit: str) -> int:
        """Convert period/unit notation to day count."""
        if unit == "w":
            return period * 7
        if unit == "m":
            return period * 30
        return period

    @staticmethod
    def _infer_steps_per_day(ts: pd.Series | None) -> int:
        """Infer median observations per trading day."""
        if ts is None or ts.empty:
            return 1
        counts = pd.to_datetime(ts).dt.normalize().value_counts()
        if counts.empty:
            return 1
        return max(1, int(np.median(counts.values)))

    def _check_unexpected_complete_columns(
        self,
        df: pd.DataFrame,
    ) -> list[ValidationIssue]:
        """Check for columns that should have NaN but don't.

        Args:
            df: Feature panel DataFrame

        Returns:
            List of validation issues
        """
        issues = []

        # Node-level panels use a lookback buffer that warms all rolling windows,
        # so 0 NaN in the trimmed output is expected — skip this check.
        if "tenor_days" in df.columns and "delta_bucket" in df.columns:
            return issues

        # Feature patterns that should have some NaN
        should_have_nan_patterns = [
            r".*_vol_\d+d$",      # Rolling volatility
            r".*_ma_\d+d$",       # Moving averages
            r".*_std_\d+d$",      # Rolling std
            r".*_zscore.*$",      # Z-scores
            r"^underlying_rv_\d+d$",  # Realized variance (renamed from rv_*)
            r".*_change_\d+.*$",  # Change features
        ]

        for col in df.columns:
            for pattern in should_have_nan_patterns:
                if re.match(pattern, col):
                    nan_count = df[col].isna().sum()
                    if nan_count == 0 and len(df) > 10:
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            message=f"Feature {col} has no NaN values (expected some for rolling/change features)",
                            column=col,
                        ))
                    break

        return issues

    def _check_macro_release_alignment(
        self,
        panel_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> list[ValidationIssue]:
        """Check that macro features respect release timestamps.

        Args:
            panel_df: Feature panel DataFrame
            reference_df: Reference DataFrame with obs_date and release_datetime_utc

        Returns:
            List of validation issues
        """
        issues = []

        # Find macro feature columns
        macro_cols = [c for c in panel_df.columns if any(
            series in c for series in ["DGS", "VIX", "FRED", "_level", "_zscore"]
        )]

        if not macro_cols:
            return issues

        if "release_datetime_utc" not in reference_df.columns:
            return issues

        # Check that macro features don't appear before their release time
        # This is a simplified check - in practice would cross-reference
        # each macro value with its original release timestamp

        # For now, just check that macro features have some NaN in early rows
        # (indicating proper time alignment)
        for col in macro_cols:
            if col in panel_df.columns:
                nan_count = panel_df[col].isna().sum()
                total_count = len(panel_df)
                if nan_count == 0 and total_count > 0:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        message=f"Macro feature {col} has no NaN values - verify release time alignment",
                        column=col,
                    ))

        return issues

    def validate_feature_ranges(
        self,
        panel_df: pd.DataFrame,
        expected_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> ValidationResult:
        """Validate that features are within expected ranges.

        Args:
            panel_df: Feature panel DataFrame
            expected_ranges: Dict mapping column names to (min, max) tuples

        Returns:
            ValidationResult
        """
        issues = []

        if expected_ranges is None:
            # Default expected ranges for common features
            expected_ranges = {
                "delta": (-1.0, 1.0),
                "gamma": (0.0, 10.0),
                "vega": (0.0, 2.0),
                "theta": (-500.0, 0.0),
                "spread_pct": (0.0, 1.0),  # Spread between 0% and 100%
            }

        for col, (min_val, max_val) in expected_ranges.items():
            if col in panel_df.columns:
                values = panel_df[col].dropna()
                if len(values) > 0:
                    actual_min = values.min()
                    actual_max = values.max()

                    if actual_min < min_val or actual_max > max_val:
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            message=f"Feature {col} has values outside expected range [{min_val}, {max_val}]",
                            column=col,
                            details={
                                "actual_min": actual_min,
                                "actual_max": actual_max,
                                "expected_min": min_val,
                                "expected_max": max_val,
                            },
                        ))

        return ValidationResult(
            passed=not any(i.severity == IssueSeverity.ERROR for i in issues),
            issues=issues,
        )

    def validate_feature_completeness(
        self,
        panel_df: pd.DataFrame,
        required_features: list[str] | None = None,
        max_nan_ratio: float = 0.5,
    ) -> ValidationResult:
        """Validate that required features are present and not too sparse.

        Args:
            panel_df: Feature panel DataFrame
            required_features: List of required feature column names
            max_nan_ratio: Maximum allowed NaN ratio per column

        Returns:
            ValidationResult
        """
        issues = []

        # Check required features exist
        if required_features:
            missing = set(required_features) - set(panel_df.columns)
            if missing:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    message=f"Missing required features: {missing}",
                    details={"missing_features": list(missing)},
                ))

        # Check NaN ratio for numeric columns
        for col in panel_df.select_dtypes(include=[np.number]).columns:
            nan_ratio = panel_df[col].isna().sum() / len(panel_df)
            if nan_ratio > max_nan_ratio:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    message=f"Feature {col} has {nan_ratio:.1%} NaN values (threshold: {max_nan_ratio:.1%})",
                    column=col,
                    details={"nan_ratio": nan_ratio, "threshold": max_nan_ratio},
                ))

        return ValidationResult(
            passed=not any(i.severity == IssueSeverity.ERROR for i in issues),
            issues=issues,
        )
