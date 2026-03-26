"""Surface builder orchestrator.

This module orchestrates the full surface construction pipeline, transforming
raw option quotes into structured IV surface snapshots.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src.config.constants import MarketConstants
from src.config.schema import SurfaceConfig, UniverseConfig
from src.data.storage.repository import DerivedRepository, RawRepository
from src.surface.buckets.assign import DeltaBucketAssigner
from src.surface.greeks.analytical import AnalyticalGreeks
from src.surface.iv.black_scholes import BlackScholesIVSolver
from src.surface.quality.filters import QualityConfig, QualityFilter


@dataclass
class BuildResult:
    """Result of a surface build operation.

    Attributes:
        build_run_id: Unique identifier for this build run
        version: Surface version string
        row_count: Number of surface snapshots created
        start: Start datetime of the build range
        end: End datetime of the build range
        quotes_processed: Total number of option quotes processed
        iv_failures: Number of options where IV solver failed
    """

    build_run_id: str
    version: str
    row_count: int
    start: datetime
    end: datetime
    quotes_processed: int = 0
    iv_failures: int = 0

    @property
    def iv_failure_ratio(self) -> float:
        """Ratio of IV solver failures to total quotes processed."""
        if self.quotes_processed == 0:
            return 0.0
        return self.iv_failures / self.quotes_processed


class SurfaceBuilder:
    """Orchestrates surface construction from raw quotes.

    This class implements the full surface construction pipeline:
    1. Load raw data (quotes, underlying bars, FRED rates)
    2. Join underlying price and risk-free rate (time-aware)
    3. Compute time-to-expiry (ACT/365)
    4. Calculate mid price and spread
    5. Invert IV using Newton-Raphson
    6. Compute Greeks
    7. Assign tenor bins and delta buckets
    8. Apply quality filters
    9. Select representative option per node
    10. Write surface snapshots to database

    The pipeline is deterministic: same inputs → same outputs.

    Example:
        config = SurfaceConfig()
        universe = UniverseConfig()
        builder = SurfaceBuilder(config, universe, raw_repo, derived_repo)
        result = builder.build_surface(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
            version="v1.0"
        )
    """

    def __init__(
        self,
        config: SurfaceConfig,
        universe: UniverseConfig,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
        quality_config: QualityConfig | None = None,
        rate_series_id: str = MarketConstants.DEFAULT_RATE_SERIES,
        close_hour_utc: int = MarketConstants.CLOSE_HOUR_UTC,
    ):
        """Initialize surface builder.

        Args:
            config: Surface configuration (buckets, tenors, BS settings)
            universe: Universe configuration (underlying symbol)
            raw_repo: Repository for raw data access
            derived_repo: Repository for derived data access
            quality_config: Quality filter configuration. If None, uses defaults.
            rate_series_id: FRED series ID for risk-free rate (default from MarketConstants)
            close_hour_utc: Hour in UTC when market closes (default from MarketConstants)
        """
        self._config = config
        self._universe = universe
        self._raw_repo = raw_repo
        self._derived_repo = derived_repo
        self._rate_series_id = rate_series_id
        self._close_hour_utc = close_hour_utc

        # Initialize components
        self._iv_solver = BlackScholesIVSolver(
            max_iters=config.black_scholes.max_iterations,
            tolerance=config.black_scholes.tolerance,
        )
        self._greeks_calculator = AnalyticalGreeks()

        # Convert DeltaBucketsConfig to dict for DeltaBucketAssigner
        bucket_dict = self._buckets_config_to_dict(config.delta_buckets)
        self._bucket_assigner = DeltaBucketAssigner(bucket_dict)

        # Quality filter with provided or default config
        self._quality_config = quality_config or QualityConfig()
        self._quality_filter = QualityFilter(self._quality_config)

    def _buckets_config_to_dict(self, buckets_config) -> dict[str, list[float]]:
        """Convert DeltaBucketsConfig to dictionary format.

        Args:
            buckets_config: Pydantic DeltaBucketsConfig model

        Returns:
            Dictionary mapping bucket names to delta ranges
        """
        # ATM should be first for proper |delta| matching
        return {
            "ATM": buckets_config.ATM,
            "P40": buckets_config.P40,
            "P25": buckets_config.P25,
            "P10": buckets_config.P10,
            "C10": buckets_config.C10,
            "C25": buckets_config.C25,
            "C40": buckets_config.C40,
        }

    def build_surface(
        self,
        start: datetime,
        end: datetime,
        version: str,
    ) -> BuildResult:
        """Build surface snapshots for date range.

        Processes quotes in daily sub-chunks to bound memory usage.
        Reference data (underlying bars, FRED rates) is loaded once and
        reused across all daily chunks.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            version: Surface version identifier (e.g., "v1.0")

        Returns:
            BuildResult with metadata about the build

        Raises:
            RuntimeError: If data loading or processing fails
        """
        build_run_id = self._generate_build_id()

        # Load reference data once (small datasets, safe to keep in memory)
        underlying_df = self._raw_repo.read_underlying_bars(
            symbol=self._universe.underlying,
            start=start - timedelta(days=1),  # Buffer for merge_asof
            end=end,
            timeframe="1m",
        )

        fred_df = self._raw_repo.read_fred_series(
            series_id=self._rate_series_id,
            start=start - timedelta(days=365),  # Buffer for lookback
            end=end,
        )

        # Process quotes in daily chunks to prevent OOM
        total_snapshots = 0
        total_iv_failures = 0
        total_quotes_processed = 0

        current = start
        while current < end:
            day_end = min(current + timedelta(days=1), end)

            quotes_df = self._raw_repo.read_option_quotes(current, day_end)

            current = day_end

            if quotes_df.empty:
                continue

            # Join daily volume from option bars (cbbo-1m quotes lack volume)
            bars_df = self._raw_repo.read_option_bars(
                day_end - timedelta(days=1), day_end, timeframe="1d",
            )
            if not bars_df.empty and "option_symbol" in bars_df.columns:
                daily_vol = (
                    bars_df.groupby("option_symbol")["volume"]
                    .sum()
                    .reset_index()
                    .rename(columns={"volume": "bar_volume"})
                )
                quotes_df = quotes_df.merge(daily_vol, on="option_symbol", how="left")
                quotes_df["volume"] = quotes_df["bar_volume"]
                quotes_df.drop(columns=["bar_volume"], inplace=True)
                del daily_vol
            del bars_df

            # Join daily OI from option statistics (stat_type=9)
            stats_df = self._raw_repo.read_option_statistics(
                day_end - timedelta(days=1), day_end, stat_type=9,
            )
            if not stats_df.empty and "option_symbol" in stats_df.columns:
                daily_oi = (
                    stats_df.groupby("option_symbol")["quantity"]
                    .last()
                    .reset_index()
                    .rename(columns={"quantity": "stats_oi"})
                )
                quotes_df = quotes_df.merge(daily_oi, on="option_symbol", how="left")
                quotes_df["open_interest"] = quotes_df["stats_oi"]
                quotes_df.drop(columns=["stats_oi"], inplace=True)
                del daily_oi
            del stats_df

            # Floor timestamps to the minute (CBBO-1m quotes have per-symbol
            # millisecond offsets within each 1-minute bar)
            quotes_df["ts_utc"] = pd.to_datetime(quotes_df["ts_utc"]).dt.floor("min")

            day_snapshots, day_iv_failures, day_quotes = self._process_and_write_chunk(
                quotes_df, underlying_df, fred_df, build_run_id, version,
            )
            del quotes_df

            total_snapshots += day_snapshots
            total_iv_failures += day_iv_failures
            total_quotes_processed += day_quotes

        return BuildResult(
            build_run_id=build_run_id,
            version=version,
            row_count=total_snapshots,
            start=start,
            end=end,
            quotes_processed=total_quotes_processed,
            iv_failures=total_iv_failures,
        )

    def _process_and_write_chunk(
        self,
        quotes_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
        fred_df: pd.DataFrame,
        build_run_id: str,
        version: str,
    ) -> tuple[int, int, int]:
        """Process a chunk of quotes through the full pipeline and write to DB.

        Args:
            quotes_df: Raw option quotes for this chunk
            underlying_df: Reference underlying bars (shared across chunks)
            fred_df: Reference FRED rates (shared across chunks)
            build_run_id: Build run identifier
            version: Surface version string

        Returns:
            Tuple of (snapshots_written, iv_failures, quotes_processed)
        """
        quotes_processed = len(quotes_df)

        # Join underlying price (time-aware)
        quotes_df = self._join_underlying_price(quotes_df, underlying_df)

        # Join risk-free rate (time-aware, release-aligned)
        quotes_df = self._join_risk_free_rate(quotes_df, fred_df)

        # Compute time-to-expiry
        quotes_df["tte_years"] = self._compute_tte(quotes_df)

        # Compute mid price and spread
        quotes_df["mid_price"] = (quotes_df["bid"] + quotes_df["ask"]) / 2
        quotes_df["spread"] = quotes_df["ask"] - quotes_df["bid"]
        quotes_df["spread_pct"] = quotes_df["spread"] / np.maximum(quotes_df["mid_price"], 1e-10)

        # Invert IV (vectorized)
        quotes_df["iv_mid"] = self._iv_solver.solve_iv_vectorized(
            prices=quotes_df["mid_price"],
            S=quotes_df["underlying_price"],
            K=quotes_df["strike"],
            T=quotes_df["tte_years"],
            r=quotes_df["rf_rate"],
            q=0.0,
            right=quotes_df["right"],
        )

        quotes_df["iv_bid"] = self._iv_solver.solve_iv_vectorized(
            prices=quotes_df["bid"],
            S=quotes_df["underlying_price"],
            K=quotes_df["strike"],
            T=quotes_df["tte_years"],
            r=quotes_df["rf_rate"],
            q=0.0,
            right=quotes_df["right"],
        )

        quotes_df["iv_ask"] = self._iv_solver.solve_iv_vectorized(
            prices=quotes_df["ask"],
            S=quotes_df["underlying_price"],
            K=quotes_df["strike"],
            T=quotes_df["tte_years"],
            r=quotes_df["rf_rate"],
            q=0.0,
            right=quotes_df["right"],
        )

        iv_failures = int(quotes_df["iv_mid"].isna().sum())

        # Compute Greeks (vectorized, only for valid IV rows)
        valid_iv_mask = quotes_df["iv_mid"].notna() & (quotes_df["iv_mid"] > 0)
        if valid_iv_mask.any():
            greeks = self._greeks_calculator.compute_greeks_vectorized(
                S=quotes_df.loc[valid_iv_mask, "underlying_price"],
                K=quotes_df.loc[valid_iv_mask, "strike"],
                T=quotes_df.loc[valid_iv_mask, "tte_years"],
                r=quotes_df.loc[valid_iv_mask, "rf_rate"],
                q=0.0,
                sigma=quotes_df.loc[valid_iv_mask, "iv_mid"],
                right=quotes_df.loc[valid_iv_mask, "right"],
            )
            quotes_df = quotes_df.join(greeks)
        else:
            for col in ["delta", "gamma", "vega", "theta"]:
                quotes_df[col] = np.nan

        # Assign tenor bins and delta buckets
        quotes_df["tenor_days"] = self._assign_tenor_bins(quotes_df)
        quotes_df["delta_bucket"] = self._bucket_assigner.assign(quotes_df["delta"])

        # Apply quality filters and flag
        quotes_df["flags"] = self._quality_filter.compute_flags(quotes_df)

        # Select representative option per node
        snapshots_df = self._select_representatives(quotes_df)
        del quotes_df

        snapshots_written = 0
        if not snapshots_df.empty:
            snapshot_columns = [
                "ts_utc", "option_symbol", "exp_date", "strike", "right",
                "bid", "ask", "mid_price", "spread", "spread_pct",
                "tte_years", "tenor_days", "underlying_price", "rf_rate",
                "iv_mid", "iv_bid", "iv_ask",
                "delta", "gamma", "vega", "theta",
                "delta_bucket", "flags", "volume", "open_interest",
            ]
            if "dividend_yield" not in snapshots_df.columns:
                snapshots_df = snapshots_df.copy()
                snapshots_df["dividend_yield"] = 0.0
                snapshot_columns.append("dividend_yield")
            else:
                snapshot_columns.append("dividend_yield")

            available_cols = [c for c in snapshot_columns if c in snapshots_df.columns]
            snapshots_df = snapshots_df[available_cols]

            self._derived_repo.write_surface_snapshots(snapshots_df, build_run_id, version)
            snapshots_written = len(snapshots_df)

        del snapshots_df
        return snapshots_written, iv_failures, quotes_processed

    def _generate_build_id(self) -> str:
        """Generate unique build run identifier.

        Returns:
            UUID string for this build run
        """
        return str(uuid.uuid4())

    def _join_underlying_price(
        self, quotes_df: pd.DataFrame, underlying_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Join most recent underlying close for each quote timestamp.

        Uses merge_asof with direction='backward' to get the most recent
        underlying bar at or before each quote timestamp.

        Args:
            quotes_df: DataFrame with option quotes (must have ts_utc column)
            underlying_df: DataFrame with underlying bars (must have ts_utc, close columns)

        Returns:
            quotes_df with 'underlying_price' column added
        """
        if underlying_df.empty:
            quotes_df["underlying_price"] = np.nan
            return quotes_df

        # Ensure datetime types
        quotes_df = quotes_df.copy()
        quotes_df["ts_utc"] = pd.to_datetime(quotes_df["ts_utc"])
        underlying_df = underlying_df.copy()
        underlying_df["ts_utc"] = pd.to_datetime(underlying_df["ts_utc"])

        # Sort for merge_asof
        quotes_df = quotes_df.sort_values("ts_utc")
        underlying_df = underlying_df.sort_values("ts_utc")

        # Merge: get most recent underlying close at or before quote timestamp
        merged = pd.merge_asof(
            quotes_df,
            underlying_df[["ts_utc", "close"]].rename(columns={"close": "underlying_price"}),
            on="ts_utc",
            direction="backward",
        )

        return merged

    def _join_risk_free_rate(
        self, quotes_df: pd.DataFrame, fred_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Join risk-free rate respecting release timestamps.

        Uses merge_asof to get the most recent FRED rate where the
        release_datetime_utc <= ts_utc (no future data leakage).

        Args:
            quotes_df: DataFrame with option quotes (must have ts_utc column)
            fred_df: DataFrame with FRED series (must have obs_date, value,
                     optionally release_datetime_utc columns)

        Returns:
            quotes_df with 'rf_rate' column added
        """
        if fred_df.empty:
            logger.info("No FRED data provided — using default rf_rate=0.05")
            quotes_df["rf_rate"] = 0.05  # Default fallback rate
            return quotes_df

        quotes_df = quotes_df.copy()

        obs_merge_ts = pd.to_datetime(fred_df["obs_date"], errors="coerce") + timedelta(hours=21)

        # Determine merge key - prefer release_datetime_utc if plausible
        if "release_datetime_utc" in fred_df.columns:
            fred_df = fred_df.copy()
            release_ts = pd.to_datetime(fred_df["release_datetime_utc"], errors="coerce")
            plausible = (
                release_ts.notna()
                & obs_merge_ts.notna()
                & (release_ts >= obs_merge_ts - timedelta(days=1))
                & (release_ts <= obs_merge_ts + timedelta(days=7))
            )
            fred_df["merge_ts"] = release_ts.where(plausible, obs_merge_ts)
            fred_df["merge_ts"] = fred_df["merge_ts"].fillna(obs_merge_ts)
        else:
            fred_df = fred_df.copy()
            fred_df["merge_ts"] = obs_merge_ts

        # Sort for merge_asof
        quotes_df = quotes_df.sort_values("ts_utc")
        fred_df = fred_df.sort_values("merge_ts")

        # Merge: get most recent rate at or before quote timestamp
        merged = pd.merge_asof(
            quotes_df,
            fred_df[["merge_ts", "value"]].rename(columns={"value": "rf_rate"}),
            left_on="ts_utc",
            right_on="merge_ts",
            direction="backward",
        )

        # Remove merge key
        if "merge_ts" in merged.columns:
            merged = merged.drop(columns=["merge_ts"])

        # Fill any missing rates with default
        n_missing = merged["rf_rate"].isna().sum()
        if n_missing > 0:
            logger.info(f"Filling {n_missing} missing rf_rate values with default 0.05")
        merged["rf_rate"] = merged["rf_rate"].fillna(0.05)

        return merged

    def _compute_tte(self, df: pd.DataFrame) -> pd.Series:
        """Compute time-to-expiry in years using ACT/365.

        TTE is computed as the time from the quote timestamp to the
        market close on the expiration date.

        Args:
            df: DataFrame with ts_utc and exp_date columns

        Returns:
            Series of time-to-expiry in years (ACT/365)
        """
        ts_utc = pd.to_datetime(df["ts_utc"])
        exp_date = pd.to_datetime(df["exp_date"])

        # Expiration is at market close (configured close hour in UTC)
        exp_datetime = exp_date + timedelta(hours=self._close_hour_utc)

        # TTE in days
        tte_days = (exp_datetime - ts_utc).dt.total_seconds() / 86400

        # Convert to years (ACT/365)
        tte_years = tte_days / 365.0

        # Ensure non-negative TTE (expired options get small positive value)
        tte_years = np.maximum(tte_years, 1e-6)

        return tte_years

    def _assign_tenor_bins(self, df: pd.DataFrame) -> pd.Series:
        """Assign each option to nearest tenor bin.

        For each option, finds the tenor bin from config that is closest
        to the option's DTE (days to expiration).

        Args:
            df: DataFrame with ts_utc and exp_date columns

        Returns:
            Series of tenor bin values (in days)
        """
        # Compute DTE
        ts_date = pd.to_datetime(df["ts_utc"]).dt.date
        exp_date = pd.to_datetime(df["exp_date"]).dt.date

        def compute_dte(t, e):
            """Compute days between dates, handling NaT."""
            if pd.isna(t) or pd.isna(e):
                return np.nan
            return (e - t).days

        dte_days = pd.Series(
            [compute_dte(t, e) for t, e in zip(ts_date, exp_date)],
            index=df.index,
        )

        # Get tenor bins from config
        tenor_bins = np.array(self._config.tenor_bins.bins)

        # Find nearest tenor for each option
        def find_nearest_tenor(dte):
            if pd.isna(dte) or dte < 0:
                return np.nan
            distances = np.abs(tenor_bins - dte)
            nearest_idx = np.argmin(distances)
            return tenor_bins[nearest_idx]

        tenor_days = dte_days.apply(find_nearest_tenor)

        return tenor_days

    def _select_representatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select one option per (ts_utc, tenor_days, delta_bucket).

        For each node (timestamp, tenor, delta bucket), selects the option
        whose delta is closest to the bucket center. Options with quality
        flags are deprioritized.

        Args:
            df: DataFrame with all processed option quotes

        Returns:
            DataFrame with one row per node (representative options)
        """
        # Filter out rows without valid assignments
        valid = df[
            df["delta_bucket"].notna() &
            df["tenor_days"].notna() &
            df["iv_mid"].notna()
        ].copy()

        if valid.empty:
            return pd.DataFrame()

        # Compute distance from bucket center
        # For ATM, center is 0.5 (|delta|); for others, use midpoint of range
        def get_bucket_center(bucket_name):
            if bucket_name == "ATM":
                return 0.5  # |delta| = 0.5
            else:
                min_d, max_d = self._bucket_assigner.get_bucket_bounds(bucket_name)
                return (min_d + max_d) / 2

        valid["bucket_center"] = valid["delta_bucket"].apply(get_bucket_center)

        # Distance to center (use |delta| for ATM)
        valid["delta_for_distance"] = np.where(
            valid["delta_bucket"] == "ATM",
            np.abs(valid["delta"]),
            valid["delta"],
        )
        valid["distance_to_center"] = np.abs(valid["delta_for_distance"] - valid["bucket_center"])

        # Deprioritize options with quality flags (add penalty)
        valid["selection_score"] = valid["distance_to_center"] + (valid["flags"] > 0).astype(float) * 100

        # Sort by ts_utc, tenor_days, delta_bucket, selection_score
        valid = valid.sort_values(
            ["ts_utc", "tenor_days", "delta_bucket", "selection_score"]
        )

        # Select first (best) option per node.
        # Use drop_duplicates (takes entire first row) instead of
        # groupby.first() which returns first non-null per column,
        # potentially mixing values from different options.
        representatives = valid.drop_duplicates(
            subset=["ts_utc", "tenor_days", "delta_bucket"], keep="first"
        )

        # Clean up helper columns
        drop_cols = ["bucket_center", "delta_for_distance", "distance_to_center", "selection_score"]
        representatives = representatives.drop(columns=[c for c in drop_cols if c in representatives.columns])

        return representatives

    def build_surface_snapshot(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Build surface from pre-loaded quotes (for live trading).

        This method allows building a surface from quotes already in memory,
        useful for live trading where data comes from real-time feeds.

        Args:
            quotes_df: DataFrame with option quotes (must have required columns)

        Returns:
            DataFrame with surface snapshot (one row per node)
        """
        if quotes_df.empty:
            return pd.DataFrame()

        # This method assumes underlying_price and rf_rate are already joined
        # or need to be provided in the quotes_df

        # Compute TTE if not present
        if "tte_years" not in quotes_df.columns:
            quotes_df = quotes_df.copy()
            quotes_df["tte_years"] = self._compute_tte(quotes_df)

        # Compute mid price and spread
        quotes_df = quotes_df.copy()
        quotes_df["mid_price"] = (quotes_df["bid"] + quotes_df["ask"]) / 2
        quotes_df["spread"] = quotes_df["ask"] - quotes_df["bid"]
        quotes_df["spread_pct"] = quotes_df["spread"] / np.maximum(quotes_df["mid_price"], 1e-10)

        # Check for required columns
        required_cols = ["underlying_price", "rf_rate"]
        for col in required_cols:
            if col not in quotes_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Invert IV
        quotes_df["iv_mid"] = self._iv_solver.solve_iv_vectorized(
            prices=quotes_df["mid_price"],
            S=quotes_df["underlying_price"],
            K=quotes_df["strike"],
            T=quotes_df["tte_years"],
            r=quotes_df["rf_rate"],
            q=0.0,
            right=quotes_df["right"],
        )

        # Compute Greeks
        valid_iv_mask = quotes_df["iv_mid"].notna() & (quotes_df["iv_mid"] > 0)
        if valid_iv_mask.any():
            greeks = self._greeks_calculator.compute_greeks_vectorized(
                S=quotes_df.loc[valid_iv_mask, "underlying_price"],
                K=quotes_df.loc[valid_iv_mask, "strike"],
                T=quotes_df.loc[valid_iv_mask, "tte_years"],
                r=quotes_df.loc[valid_iv_mask, "rf_rate"],
                q=0.0,
                sigma=quotes_df.loc[valid_iv_mask, "iv_mid"],
                right=quotes_df.loc[valid_iv_mask, "right"],
            )
            quotes_df = quotes_df.join(greeks)
        else:
            for col in ["delta", "gamma", "vega", "theta"]:
                quotes_df[col] = np.nan

        # Assign tenor bins and delta buckets
        quotes_df["tenor_days"] = self._assign_tenor_bins(quotes_df)
        quotes_df["delta_bucket"] = self._bucket_assigner.assign(quotes_df["delta"])

        # Quality flags
        quotes_df["flags"] = self._quality_filter.compute_flags(quotes_df)

        # Select representatives
        return self._select_representatives(quotes_df)
