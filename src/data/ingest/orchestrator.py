"""Ingestion orchestrator for coordinating multi-source data ingestion.

This module coordinates the ingestion of market data, option quotes, and macro
data from multiple providers into the database. It handles cost estimation,
data validation, and logging.
"""

import logging
import re
import warnings
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config.schema import ConfigSchema
from src.exceptions import ProviderError
from src.data.ingest.manifest import (
    Manifest,
    ManifestGenerator,
    get_spot_reference,
)
from src.data.providers.protocol import MacroDataProvider, MarketDataProvider
from src.data.quality.validators import (
    DataQualityValidator,
    IssueSeverity,
    ValidationResult,
)
from src.data.storage.repository import RawRepository


class CostExceededException(Exception):
    """Raised when estimated cost exceeds configured limit."""

    pass


class DataQualityException(Exception):
    """Raised when data fails quality validation."""

    pass


@dataclass
class IngestionResult:
    """Result of an ingestion run."""

    run_id: str
    start_time: datetime
    end_time: datetime
    data_start: datetime | None = None  # Data date range start
    data_end: datetime | None = None  # Data date range end (may be adjusted)
    preview_only: bool = False
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    underlying_rows: int = 0
    option_rows: int = 0
    option_bar_rows: int = 0
    statistics_rows: int = 0
    macro_rows: int = 0
    skipped_underlying: bool = False
    skipped_chunks: int = 0
    skipped_series: int = 0
    failed_chunks: list[str] = field(default_factory=list)
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    adjusted_end_date: datetime | None = None  # Set if date range was truncated due to data availability
    cost_limit_exceeded: bool = False  # Set if estimated cost exceeds configured limit

    @property
    def total_rows(self) -> int:
        """Total rows ingested across all data types."""
        return self.underlying_rows + self.option_rows + self.option_bar_rows + self.statistics_rows + self.macro_rows

    @property
    def success(self) -> bool:
        """Check if ingestion completed without errors."""
        return len(self.errors) == 0 and not self.preview_only

    @property
    def partial_success(self) -> bool:
        """Check if some data was written despite errors."""
        return len(self.errors) > 0 and self.total_rows > 0

    def summary(self) -> str:
        """Generate summary of ingestion result."""
        if self.preview_only:
            return f"Preview: estimated cost ${self.estimated_cost:.2f}"

        if self.success:
            status = "SUCCESS"
        elif self.partial_success:
            status = "PARTIAL_SUCCESS"
        else:
            status = "FAILED"

        # Date range string (use adjusted end if available)
        date_range = ""
        if self.data_start and self.data_end:
            start_str = self.data_start.date()
            end_str = (self.adjusted_end_date or self.data_end).date()
            date_range = f"[{start_str} to {end_str}] "

        skipped_parts = []
        if self.skipped_underlying:
            skipped_parts.append("underlying")
        if self.skipped_chunks > 0:
            skipped_parts.append(f"{self.skipped_chunks} option chunks")
        if self.skipped_series > 0:
            skipped_parts.append(f"{self.skipped_series} FRED series")
        skipped_msg = f", skipped=[{', '.join(skipped_parts)}]" if skipped_parts else ""

        failed_msg = f", failed_chunks={len(self.failed_chunks)}" if self.failed_chunks else ""

        return (
            f"Ingestion {status} (run_id={self.run_id[:8]}): "
            f"{date_range}"
            f"{self.total_rows} total rows "
            f"(underlying={self.underlying_rows}, options={self.option_rows}, "
            f"option_bars={self.option_bar_rows}, statistics={self.statistics_rows}, "
            f"macro={self.macro_rows}){skipped_msg}{failed_msg}, cost=${self.actual_cost:.2f}"
        )


class IngestionOrchestrator:
    """Coordinates multi-source data ingestion runs.

    Manages the full ingestion workflow:
    1. Generate or load options manifest
    2. Estimate costs (reject if over limit)
    3. Fetch data from providers
    4. Validate data quality
    5. Write to storage
    6. Log ingestion metadata

    Attributes:
        _market: Market data provider (Databento, mock, etc.)
        _macro: Macro data provider (FRED, mock, etc.)
        _repo: Raw data repository for storage
        _config: Application configuration
    """

    def __init__(
        self,
        market_provider: MarketDataProvider,
        macro_provider: MacroDataProvider,
        repository: RawRepository,
        config: ConfigSchema,
        logger: logging.Logger | None = None,
    ):
        """Initialize ingestion orchestrator.

        Args:
            market_provider: Market data provider instance
            macro_provider: Macro data provider instance
            repository: Raw data repository for storage
            config: Application configuration
            logger: Optional logger instance
        """
        self._market = market_provider
        self._macro = macro_provider
        self._repo = repository
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._validator = DataQualityValidator(logger=self._logger)
        self._manifest_generator = ManifestGenerator(config=config, logger=self._logger)
        self._adjusted_end_date: datetime | None = None  # Track if date range was adjusted

    def run_ingestion(
        self,
        start: datetime,
        end: datetime,
        preview_only: bool = False,
        skip_validation: bool = False,
        manifest_path: Path | None = None,
        force: bool = False,
        skip_cost_check: bool = False,
        fred_only: bool = False,
        quotes_only: bool = False,
        stats_only: bool = False,
    ) -> IngestionResult:
        """Execute full ingestion workflow.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            preview_only: If True, only estimate costs without fetching data
            skip_validation: If True, skip data quality validation
            manifest_path: Optional path to pre-generated manifest
            force: If True, re-fetch all data even if already ingested
            skip_cost_check: If True, skip cost estimation (when already done)
            fred_only: If True, skip all Databento steps and only fetch FRED series
            quotes_only: If True, only fetch option quotes (skip underlying bars, option bars, statistics, and FRED)
            stats_only: If True, only fetch option statistics/OI (skip underlying bars, option bars, quotes, and FRED)

        Returns:
            IngestionResult with details of the ingestion run

        Raises:
            CostExceededException: If estimated cost exceeds configured limit
            DataQualityException: If data fails validation (and skip_validation=False)
        """
        run_id = self._generate_run_id()
        start_time = datetime.now(UTC)
        errors: list[str] = []
        self._adjusted_end_date = None  # Reset adjusted date tracker

        self._logger.info(
            f"Starting ingestion run {run_id[:8]} "
            f"from {start.date()} to {end.date()}"
        )

        # Initialize variables produced by Databento steps (0-4b)
        underlying_df = pd.DataFrame()
        skipped_underlying = False
        option_rows_written = 0
        option_bar_rows_written = 0
        statistics_rows_written = 0
        chunk_validation_results: list[ValidationResult] = []
        skipped_chunks = 0
        failed_chunks: list[str] = []
        cost_estimate = 0.0
        cost_limit_exceeded = False
        chunk_manifests: dict[pd.Timestamp, Manifest] = {}

        if fred_only:
            self._logger.info("--fred-only: skipping all Databento steps")

        if quotes_only:
            self._logger.info("--quotes-only: fetching option quotes only")

        if stats_only:
            self._logger.info("--stats-only: fetching option statistics only")

        # Step 0: Detect maximum available data date by probing the end of the range
        # This prevents querying beyond available data limits
        max_available_date = self._detect_max_available_date(end) if not fred_only else None
        if max_available_date and max_available_date < end:
            self._adjusted_end_date = max_available_date
            self._logger.warning(
                f"Data availability limited to {max_available_date.date()}. "
                f"Requested end date {end.date()} will be truncated."
            )

        # Step 1: Compute chunk boundaries and generate per-chunk manifests.
        # Each chunk gets its own manifest with symbols resolved relative to
        # that chunk's start date, so a 5-month ingestion discovers options
        # with later expirations as chunks advance.
        chunk_starts: list[pd.Timestamp] = []
        chunk_ends: list[pd.Timestamp] = []
        all_option_symbols: set[str] = set()

        if not fred_only:
            chunk_starts = pd.date_range(start, end, freq="MS").tolist()
            if not chunk_starts or chunk_starts[0] > pd.Timestamp(start):
                chunk_starts.insert(0, pd.Timestamp(start))
            chunk_ends = chunk_starts[1:] + [pd.Timestamp(end)]

            for cs, ce in zip(chunk_starts, chunk_ends):
                if cs >= ce:
                    continue
                manifest = self._load_or_generate_manifest(
                    cs.to_pydatetime(), ce.to_pydatetime(), manifest_path, force=force,
                    max_available_date=max_available_date
                )
                chunk_manifests[cs] = manifest
                all_option_symbols.update(manifest.symbols)

            # If date range was adjusted during manifest generation, update the last chunk_end
            if self._adjusted_end_date is not None:
                if chunk_ends:
                    original_end = chunk_ends[-1]
                    adjusted_ts = pd.Timestamp(self._adjusted_end_date)
                    if adjusted_ts < original_end:
                        chunk_ends[-1] = adjusted_ts
                        self._logger.info(
                            f"Updated last chunk end from {original_end.date()} to {adjusted_ts.date()}"
                        )

        if not fred_only:
            total_unique_symbols = len(all_option_symbols)
            self._logger.info(
                f"Generated {len(chunk_manifests)} chunk manifests: "
                f"{total_unique_symbols} unique option symbols"
            )

        # Step 2: Estimate costs
        # Skip if cost check already performed (e.g., during interactive confirmation)
        # (Also skipped entirely with --fred-only)

        if not skip_cost_check and not fred_only:
            # Use adjusted end date if data availability was limited
            effective_end = self._adjusted_end_date if self._adjusted_end_date else end

            underlying_cost = 0.0
            quotes_cost = 0.0
            bars_cost = 0.0
            stats_cost = 0.0

            if not quotes_only and not stats_only:
                try:
                    underlying_cost = self._market.estimate_cost(
                        dataset=self._config.data.providers.databento.dataset_equities,
                        schema="ohlcv-1m",
                        symbols=[self._config.universe.underlying],
                        start=start,
                        end=effective_end,
                    )
                except Exception as e:
                    self._logger.warning(f"Could not estimate underlying cost: {e}")

            for cs, manifest in chunk_manifests.items():
                if not manifest.symbols:
                    continue
                ce = chunk_ends[chunk_starts.index(cs)]
                chunk_symbols = self._filter_symbols_for_chunk(manifest, cs.date())
                if not chunk_symbols:
                    continue

                symbol_batches = self._batch_symbols(chunk_symbols, max_batch_size=2000)

                if not stats_only:
                    for batch in symbol_batches:
                        try:
                            quotes_cost += self._market.estimate_cost(
                                dataset=self._config.data.providers.databento.dataset_options,
                                schema=self._config.data.ingestion.databento.options.quote_schema,
                                symbols=batch,
                                start=cs.to_pydatetime(),
                                end=ce.to_pydatetime(),
                            )
                        except Exception as e:
                            self._logger.warning(
                                f"Could not estimate quotes cost for {cs.date()}: {e}"
                            )

                if not quotes_only and not stats_only:
                    for batch in symbol_batches:
                        try:
                            bars_cost += self._market.estimate_cost(
                                dataset=self._config.data.providers.databento.dataset_options,
                                schema=self._config.data.ingestion.databento.options.bar_schema,
                                symbols=batch,
                                start=cs.to_pydatetime(),
                                end=ce.to_pydatetime(),
                            )
                        except Exception as e:
                            self._logger.warning(
                                f"Could not estimate bars cost for {cs.date()}: {e}"
                            )

                if not quotes_only:
                    for batch in symbol_batches:
                        try:
                            stats_cost += self._market.estimate_cost(
                                dataset=self._config.data.providers.databento.dataset_options,
                                schema=self._config.data.ingestion.databento.options.statistics_schema,
                                symbols=batch,
                                start=cs.to_pydatetime(),
                                end=ce.to_pydatetime(),
                            )
                        except Exception as e:
                            self._logger.warning(
                                f"Could not estimate statistics cost for {cs.date()}: {e}"
                            )

            cost_estimate = underlying_cost + quotes_cost + bars_cost + stats_cost
            self._logger.info(
                f"Estimated cost: ${cost_estimate:.2f} "
                f"(equity_bars=${underlying_cost:.2f}, "
                f"option_quotes=${quotes_cost:.2f}, "
                f"option_bars=${bars_cost:.2f}, "
                f"statistics=${stats_cost:.2f})"
            )

            # Check cost limit
            max_cost = self._config.data.ingestion.databento.cost.max_usd
            cost_limit_exceeded = cost_estimate > max_cost
            if cost_limit_exceeded:
                if preview_only:
                    # In preview mode, just warn - don't crash
                    self._logger.warning(
                        f"Estimated cost ${cost_estimate:.2f} exceeds configured limit ${max_cost:.2f}"
                    )
                else:
                    # In actual ingestion, enforce the limit
                    raise CostExceededException(
                        f"Estimated cost ${cost_estimate:.2f} exceeds limit ${max_cost:.2f}"
                    )
        else:
            self._logger.info("Skipping cost estimation (already confirmed)")

        # If preview only, return early
        if preview_only:
            return IngestionResult(
                run_id=run_id,
                start_time=start_time,
                end_time=datetime.now(UTC),
                data_start=start,
                data_end=end,
                preview_only=True,
                estimated_cost=cost_estimate,
                adjusted_end_date=self._adjusted_end_date,
                cost_limit_exceeded=cost_limit_exceeded,
            )

        # Step 3: Fetch underlying bars
        if not fred_only and not quotes_only and not stats_only and not force:
            existing_bars = self._repo.count_underlying_bars(
                self._config.universe.underlying, start, end, timeframe="1m",
            )
            if existing_bars > 0:
                self._logger.info(
                    f"Found {existing_bars:,} existing underlying rows; re-fetching "
                    "to avoid preserving partial chunks"
                )

        if not fred_only and not quotes_only and not stats_only:
            try:
                self._logger.info("Fetching underlying bars...")
                underlying_df = self._fetch_with_retry(
                    lambda: self._market.fetch_underlying_bars(
                        symbol=self._config.universe.underlying,
                        start=start,
                        end=end,
                        interval="1m",
                    ),
                    "Underlying bars fetch",
                )
                self._logger.info(f"Fetched {len(underlying_df)} underlying bars")
            except Exception as e:
                self._logger.error(f"Failed to fetch underlying bars: {e}")
                errors.append(f"Underlying fetch error: {e}")

        # Step 4: Fetch option quotes (chunked by month, validate and write each immediately)
        if not fred_only and not stats_only and all_option_symbols:
            for i, (cs, ce) in enumerate(zip(chunk_starts, chunk_ends)):
                if cs >= ce:
                    continue
                manifest = chunk_manifests.get(cs)
                if not manifest or not manifest.symbols:
                    self._logger.info(
                        f"Skipping quotes chunk {i+1}/{len(chunk_starts)}: "
                        f"no active symbols for {cs.date()} to {ce.date()}"
                    )
                    continue
                # Filter out symbols that expired before this chunk starts
                chunk_symbols = self._filter_symbols_for_chunk(manifest, cs.date())
                if not chunk_symbols:
                    self._logger.info(
                        f"Skipping quotes chunk {i+1}/{len(chunk_starts)}: "
                        f"no active symbols after expiry filter for "
                        f"{cs.date()} to {ce.date()}"
                    )
                    continue
                chunk_desc = f"{cs.date()} to {ce.date()}"
                try:
                    # Databento has a 2,000 symbol limit per request
                    # If chunk has more symbols, split into batches
                    symbol_batches = self._batch_symbols(chunk_symbols, max_batch_size=2000)

                    if len(symbol_batches) > 1:
                        self._logger.info(
                            f"Fetching quotes chunk {i+1}/{len(chunk_starts)}: "
                            f"{chunk_desc} ({len(chunk_symbols)} symbols in {len(symbol_batches)} batches)..."
                        )
                    else:
                        self._logger.info(
                            f"Fetching quotes chunk {i+1}/{len(chunk_starts)}: "
                            f"{chunk_desc} ({len(chunk_symbols)} symbols)..."
                        )

                    chunk_batch_dfs = []
                    # Filter duplicate BentoWarnings within a chunk (same warning
                    # fires for each symbol batch but the message is identical).
                    with warnings.catch_warnings():
                        warnings.simplefilter("once")
                        for batch_idx, symbol_batch in enumerate(symbol_batches, 1):
                            self._logger.info(
                                f"  Batch {batch_idx}/{len(symbol_batches)}: "
                                f"downloading {len(symbol_batch)} symbols..."
                            )

                            batch_start = time.monotonic()
                            batch_df = self._fetch_with_retry(
                                lambda _cs=cs, _ce=ce, _syms=symbol_batch: self._market.fetch_option_quotes(
                                    symbols=_syms,
                                    start=_cs.to_pydatetime(),
                                    end=_ce.to_pydatetime(),
                                    schema=self._config.data.ingestion.databento.options.quote_schema,
                                ),
                                f"Quotes chunk {i+1}/{len(chunk_starts)} batch {batch_idx}/{len(symbol_batches)} ({chunk_desc})",
                            )
                            elapsed = time.monotonic() - batch_start
                            if not batch_df.empty:
                                self._logger.info(
                                    f"  Batch {batch_idx}/{len(symbol_batches)}: "
                                    f"{len(batch_df):,} rows in {elapsed:.0f}s"
                                )
                                chunk_batch_dfs.append(batch_df)
                            else:
                                self._logger.info(
                                    f"  Batch {batch_idx}/{len(symbol_batches)}: "
                                    f"empty ({elapsed:.0f}s)"
                                )

                    # Combine batches, validate, write, and free memory for this chunk
                    if chunk_batch_dfs:
                        chunk_df = pd.concat(chunk_batch_dfs, ignore_index=True)
                        del chunk_batch_dfs

                        # Per-chunk validation
                        if not skip_validation:
                            chunk_vr = self._validator.check_option_quotes(chunk_df)
                            chunk_validation_results.append(chunk_vr)
                            if chunk_vr.has_errors:
                                for issue in chunk_vr.issues:
                                    self._logger.warning(
                                        f"  [chunk {i+1}] {issue.check_name}: {issue.message}"
                                    )

                        # Prepare metadata and write to DB immediately
                        chunk_df = self._prepare_option_quotes(chunk_df)
                        self._repo.write_option_quotes(chunk_df, run_id)
                        rows_in_chunk = len(chunk_df)
                        option_rows_written += rows_in_chunk
                        self._logger.info(
                            f"Wrote {rows_in_chunk:,} quotes for chunk {i+1} "
                            f"({option_rows_written:,} cumulative)"
                        )
                        del chunk_df
                    else:
                        self._logger.info(f"No quotes fetched for chunk {i+1}")
                except Exception as e:
                    self._logger.error(f"Failed to fetch/write quotes chunk {i+1} ({chunk_desc}): {e}")
                    errors.append(f"Options chunk error ({chunk_desc}): {e}")
                    failed_chunks.append(chunk_desc)

        if option_rows_written > 0:
            self._logger.info(f"Total option quotes written: {option_rows_written:,}")

        # Step 4b: Fetch option bars (daily OHLCV for volume data)
        if not fred_only and not quotes_only and not stats_only and all_option_symbols:
            for i, (cs, ce) in enumerate(zip(chunk_starts, chunk_ends)):
                if cs >= ce:
                    continue
                manifest = chunk_manifests.get(cs)
                if not manifest or not manifest.symbols:
                    continue
                chunk_symbols = self._filter_symbols_for_chunk(manifest, cs.date())
                if not chunk_symbols:
                    continue
                chunk_desc = f"{cs.date()} to {ce.date()}"
                try:
                    symbol_batches = self._batch_symbols(chunk_symbols, max_batch_size=2000)
                    self._logger.info(
                        f"Fetching option bars chunk {i+1}/{len(chunk_starts)}: "
                        f"{chunk_desc} ({len(chunk_symbols)} symbols)..."
                    )

                    chunk_bar_dfs = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("once")
                        for batch_idx, symbol_batch in enumerate(symbol_batches, 1):
                            batch_df = self._fetch_with_retry(
                                lambda _cs=cs, _ce=ce, _syms=symbol_batch: self._market.fetch_option_bars(
                                    symbols=_syms,
                                    start=_cs.to_pydatetime(),
                                    end=_ce.to_pydatetime(),
                                    interval="1d",
                                ),
                                f"Option bars chunk {i+1}/{len(chunk_starts)} batch {batch_idx}/{len(symbol_batches)} ({chunk_desc})",
                            )
                            if not batch_df.empty:
                                chunk_bar_dfs.append(batch_df)

                    if chunk_bar_dfs:
                        chunk_bar_df = pd.concat(chunk_bar_dfs, ignore_index=True)
                        del chunk_bar_dfs
                        chunk_bar_df = self._prepare_option_bars(chunk_bar_df)
                        self._repo.write_option_bars(chunk_bar_df, run_id)
                        rows_in_chunk = len(chunk_bar_df)
                        option_bar_rows_written += rows_in_chunk
                        self._logger.info(
                            f"Wrote {rows_in_chunk:,} option bars for chunk {i+1} "
                            f"({option_bar_rows_written:,} cumulative)"
                        )
                        del chunk_bar_df
                except Exception as e:
                    self._logger.error(f"Failed to fetch/write option bars chunk {i+1} ({chunk_desc}): {e}")
                    errors.append(f"Option bars chunk error ({chunk_desc}): {e}")
                    failed_chunks.append(f"bars:{chunk_desc}")

        if option_bar_rows_written > 0:
            self._logger.info(f"Total option bars written: {option_bar_rows_written:,}")

        # Step 4c: Fetch option statistics (daily OI from statistics schema)
        if not fred_only and not quotes_only and all_option_symbols:
            for i, (cs, ce) in enumerate(zip(chunk_starts, chunk_ends)):
                if cs >= ce:
                    continue
                manifest = chunk_manifests.get(cs)
                if not manifest or not manifest.symbols:
                    continue
                chunk_symbols = self._filter_symbols_for_chunk(manifest, cs.date())
                if not chunk_symbols:
                    continue
                chunk_desc = f"{cs.date()} to {ce.date()}"
                try:
                    symbol_batches = self._batch_symbols(chunk_symbols, max_batch_size=2000)
                    self._logger.info(
                        f"Fetching option statistics chunk {i+1}/{len(chunk_starts)}: "
                        f"{chunk_desc} ({len(chunk_symbols)} symbols)..."
                    )

                    chunk_stats_dfs = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("once")
                        for batch_idx, symbol_batch in enumerate(symbol_batches, 1):
                            batch_df = self._fetch_with_retry(
                                lambda _cs=cs, _ce=ce, _syms=symbol_batch: self._market.fetch_option_statistics(
                                    symbols=_syms,
                                    start=_cs.to_pydatetime(),
                                    end=_ce.to_pydatetime(),
                                ),
                                f"Statistics chunk {i+1}/{len(chunk_starts)} batch {batch_idx}/{len(symbol_batches)} ({chunk_desc})",
                            )
                            if not batch_df.empty:
                                chunk_stats_dfs.append(batch_df)

                    if chunk_stats_dfs:
                        chunk_stats_df = pd.concat(chunk_stats_dfs, ignore_index=True)
                        del chunk_stats_dfs
                        self._repo.write_option_statistics(chunk_stats_df, run_id)
                        rows_in_chunk = len(chunk_stats_df)
                        statistics_rows_written += rows_in_chunk
                        self._logger.info(
                            f"Wrote {rows_in_chunk:,} option statistics for chunk {i+1} "
                            f"({statistics_rows_written:,} cumulative)"
                        )
                        del chunk_stats_df
                except Exception as e:
                    self._logger.error(f"Failed to fetch/write statistics chunk {i+1} ({chunk_desc}): {e}")
                    errors.append(f"Statistics chunk error ({chunk_desc}): {e}")
                    failed_chunks.append(f"stats:{chunk_desc}")

        if statistics_rows_written > 0:
            self._logger.info(f"Total option statistics written: {statistics_rows_written:,}")

        # Step 5: Fetch macro series
        macro_dfs: dict[str, pd.DataFrame] = {}
        skipped_series = 0
        if quotes_only or stats_only:
            self._logger.info(f"--{'quotes' if quotes_only else 'stats'}-only: skipping FRED series")
        for series_id in self._config.features.macro.series:
            if quotes_only or stats_only:
                break
            # Skip series that already have data in the DB
            if not force:
                existing_fred = self._repo.count_fred_series(series_id, start, end)
                if existing_fred > 0:
                    self._logger.info(
                        f"Skipping FRED {series_id}: "
                        f"{existing_fred:,} rows already exist"
                    )
                    skipped_series += 1
                    continue
            try:
                self._logger.info(f"Fetching FRED series {series_id}...")
                df = self._fetch_with_retry(
                    lambda _sid=series_id: self._macro.fetch_series(
                        series_id=_sid,
                        start=start,
                        end=end,
                    ),
                    f"FRED {series_id} fetch",
                )
                if not df.empty:
                    df["series_id"] = series_id
                macro_dfs[series_id] = df
                self._logger.info(f"Fetched {len(df)} observations for {series_id}")
            except Exception as e:
                self._logger.error(f"Failed to fetch FRED series {series_id}: {e}")
                errors.append(f"FRED {series_id} fetch error: {e}")

        # Step 6: Validate data quality
        # (Option quotes already validated per-chunk in Step 4)
        validation_results: dict[str, ValidationResult] = {}
        if not skip_validation:
            self._logger.info("Running data quality validation...")
            validation_results = self._validator.validate_all(
                underlying_df=underlying_df if not underlying_df.empty else None,
                options_df=None,  # Options validated per-chunk in Step 4
                macro_dfs=macro_dfs if macro_dfs else None,
            )

            # Merge per-chunk option validation results
            if chunk_validation_results:
                validation_results["option_quotes"] = self._merge_chunk_validations(
                    chunk_validation_results
                )

            # Log validation failures but don't abort — write valid data
            failed_checks = [
                name for name, result in validation_results.items() if not result.passed
            ]
            if failed_checks:
                for name in failed_checks:
                    result = validation_results[name]
                    for issue in result.issues:
                        self._logger.warning(
                            f"  [{name}] {issue.check_name}: {issue.message}"
                        )
                self._logger.warning(
                    f"Validation issues in: {failed_checks}. "
                    f"Writing data that passed validation."
                )

        # Step 7: Write to storage (option quotes already written per-chunk in Step 4)
        has_data = not underlying_df.empty or option_rows_written > 0 or option_bar_rows_written > 0 or statistics_rows_written > 0 or macro_dfs
        if has_data:
            passed_checks = {
                name for name, result in validation_results.items() if result.passed
            } if validation_results else set()
            # If validation was skipped, write everything
            write_all = skip_validation or not validation_results

            try:
                self._logger.info("Writing data to storage...")

                # Write underlying bars
                if not underlying_df.empty and (
                    write_all or "underlying_bars" in passed_checks
                ):
                    underlying_df = self._prepare_underlying_bars(underlying_df)
                    self._repo.write_underlying_bars(underlying_df, run_id)
                    self._logger.info(f"Wrote {len(underlying_df)} underlying bars")
                elif not underlying_df.empty:
                    self._logger.warning("Skipped underlying bars (failed validation)")

                # Option quotes already written per-chunk in Step 4

                # Write macro series (each validated independently)
                # Validation keys are prefixed: "fred_{series_id}"
                for series_id, df in macro_dfs.items():
                    validation_key = f"fred_{series_id}"
                    if not df.empty and (
                        write_all or validation_key in passed_checks
                    ):
                        self._repo.write_fred_series(df)
                        self._logger.info(f"Wrote {len(df)} {series_id} observations")

                self._logger.info("Data written to storage successfully")
            except Exception as e:
                self._logger.error(f"Failed to write to storage: {e}")
                errors.append(f"Storage write error: {e}")

        # Step 8: Write ingestion log
        end_time = datetime.now(UTC)
        if not errors:
            try:
                # Use first chunk's manifest for log metadata
                first_manifest = next(iter(chunk_manifests.values())) if chunk_manifests else None
                # Compute data timestamp bounds for audit trail
                ts_bounds: dict[str, datetime | None] = {}
                if not underlying_df.empty and "ts_utc" in underlying_df.columns:
                    ts_bounds["min_ts_event"] = underlying_df["ts_utc"].min()
                    ts_bounds["max_ts_event"] = underlying_df["ts_utc"].max()
                log_entry = self._build_log_entry(
                    run_id=run_id,
                    start=start,
                    end=end,
                    underlying_rows=len(underlying_df),
                    option_rows=option_rows_written + option_bar_rows_written + statistics_rows_written,
                    macro_rows=sum(len(df) for df in macro_dfs.values()),
                    cost=cost_estimate,
                    manifest=first_manifest,
                    ts_bounds=ts_bounds if ts_bounds else None,
                )
                self._repo.write_ingestion_log(log_entry)
                self._logger.info("Ingestion log written")
            except Exception as e:
                self._logger.error(f"Failed to write ingestion log: {e}")
                errors.append(f"Log write error: {e}")

        result = IngestionResult(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            data_start=start,
            data_end=end,
            preview_only=False,
            estimated_cost=cost_estimate,
            actual_cost=cost_estimate,  # In real implementation, track actual cost
            underlying_rows=len(underlying_df),
            option_rows=option_rows_written,
            option_bar_rows=option_bar_rows_written,
            statistics_rows=statistics_rows_written,
            macro_rows=sum(len(df) for df in macro_dfs.values()),
            skipped_underlying=skipped_underlying,
            skipped_chunks=skipped_chunks,
            skipped_series=skipped_series,
            failed_chunks=failed_chunks,
            validation_results=validation_results,
            errors=errors,
            adjusted_end_date=self._adjusted_end_date,
        )

        self._logger.info(result.summary())
        return result

    def _generate_run_id(self) -> str:
        """Generate unique run identifier.

        Returns:
            UUID string for the ingestion run
        """
        return str(uuid.uuid4())

    def _detect_max_available_date(self, requested_end: datetime) -> datetime | None:
        """Detect maximum available data date by probing the API.

        Attempts to fetch a small amount of data at the requested end date.
        If this fails due to data availability limits, parses the actual
        available end date from the error message.

        Args:
            requested_end: The requested end date to probe

        Returns:
            Maximum available date, or None if requested_end is available
        """
        try:
            # Probe with underlying data (cheaper than options)
            # Just try to fetch 1 day at the end of the range
            probe_start = requested_end - timedelta(days=1)
            self._market.fetch_underlying_bars(
                symbol=self._config.universe.underlying,
                start=probe_start,
                end=requested_end,
                interval="1d",
                suppress_error_log=True,  # Expected error during probe
            )
            # If successful, requested_end is available
            return None

        except ProviderError as e:
            error_str = str(e)

            # Check for data availability errors (both end and start variants)
            if "data_end_after_available_end" in error_str or \
               "data_start_after_available_end" in error_str or \
               "license_not_found_unauthorized" in error_str or \
               "live data license is required" in error_str:

                # Parse the actual available end date from various error formats
                # Format 1: "available up to 'YYYY-MM-DD...'" (most common)
                # Format 2: "dataset X ('YYYY-MM-DD...')" (for data_start_after_available_end)
                # Format 3: "after YYYY-MM-DD..." (fallback)
                match = re.search(r"available up to '([^']+)'", error_str) or \
                        re.search(r"available end of dataset [A-Z.]+ \('([^']+)'\)", error_str) or \
                        re.search(r"after (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", error_str)

                if match:
                    cutoff_str = match.group(1)
                    cutoff_date = datetime.fromisoformat(cutoff_str.replace('+00:00', ''))
                    self._logger.info(
                        f"Detected data availability limit: {cutoff_date.date()}"
                    )
                    return cutoff_date
                else:
                    # Couldn't parse, assume no limit
                    self._logger.warning(
                        f"Could not parse data availability limit from error: {error_str}"
                    )
                    return None
            else:
                # Different ProviderError, assume no limit
                self._logger.debug(
                    f"Non-availability ProviderError during probe: {error_str}"
                )
                return None

        except Exception as e:
            # Probe is best-effort; don't let it crash the whole ingestion
            self._logger.debug(f"Data availability probe failed: {e}")
            return None

    def _batch_symbols(self, symbols: list[str], max_batch_size: int) -> list[list[str]]:
        """Split symbol list into batches to respect API limits.

        Args:
            symbols: List of symbols to batch
            max_batch_size: Maximum symbols per batch (e.g., 2000 for Databento)

        Returns:
            List of symbol batches
        """
        batches = []
        for i in range(0, len(symbols), max_batch_size):
            batches.append(symbols[i : i + max_batch_size])
        return batches

    def _get_git_sha(self) -> str:
        """Get current git commit SHA.

        Returns:
            Short git SHA or 'unknown' if not in a git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            self._logger.debug(f"Failed to detect git SHA: {e}")
        return "unknown"

    def _load_or_generate_manifest(
        self,
        start: datetime,
        end: datetime,
        manifest_path: Path | None = None,
        force: bool = False,
        max_available_date: datetime | None = None,
    ) -> Manifest:
        """Load existing manifest or generate new one.

        Args:
            start: Start datetime for data range
            end: End datetime for data range
            manifest_path: Optional path to pre-generated manifest
            force: If True, regenerate manifest even if cached one exists
            max_available_date: Maximum date that data is available (caps queries)

        Returns:
            Manifest with option symbols to fetch
        """
        # If manifest path explicitly provided, always load it (user override)
        if manifest_path is not None and manifest_path.exists():
            self._logger.info(f"Loading manifest from {manifest_path}")
            return self._manifest_generator.load_manifest(manifest_path)

        # Check for existing manifest for this date
        as_of_date = start.date()
        default_path = self._manifest_generator.get_manifest_path(as_of_date)

        if not force and default_path.exists():
            self._logger.info(f"Loading existing manifest from {default_path}")
            return self._manifest_generator.load_manifest(default_path)

        if force and default_path.exists():
            self._logger.info(f"Regenerating manifest (--force), replacing {default_path}")

        # Generate new manifest
        self._logger.info("Generating new options manifest...")

        # First, we need underlying data to get spot reference
        # Cap end date to max available if provided
        effective_end = min(end, max_available_date) if max_available_date else end

        underlying_df = self._market.fetch_underlying_bars(
            symbol=self._config.universe.underlying,
            start=start,
            end=effective_end,
            interval="1d",  # Daily bars for spot reference
        )

        if underlying_df.empty:
            self._logger.warning("No underlying data for spot reference, using default spot")
            spot = 400.0  # Default for SPY
        else:
            spot = get_spot_reference(underlying_df, start)

        # Resolve available option symbols from provider
        available_symbols = self._market.resolve_option_symbols(
            parent=self._config.universe.underlying,
            as_of=start,
            dte_min=self._config.dataset.min_dte,
            dte_max=self._config.dataset.max_dte,
            moneyness_min=self._config.dataset.moneyness_min,
            moneyness_max=self._config.dataset.moneyness_max,
            max_available_date=max_available_date,
        )

        # Generate manifest with filtering
        manifest = self._manifest_generator.generate_manifest(
            available_symbols=available_symbols,
            spot_reference=spot,
            as_of_date=as_of_date,
            dte_min=self._config.dataset.min_dte,
            dte_max=self._config.dataset.max_dte,
            moneyness_min=self._config.dataset.moneyness_min,
            moneyness_max=self._config.dataset.moneyness_max,
            options_per_expiry_side=50,
        )

        # Save manifest for reproducibility
        self._manifest_generator.write_manifest(manifest, default_path)

        return manifest

    def _fetch_with_retry(self, fetch_fn: Callable, description: str):
        """Execute a fetch function with exponential backoff retry.

        Args:
            fetch_fn: Zero-argument callable that performs the fetch
            description: Human-readable description for log messages

        Returns:
            Result of fetch_fn()

        Raises:
            Exception: Re-raises the last exception after all retries exhausted
        """
        retry_cfg = self._config.data.ingestion.databento.retry
        for attempt in range(retry_cfg.max_retries + 1):
            try:
                return fetch_fn()
            except Exception as e:
                if attempt < retry_cfg.max_retries:
                    delay = min(
                        retry_cfg.base_delay_seconds * (2 ** attempt),
                        retry_cfg.max_delay_seconds,
                    )
                    self._logger.warning(
                        f"{description} failed (attempt {attempt + 1}/{retry_cfg.max_retries + 1}): "
                        f"{e}. Retrying in {delay:.0f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise

    def _filter_symbols_for_chunk(
        self,
        manifest: Manifest,
        chunk_start: date,
    ) -> list[str]:
        """Filter manifest symbols to those not expired before chunk start.

        Uses manifest.symbols_by_expiry to efficiently drop symbols whose
        expiry date is before the chunk's start date.

        Args:
            manifest: Options manifest with symbols grouped by expiry
            chunk_start: Start date of the current chunk

        Returns:
            List of symbols that are still active (expiry >= chunk_start)
        """
        if not manifest.symbols_by_expiry:
            return manifest.symbols

        active_symbols: list[str] = []
        for expiry_str, symbols in manifest.symbols_by_expiry.items():
            expiry = date.fromisoformat(expiry_str)
            if expiry >= chunk_start:
                active_symbols.extend(symbols)

        return sorted(active_symbols)

    def _prepare_underlying_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare underlying bars DataFrame with required metadata.

        Args:
            df: Raw underlying bars DataFrame

        Returns:
            DataFrame with additional metadata columns
        """
        df = df.copy()

        # Add dataset metadata
        df["dataset"] = self._config.data.providers.databento.dataset_equities
        df["schema"] = "ohlcv-1m"
        df["stype_in"] = self._config.data.providers.databento.stype_in
        df["symbol"] = self._config.universe.underlying
        df["timeframe"] = "1m"

        # Add placeholder columns if missing
        if "instrument_id" not in df.columns:
            df["instrument_id"] = None
        if "publisher_id" not in df.columns:
            df["publisher_id"] = None
        if "ts_recv_utc" not in df.columns:
            df["ts_recv_utc"] = None

        return df

    def _merge_chunk_validations(
        self, chunk_results: list[ValidationResult]
    ) -> ValidationResult:
        """Merge per-chunk ValidationResults into one aggregate result.

        Args:
            chunk_results: List of per-chunk ValidationResult objects

        Returns:
            Merged ValidationResult with combined issues and row counts
        """
        all_issues = []
        total_rows = 0
        for cr in chunk_results:
            all_issues.extend(cr.issues)
            total_rows += cr.total_rows
        passed = not any(i.severity == IssueSeverity.ERROR for i in all_issues)
        return ValidationResult(passed=passed, issues=all_issues, total_rows=total_rows)

    def _prepare_option_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare option quotes DataFrame with required metadata.

        Args:
            df: Raw option quotes DataFrame

        Returns:
            DataFrame with additional metadata columns
        """
        df = df.copy()

        # Add dataset metadata
        df["dataset"] = self._config.data.providers.databento.dataset_options
        df["schema"] = self._config.data.ingestion.databento.options.quote_schema
        df["stype_in"] = self._config.data.providers.databento.stype_in

        # Parse exp_date, strike, right from OCC option symbol if missing
        # OCC format: "SPY   230413P00402000" → underlying(6) + date(6) + right(1) + strike(8)
        if "option_symbol" in df.columns and (
            "exp_date" not in df.columns or df["exp_date"].isna().all()
        ):
            stripped = df["option_symbol"].str.strip()
            df["exp_date"] = pd.to_datetime(
                stripped.str[-15:-9], format="%y%m%d"
            ).dt.date
            df["right"] = stripped.str[-9]
            df["strike"] = stripped.str[-8:].astype(float) / 1000.0

        # Add placeholder columns if missing
        if "instrument_id" not in df.columns:
            df["instrument_id"] = None
        if "publisher_id" not in df.columns:
            df["publisher_id"] = None
        if "ts_recv_utc" not in df.columns:
            df["ts_recv_utc"] = None
        if "volume" not in df.columns:
            df["volume"] = None
        if "open_interest" not in df.columns:
            df["open_interest"] = None

        return df

    def _prepare_option_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare option bars DataFrame with required metadata.

        Args:
            df: Raw option bars DataFrame

        Returns:
            DataFrame with additional metadata columns
        """
        df = df.copy()
        df["timeframe"] = "1d"
        return df

    def _build_log_entry(
        self,
        run_id: str,
        start: datetime,
        end: datetime,
        underlying_rows: int,
        option_rows: int,
        macro_rows: int,
        cost: float,
        manifest: Manifest | None,
        ts_bounds: dict[str, datetime | None] | None = None,
    ) -> dict:
        """Build ingestion log entry for database.

        Args:
            run_id: Unique run identifier
            start: Start datetime
            end: End datetime
            underlying_rows: Number of underlying bars ingested
            option_rows: Number of option quotes ingested
            macro_rows: Number of macro observations ingested
            cost: Total cost of ingestion
            manifest: Options manifest used (first chunk, or None)
            ts_bounds: Optional dict with min/max event timestamps

        Returns:
            Dictionary with log entry fields
        """
        import json

        # Build config snapshot with manifest info
        if manifest is not None:
            config_snapshot = json.dumps({
                "manifest_hash": manifest.metadata.config_hash,
                "symbols_count": manifest.metadata.symbols_count,
                "expiries_count": manifest.metadata.expiries_count,
                "spot_reference": manifest.metadata.spot_reference,
                "dte_min": manifest.metadata.dte_min,
                "dte_max": manifest.metadata.dte_max,
            })
        else:
            config_snapshot = json.dumps({})

        return {
            "ingest_run_id": run_id,
            "dataset": self._config.data.providers.databento.dataset_equities,
            "schema": "ohlcv-1m",
            "stype_in": self._config.data.providers.databento.stype_in,
            "symbols": json.dumps([self._config.universe.underlying]),
            "start_date": start.date(),
            "end_date": end.date(),
            "row_count": underlying_rows + option_rows + macro_rows,
            "cost_usd": cost,
            "git_sha": self._get_git_sha(),
            "config_snapshot": config_snapshot,
            "source_ingested_at": datetime.now(UTC),
            "min_ts_event_utc": ts_bounds.get("min_ts_event") if ts_bounds else None,
            "max_ts_event_utc": ts_bounds.get("max_ts_event") if ts_bounds else None,
        }


def create_orchestrator(
    config: ConfigSchema,
    market_provider: MarketDataProvider | None = None,
    macro_provider: MacroDataProvider | None = None,
    repository: RawRepository | None = None,
) -> IngestionOrchestrator:
    """Factory function to create IngestionOrchestrator with providers.

    Creates providers from configuration if not provided.

    Args:
        config: Application configuration
        market_provider: Optional market data provider
        macro_provider: Optional macro data provider
        repository: Optional raw data repository

    Returns:
        Configured IngestionOrchestrator instance
    """
    # Import here to avoid circular imports
    from src.data.providers.databento import DatabentoProvider
    from src.data.providers.fred import FREDProvider
    from src.data.storage.engine import create_engine

    if market_provider is None:
        market_provider = DatabentoProvider(
            dataset_equities=config.data.providers.databento.dataset_equities,
            dataset_options=config.data.providers.databento.dataset_options,
            definition_query_days=config.data.providers.databento.definition_query_days,
        )

    if macro_provider is None:
        macro_provider = FREDProvider()

    if repository is None:
        db_engine = create_engine(config.paths.db_path)
        db_engine.create_tables()
        repository = RawRepository(db_engine.engine)

    return IngestionOrchestrator(
        market_provider=market_provider,
        macro_provider=macro_provider,
        repository=repository,
        config=config,
    )
