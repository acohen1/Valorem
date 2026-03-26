"""Ingestion services for streaming data to the database.

This module provides the ingestion layer that persists market data to the
database. All live/mock data flows through ingestion before being read
by the SurfaceProvider.

Architecture:
    Databento ──→ DatabentoIngestionService ──→ DB
    Synthetic ──→ MockIngestionService ──────→ DB
                                               ↓
                                        SurfaceProvider ──→ TradingLoop
"""

import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from src.config.constants import SurfaceConstants
from src.config.schema import SurfaceConfig, UniverseConfig
from src.data.storage.repository import DerivedRepository, RawRepository
from src.surface.buckets.assign import DeltaBucketAssigner
from src.surface.greeks.analytical import AnalyticalGreeks
from src.surface.iv.black_scholes import BlackScholesIVSolver
from src.surface.quality.filters import QualityConfig, QualityFilter

logger = logging.getLogger(__name__)


class IngestionService(Protocol):
    """Protocol for data ingestion services.

    Ingestion services are responsible for:
    1. Fetching/generating market data
    2. Building volatility surfaces
    3. Persisting surfaces to the database

    The TradingLoop does not interact with ingestion directly - it reads
    surfaces via SurfaceProvider which queries the database.
    """

    def start(self) -> None:
        """Start the ingestion service.

        Begins fetching/generating data and persisting to database.
        For streaming services, this starts a background thread.
        """
        ...

    def stop(self) -> None:
        """Stop the ingestion service.

        Gracefully shuts down data fetching and persists any buffered data.
        """
        ...

    @property
    def is_running(self) -> bool:
        """Check if the ingestion service is currently running."""
        ...


class BaseIngestionService(ABC):
    """Base class for ingestion services with common functionality."""

    def __init__(
        self,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
        surface_config: SurfaceConfig,
        universe_config: UniverseConfig,
        risk_free_rate: float = 0.05,
        surface_version: str = "live",
    ):
        """Initialize base ingestion service.

        Args:
            raw_repo: Repository for raw data (quotes, bars)
            derived_repo: Repository for derived data (surfaces)
            surface_config: Surface configuration
            universe_config: Universe configuration
            risk_free_rate: Risk-free rate for IV calculations
            surface_version: Version tag for persisted surfaces (default: "live")
        """
        self._raw_repo = raw_repo
        self._derived_repo = derived_repo
        self._surface_config = surface_config
        self._universe_config = universe_config
        self._risk_free_rate = risk_free_rate
        self._surface_version = surface_version
        self._running = False

        # Surface building components
        self._iv_solver = BlackScholesIVSolver(
            max_iters=surface_config.black_scholes.max_iterations,
            tolerance=surface_config.black_scholes.tolerance,
        )
        self._greeks_calculator = AnalyticalGreeks()
        self._bucket_assigner = DeltaBucketAssigner(
            self._buckets_config_to_dict(surface_config.delta_buckets)
        )
        self._quality_filter = QualityFilter(QualityConfig())

    def _buckets_config_to_dict(self, buckets_config) -> dict[str, list[float]]:
        """Convert DeltaBucketsConfig to dictionary format."""
        return {
            "ATM": buckets_config.ATM,
            "P40": buckets_config.P40,
            "P25": buckets_config.P25,
            "P10": buckets_config.P10,
            "C10": buckets_config.C10,
            "C25": buckets_config.C25,
            "C40": buckets_config.C40,
        }

    @abstractmethod
    def start(self) -> None:
        """Start the ingestion service."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the ingestion service."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if the ingestion service is running."""
        return self._running

    def _build_and_persist_surface(
        self,
        quotes_df: pd.DataFrame,
        underlying_price: float,
        timestamp: datetime,
    ) -> None:
        """Build surface from quotes and persist to database.

        Args:
            quotes_df: DataFrame with option quotes
            underlying_price: Current underlying price
            timestamp: Timestamp for the surface snapshot
        """
        if quotes_df.empty:
            logger.debug("Empty quotes DataFrame, skipping surface build")
            return

        # Add required columns
        quotes_df = quotes_df.copy()
        quotes_df["underlying_price"] = underlying_price
        quotes_df["rf_rate"] = self._risk_free_rate

        # Build surface
        surface = self._build_surface(quotes_df, timestamp)

        if surface.empty:
            logger.debug("Empty surface after build, skipping persistence")
            return

        # Persist to database
        build_run_id = str(uuid.uuid4())[:8]
        try:
            self._derived_repo.write_surface_snapshots(
                surface,
                build_run_id=build_run_id,
                version=self._surface_version,
            )
            logger.debug(
                f"Persisted surface snapshot: {len(surface)} nodes at {timestamp}"
            )
        except Exception as e:
            logger.error(f"Failed to persist surface: {e}")

    def _build_surface(
        self,
        quotes_df: pd.DataFrame,
        timestamp: datetime,
    ) -> pd.DataFrame:
        """Build surface from quotes DataFrame.

        Args:
            quotes_df: DataFrame with parsed option quotes
            timestamp: Timestamp for the snapshot

        Returns:
            DataFrame with surface (one row per node)
        """
        if quotes_df.empty:
            return pd.DataFrame()

        # Compute TTE
        now = timestamp
        quotes_df["tte_years"] = quotes_df["exp_date"].apply(
            lambda exp: max(
                (datetime.combine(exp, datetime.min.time()).replace(tzinfo=timezone.utc) - now).total_seconds()
                / (365.25 * 86400),
                1e-6,
            )
        )

        # Mid price and spread
        quotes_df["mid_price"] = (quotes_df["bid"] + quotes_df["ask"]) / 2
        quotes_df["spread"] = quotes_df["ask"] - quotes_df["bid"]
        quotes_df["spread_pct"] = quotes_df["spread"] / np.maximum(quotes_df["mid_price"], 1e-10)

        # Solve IV
        quotes_df["iv_mid"] = self._iv_solver.solve_iv_vectorized(
            prices=quotes_df["mid_price"],
            S=quotes_df["underlying_price"],
            K=quotes_df["strike"],
            T=quotes_df["tte_years"],
            r=quotes_df["rf_rate"],
            q=0.0,
            right=quotes_df["right"],
        )

        # Compute Greeks for valid IV
        # Drop any existing Greeks columns to avoid conflicts
        greeks_cols = ["delta", "gamma", "vega", "theta"]
        existing_greeks = [c for c in greeks_cols if c in quotes_df.columns]
        if existing_greeks:
            quotes_df = quotes_df.drop(columns=existing_greeks)

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
            for col in greeks_cols:
                quotes_df[col] = np.nan

        # Assign tenor bins
        tenor_bins = np.array(self._surface_config.tenor_bins.bins)
        quotes_df["tenor_days"] = quotes_df["tte_years"].apply(
            lambda tte: int(tenor_bins[np.argmin(np.abs(tenor_bins - tte * 365))])
        )

        # Assign delta buckets
        quotes_df["delta_bucket"] = self._bucket_assigner.assign(quotes_df["delta"])

        # Quality flags
        quotes_df["flags"] = self._quality_filter.compute_flags(quotes_df)

        # Add timestamp and rename columns for consistency
        quotes_df["ts_utc"] = timestamp
        quotes_df["expiry"] = quotes_df["exp_date"]

        # Select representatives (one per node)
        return self._select_representatives(quotes_df)

    def _select_representatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select one option per (tenor_days, delta_bucket) node.

        Args:
            df: DataFrame with all processed quotes

        Returns:
            DataFrame with one row per node
        """
        valid = df[
            df["delta_bucket"].notna()
            & df["tenor_days"].notna()
            & df["iv_mid"].notna()
        ].copy()

        if valid.empty:
            return pd.DataFrame()

        # Compute distance from bucket center
        def get_bucket_center(bucket_name):
            if bucket_name == "ATM":
                return 0.5
            else:
                min_d, max_d = self._bucket_assigner.get_bucket_bounds(bucket_name)
                return (min_d + max_d) / 2

        valid["bucket_center"] = valid["delta_bucket"].apply(get_bucket_center)

        valid["delta_for_distance"] = np.where(
            valid["delta_bucket"] == "ATM",
            np.abs(valid["delta"]),
            valid["delta"],
        )
        valid["distance_to_center"] = np.abs(
            valid["delta_for_distance"] - valid["bucket_center"]
        )

        # Deprioritize flagged options
        valid["selection_score"] = (
            valid["distance_to_center"] + (valid["flags"] > 0).astype(float) * 100
        )

        # Sort and select best per node
        valid = valid.sort_values(["tenor_days", "delta_bucket", "selection_score"])

        representatives = valid.groupby(
            ["tenor_days", "delta_bucket"], as_index=False
        ).first()

        # Clean up helper columns
        drop_cols = [
            "bucket_center",
            "delta_for_distance",
            "distance_to_center",
            "selection_score",
            "exp_date",
            "tte_years",
            "rf_rate",
        ]
        representatives = representatives.drop(
            columns=[c for c in drop_cols if c in representatives.columns]
        )

        return representatives


class DatabentoIngestionService(BaseIngestionService):
    """Ingestion service that streams live data from Databento to database.

    This service:
    1. Connects to Databento's Live API (websocket streaming)
    2. Subscribes to option quotes for configured symbols
    3. Builds surfaces at regular intervals
    4. Persists surfaces to the database

    Example:
        service = DatabentoIngestionService(
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            option_symbols=symbols,
        )
        service.start()  # Starts background streaming and persistence
        # ... trading loop reads from database ...
        service.stop()
    """

    def __init__(
        self,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
        surface_config: SurfaceConfig,
        universe_config: UniverseConfig,
        api_key: Optional[str] = None,
        option_symbols: Optional[list[str]] = None,
        risk_free_rate: float = 0.05,
        quote_buffer_seconds: int = 60,
        surface_interval_seconds: float = 5.0,
    ):
        """Initialize Databento ingestion service.

        Args:
            raw_repo: Repository for raw data
            derived_repo: Repository for derived data (surfaces)
            surface_config: Surface configuration
            universe_config: Universe configuration
            api_key: Databento API key (defaults to DATABENTO_API_KEY env var)
            option_symbols: List of option symbols to subscribe to
            risk_free_rate: Risk-free rate for IV calculations
            quote_buffer_seconds: How long to keep quotes in buffer
            surface_interval_seconds: How often to build and persist surfaces
        """
        super().__init__(
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            risk_free_rate=risk_free_rate,
            surface_version="live",
        )

        key = api_key or os.getenv("DATABENTO_API_KEY")
        if not key:
            raise ValueError(
                "Databento API key required. Provide via api_key parameter "
                "or DATABENTO_API_KEY environment variable."
            )

        self._api_key = key
        self._option_symbols = option_symbols or []
        self._underlying_symbol = universe_config.underlying
        self._quote_buffer_seconds = quote_buffer_seconds
        self._surface_interval_seconds = surface_interval_seconds

        # Live client (created on start)
        self._live_client = None

        # Quote buffers (thread-safe access)
        self._quote_buffer: dict[str, dict] = {}
        self._quote_timestamps: dict[str, datetime] = {}
        self._underlying_price: Optional[float] = None
        self._underlying_timestamp: Optional[datetime] = None
        self._buffer_lock = threading.Lock()

        # Threads
        self._stream_thread: Optional[threading.Thread] = None
        self._surface_thread: Optional[threading.Thread] = None

    def set_symbols(self, option_symbols: list[str]) -> None:
        """Set option symbols to subscribe to.

        Must be called before start() if symbols weren't provided in __init__.
        """
        if self._running:
            raise RuntimeError("Cannot change symbols while running. Call stop() first.")
        self._option_symbols = option_symbols

    def start(self) -> None:
        """Start the ingestion service.

        Connects to Databento and begins streaming quotes to the database.
        """
        if self._running:
            logger.warning("DatabentoIngestionService is already running")
            return

        if not self._option_symbols:
            raise RuntimeError("No option symbols configured. Call set_symbols() first.")

        logger.info(
            f"Starting DatabentoIngestionService for {len(self._option_symbols)} symbols"
        )

        import databento as db

        # Create live client
        self._live_client = db.Live(key=self._api_key)

        # Subscribe to option quotes
        self._live_client.subscribe(
            dataset="OPRA.PILLAR",
            schema="cbbo",
            symbols=self._option_symbols,
            stype_in="raw_symbol",
        )

        # Subscribe to underlying quotes
        self._live_client.subscribe(
            dataset="XNAS.ITCH",
            schema="cbbo",
            symbols=[self._underlying_symbol],
            stype_in="raw_symbol",
        )

        self._running = True

        # Start streaming thread
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            daemon=True,
            name="DatabentoIngestion-Stream",
        )
        self._stream_thread.start()

        # Start surface building thread
        self._surface_thread = threading.Thread(
            target=self._surface_loop,
            daemon=True,
            name="DatabentoIngestion-Surface",
        )
        self._surface_thread.start()

        logger.info("DatabentoIngestionService started")

    def stop(self) -> None:
        """Stop the ingestion service."""
        if not self._running:
            return

        logger.info("Stopping DatabentoIngestionService")
        self._running = False

        if self._live_client:
            try:
                self._live_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping live client: {e}")
            self._live_client = None

        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
            self._stream_thread = None

        if self._surface_thread:
            self._surface_thread.join(timeout=5.0)
            self._surface_thread = None

        logger.info("DatabentoIngestionService stopped")

    def _stream_loop(self) -> None:
        """Background thread that receives and buffers quotes."""
        try:
            self._live_client.start()

            for record in self._live_client:
                if not self._running:
                    break
                self._process_record(record)

        except Exception as e:
            if self._running:
                logger.error(f"Error in stream loop: {e}")
        finally:
            logger.debug("Stream loop exited")

    def _process_record(self, record) -> None:
        """Process a single record from the live stream."""
        try:
            symbol = getattr(record, "symbol", None)
            if not symbol:
                return

            ts = datetime.fromtimestamp(
                record.ts_event / 1_000_000_000, tz=timezone.utc
            )

            with self._buffer_lock:
                if symbol == self._underlying_symbol:
                    self._underlying_price = float(record.bid_px + record.ask_px) / 2
                    self._underlying_timestamp = ts
                else:
                    # Parse option symbol for strike/expiry/right
                    parsed = self._parse_option_symbol(symbol)
                    if parsed:
                        self._quote_buffer[symbol] = {
                            "option_symbol": symbol,
                            "bid": float(record.bid_px) / 1e9,
                            "ask": float(record.ask_px) / 1e9,
                            "bid_size": int(record.bid_sz),
                            "ask_size": int(record.ask_sz),
                            "strike": parsed["strike"],
                            "exp_date": parsed["expiry"],
                            "right": parsed["right"],
                        }
                        self._quote_timestamps[symbol] = ts

        except Exception as e:
            logger.debug(f"Error processing record: {e}")

    def _parse_option_symbol(self, symbol: str) -> Optional[dict]:
        """Parse OCC option symbol to extract strike, expiry, right."""
        try:
            i = 0
            while i < len(symbol) and not symbol[i].isdigit():
                i += 1

            date_str = symbol[i : i + 6]
            right_char = symbol[i + 6]
            strike_str = symbol[i + 7 :]

            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            expiry = date(year, month, day)

            right = "C" if right_char == "C" else "P"
            strike = int(strike_str) / 1000.0

            return {"strike": strike, "expiry": expiry, "right": right}
        except Exception as e:
            logger.debug(f"Failed to parse option symbol '{symbol}': {e}")
            return None

    def _surface_loop(self) -> None:
        """Background thread that builds and persists surfaces at intervals."""
        import time

        while self._running:
            try:
                self._build_current_surface()
            except Exception as e:
                logger.error(f"Error building surface: {e}")

            time.sleep(self._surface_interval_seconds)

    def _build_current_surface(self) -> None:
        """Build surface from current quote buffer and persist."""
        with self._buffer_lock:
            if not self._quote_buffer:
                return

            if self._underlying_price is None:
                return

            # Filter stale quotes
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=self._quote_buffer_seconds)

            fresh_quotes = []
            for symbol, quote in self._quote_buffer.items():
                quote_ts = self._quote_timestamps.get(symbol)
                if quote_ts and quote_ts >= cutoff:
                    fresh_quotes.append(quote)

            underlying_price = self._underlying_price

        if not fresh_quotes:
            return

        quotes_df = pd.DataFrame(fresh_quotes)
        self._build_and_persist_surface(quotes_df, underlying_price, now)


class MockIngestionService(BaseIngestionService):
    """Ingestion service that generates synthetic data for testing.

    This service generates synthetic option surfaces at regular intervals
    and persists them to the database. Useful for testing the trading loop
    without requiring a live data connection.

    Example:
        service = MockIngestionService(
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
        )
        service.start()  # Starts background generation and persistence
        # ... trading loop reads from database ...
        service.stop()
    """

    def __init__(
        self,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
        surface_config: SurfaceConfig,
        universe_config: UniverseConfig,
        underlying_price: float = 450.0,
        base_iv: float = 0.20,
        risk_free_rate: float = 0.05,
        random_walk: bool = True,
        surface_interval_seconds: float = 5.0,
        seed: Optional[int] = None,
    ):
        """Initialize mock ingestion service.

        Args:
            raw_repo: Repository for raw data (not used, but kept for interface)
            derived_repo: Repository for derived data (surfaces)
            surface_config: Surface configuration
            universe_config: Universe configuration
            underlying_price: Starting underlying price
            base_iv: Base implied volatility
            risk_free_rate: Risk-free rate
            random_walk: Whether to apply random walk to underlying price
            surface_interval_seconds: How often to generate and persist surfaces
            seed: Random seed for reproducibility
        """
        super().__init__(
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            risk_free_rate=risk_free_rate,
            surface_version="mock",
        )

        self._underlying_price = underlying_price
        self._base_iv = base_iv
        self._random_walk = random_walk
        self._surface_interval_seconds = surface_interval_seconds
        self._rng = np.random.default_rng(seed)

        self._surface_thread: Optional[threading.Thread] = None
        self._generation_count = 0

    def start(self) -> None:
        """Start the mock ingestion service."""
        if self._running:
            logger.warning("MockIngestionService is already running")
            return

        logger.info("Starting MockIngestionService")
        self._running = True

        # Start surface generation thread
        self._surface_thread = threading.Thread(
            target=self._generation_loop,
            daemon=True,
            name="MockIngestion-Surface",
        )
        self._surface_thread.start()

        logger.info("MockIngestionService started")

    def stop(self) -> None:
        """Stop the mock ingestion service."""
        if not self._running:
            return

        logger.info("Stopping MockIngestionService")
        self._running = False

        if self._surface_thread:
            self._surface_thread.join(timeout=5.0)
            self._surface_thread = None

        logger.info("MockIngestionService stopped")

    def _generation_loop(self) -> None:
        """Background thread that generates and persists mock surfaces."""
        import time

        while self._running:
            try:
                self._generate_and_persist_surface()
            except Exception as e:
                logger.error(f"Error generating mock surface: {e}")

            time.sleep(self._surface_interval_seconds)

    def _generate_and_persist_surface(self) -> None:
        """Generate synthetic surface and persist to database."""
        self._generation_count += 1

        # Apply random walk to underlying
        if self._random_walk:
            self._underlying_price *= 1 + self._rng.normal(0, 0.002)

        # Generate synthetic quotes
        quotes_df = self._generate_synthetic_quotes()

        # Build and persist
        now = datetime.now(timezone.utc)
        self._build_and_persist_surface(quotes_df, self._underlying_price, now)

    def _generate_synthetic_quotes(self) -> pd.DataFrame:
        """Generate synthetic option quotes."""
        tenors = list(SurfaceConstants.TENOR_DAYS_DEFAULT)
        strike_pcts = [0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10]
        as_of_date = date.today()

        records = []
        for tenor in tenors:
            expiry = as_of_date + timedelta(days=tenor)

            for strike_pct in strike_pcts:
                strike = round(self._underlying_price * strike_pct, 2)

                for right in ["C", "P"]:
                    # Compute synthetic delta
                    moneyness = self._underlying_price / strike
                    if right == "C":
                        delta = max(0.05, min(0.95, 0.5 + 0.4 * (moneyness - 1)))
                    else:
                        delta = -max(0.05, min(0.95, 0.5 - 0.4 * (moneyness - 1)))

                    # IV with skew
                    iv = self._base_iv * (1 + 0.1 * (1 - moneyness))

                    # Synthetic price
                    time_value = iv * np.sqrt(tenor / 365) * self._underlying_price * 0.4
                    intrinsic = max(
                        0,
                        (self._underlying_price - strike)
                        if right == "C"
                        else (strike - self._underlying_price),
                    )
                    mid_price = intrinsic + time_value * abs(delta)

                    bid = max(0.01, mid_price * 0.98)
                    ask = mid_price * 1.02

                    # Generate OSI symbol
                    expiry_str = expiry.strftime("%y%m%d")
                    strike_str = f"{int(strike * 1000):08d}"
                    symbol = f"SPY{expiry_str}{right}{strike_str}"

                    records.append(
                        {
                            "option_symbol": symbol,
                            "strike": strike,
                            "exp_date": expiry,
                            "right": right,
                            "bid": round(bid, 2),
                            "ask": round(ask, 2),
                            "delta": round(delta, 4),
                        }
                    )

        return pd.DataFrame(records)

    @property
    def generation_count(self) -> int:
        """Get number of surfaces generated."""
        return self._generation_count

    @property
    def underlying_price(self) -> float:
        """Get current underlying price."""
        return self._underlying_price
