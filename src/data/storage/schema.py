"""SQLAlchemy schema definitions for Valorem database.

This module defines all raw and derived tables using SQLAlchemy ORM.
Tables follow the immutable raw data / versioned derived data pattern.
"""

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

# Create metadata and base
metadata = MetaData()
Base = declarative_base(metadata=metadata)


# ============================================================================
# Raw Tables (Immutable, Append-Only)
# ============================================================================


class RawUnderlyingBar(Base):
    """Raw underlying OHLCV bars from market data provider."""

    __tablename__ = "raw_underlying_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Provider metadata
    dataset = Column(String, nullable=False)
    schema = Column(String, nullable=False)
    stype_in = Column(String, nullable=False)
    instrument_id = Column(Integer, nullable=True)
    publisher_id = Column(Integer, nullable=True)

    # Timestamps
    ts_utc = Column(DateTime, nullable=False)
    ts_recv_utc = Column(DateTime, nullable=True)

    # Symbol
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)

    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)

    # Ingestion metadata
    source_ingested_at = Column(DateTime, nullable=False)
    ingest_run_id = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "ts_utc", "timeframe", name="uq_bars_symbol_ts_timeframe"),
        Index("idx_bars_ts", "ts_utc"),
        Index("idx_bars_symbol", "symbol"),
    )


class RawOptionQuote(Base):
    """Raw option quotes (CBBO) from market data provider."""

    __tablename__ = "raw_option_quotes"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Provider metadata
    dataset = Column(String, nullable=False)
    schema = Column(String, nullable=False)
    stype_in = Column(String, nullable=False)
    instrument_id = Column(Integer, nullable=True)
    publisher_id = Column(Integer, nullable=True)

    # Timestamps
    ts_utc = Column(DateTime, nullable=False)
    ts_recv_utc = Column(DateTime, nullable=True)

    # Option identification
    option_symbol = Column(String, nullable=False)
    exp_date = Column(Date, nullable=False)
    strike = Column(Float, nullable=False)
    right = Column(String, nullable=False)  # 'C' or 'P'

    # Quote data
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    bid_size = Column(Integer, nullable=True)
    ask_size = Column(Integer, nullable=True)
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)

    # Ingestion metadata
    source_ingested_at = Column(DateTime, nullable=False)
    ingest_run_id = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("option_symbol", "ts_utc", name="uq_quotes_symbol_ts"),
        Index("idx_quotes_ts", "ts_utc"),
        Index("idx_quotes_exp", "exp_date"),
    )


class RawOptionBar(Base):
    """Raw option OHLCV bars from market data provider."""

    __tablename__ = "raw_option_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timestamps
    ts_utc = Column(DateTime, nullable=False)

    # Option identification
    option_symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)  # '1d', '1m', etc.

    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)

    # Ingestion metadata
    source_ingested_at = Column(DateTime, nullable=False)
    ingest_run_id = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "option_symbol", "ts_utc", "timeframe",
            name="uq_option_bars_symbol_ts_timeframe",
        ),
        Index("idx_option_bars_ts", "ts_utc"),
        Index("idx_option_bars_symbol", "option_symbol"),
    )


class RawOptionStatistic(Base):
    """Raw option statistics from market data provider (e.g., daily open interest)."""

    __tablename__ = "raw_option_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timestamps
    ts_ref = Column(DateTime, nullable=False)  # Reference date the stat applies to
    ts_event = Column(DateTime, nullable=False)  # When the stat was published

    # Option identification
    option_symbol = Column(String, nullable=False)

    # Statistic data
    stat_type = Column(Integer, nullable=False)  # 9=OI, 6=cleared_vol, 3=settlement
    quantity = Column(Integer, nullable=True)  # Value for non-price stats (e.g., OI count)
    price = Column(Float, nullable=True)  # Value for price-type stats (fixed-point 1e-9)

    # Ingestion metadata
    source_ingested_at = Column(DateTime, nullable=False)
    ingest_run_id = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "option_symbol", "ts_ref", "stat_type",
            name="uq_option_stats_symbol_tsref_type",
        ),
        Index("idx_option_stats_ts_ref", "ts_ref"),
        Index("idx_option_stats_symbol", "option_symbol"),
    )


class RawFredSeries(Base):
    """Raw FRED (Federal Reserve Economic Data) time series."""

    __tablename__ = "raw_fred_series"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Series identification
    series_id = Column(String, nullable=False)
    obs_date = Column(Date, nullable=False)
    value = Column(Float, nullable=False)

    # Release metadata
    release_datetime_utc = Column(DateTime, nullable=True)  # NULL if unknown

    # Ingestion metadata
    source_ingested_at = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("series_id", "obs_date", name="uq_fred_series_obs"),
        Index("idx_fred_series", "series_id", "obs_date"),
    )


class RawIngestionLog(Base):
    """Log of raw data ingestion runs for audit trail."""

    __tablename__ = "raw_ingestion_log"

    ingest_run_id = Column(String, primary_key=True)

    # Dataset metadata
    dataset = Column(String, nullable=False)
    schema = Column(String, nullable=False)
    stype_in = Column(String, nullable=False)
    symbols = Column(Text, nullable=False)  # JSON array

    # Date range
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

    # Statistics
    row_count = Column(Integer, nullable=False)
    min_ts_event_utc = Column(DateTime, nullable=True)
    max_ts_event_utc = Column(DateTime, nullable=True)
    min_ts_recv_utc = Column(DateTime, nullable=True)
    max_ts_recv_utc = Column(DateTime, nullable=True)

    # Cost tracking
    cost_usd = Column(Float, nullable=True)

    # Reproducibility
    git_sha = Column(String, nullable=True)
    config_snapshot = Column(Text, nullable=True)  # JSON

    # Ingestion metadata
    source_ingested_at = Column(DateTime, nullable=False)


# ============================================================================
# Derived Tables (Versioned, Reproducible)
# ============================================================================


class SurfaceSnapshot(Base):
    """Derived surface snapshots with IV, Greeks, and bucket assignments."""

    __tablename__ = "surface_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timestamp
    ts_utc = Column(DateTime, nullable=False)

    # Option identification
    option_symbol = Column(String, nullable=False)
    exp_date = Column(Date, nullable=False)
    strike = Column(Float, nullable=False)
    right = Column(String, nullable=False)

    # Pricing
    bid = Column(Float, nullable=False)
    ask = Column(Float, nullable=False)
    mid_price = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    spread_pct = Column(Float, nullable=False)

    # Time
    tte_years = Column(Float, nullable=False)
    tenor_days = Column(Integer, nullable=False)

    # Market data
    underlying_price = Column(Float, nullable=False)
    rf_rate = Column(Float, nullable=False)
    dividend_yield = Column(Float, nullable=False)

    # Implied volatility
    iv_mid = Column(Float, nullable=False)
    iv_bid = Column(Float, nullable=True)
    iv_ask = Column(Float, nullable=True)

    # Greeks
    delta = Column(Float, nullable=False)
    gamma = Column(Float, nullable=False)
    vega = Column(Float, nullable=False)
    theta = Column(Float, nullable=False)

    # Bucket assignment
    delta_bucket = Column(String, nullable=True)  # 'P10', 'P25', 'ATM', etc. or NULL

    # Quality flags
    flags = Column(Integer, nullable=False)  # Bitfield: 1=crossed, 2=stale, 4=wide_spread, 8=low_volume, 16=low_oi
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)

    # Metadata
    snapshot_version = Column(String, nullable=False)
    build_run_id = Column(String, nullable=False)
    source_created_at = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "ts_utc", "option_symbol", "snapshot_version", name="uq_snapshots_ts_symbol_version"
        ),
        Index("idx_snapshots_ts_bucket", "ts_utc", "delta_bucket", "tenor_days"),
        Index("idx_snapshots_version", "snapshot_version"),
    )


class NodePanel(Base):
    """Node-level panel data (features aggregated at tenor/delta bucket level)."""

    __tablename__ = "node_panel"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Node identification
    ts_utc = Column(DateTime, nullable=False)
    tenor_days = Column(Integer, nullable=False)
    delta_bucket = Column(String, nullable=False)

    # Representative option (may be NULL if node masked)
    option_symbol = Column(String, nullable=True)

    # Label-only columns (NOT model features — used solely for DHR label
    # construction in DatasetBuilder._build_dhr_labels).
    # iv_mid/iv_bid/iv_ask remain excluded to prevent label-feature leakage.
    mid_price = Column(Float, nullable=True)

    # Node features (may be NULL if masked)
    # spread_pct (derived from bid/ask) is safe — it captures liquidity, not IV level.
    spread_pct = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)

    # Derived features
    iv_change_1d = Column(Float, nullable=True)
    iv_change_5d = Column(Float, nullable=True)
    iv_vol_5d = Column(Float, nullable=True)
    iv_vol_10d = Column(Float, nullable=True)
    iv_vol_21d = Column(Float, nullable=True)
    iv_zscore_5d = Column(Float, nullable=True)
    iv_zscore_10d = Column(Float, nullable=True)
    iv_zscore_21d = Column(Float, nullable=True)
    skew_slope = Column(Float, nullable=True)
    term_slope = Column(Float, nullable=True)
    curvature = Column(Float, nullable=True)
    oi_change_5d = Column(Float, nullable=True)
    volume_ratio_5d = Column(Float, nullable=True)
    log_volume = Column(Float, nullable=True)
    log_oi = Column(Float, nullable=True)

    # Global features (denormalized)
    underlying_rv_5d = Column(Float, nullable=True)
    underlying_rv_10d = Column(Float, nullable=True)
    underlying_rv_21d = Column(Float, nullable=True)

    # Macro features (denormalized — column names match FRED series IDs)
    # Update these if config.features.macro.series changes
    DGS10_level = Column(Float, nullable=True)
    DGS10_change_1w = Column(Float, nullable=True)
    DGS2_level = Column(Float, nullable=True)
    DGS2_change_1w = Column(Float, nullable=True)
    VIXCLS_level = Column(Float, nullable=True)
    VIXCLS_change_1w = Column(Float, nullable=True)

    # Metadata
    feature_version = Column(String, nullable=False)
    is_masked = Column(Boolean, nullable=False)
    mask_reason = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "ts_utc", "tenor_days", "delta_bucket", "feature_version", name="uq_panel_node_version"
        ),
        Index("idx_panel_ts", "ts_utc"),
        Index("idx_panel_node", "tenor_days", "delta_bucket"),
    )
