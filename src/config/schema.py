"""Configuration schema with Pydantic models.

This module defines the complete configuration schema for Valorem v2.0.
All configuration is type-safe and validated at load time.
"""

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config.constants import MarketConstants, SurfaceConstants


# ============================================================================
# Project Configuration
# ============================================================================


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = Field(default="valorem", description="Project name")
    version: str = Field(default="2.0.0", description="Project version")
    description: str = Field(
        default="Production-grade volatility arbitrage trading system",
        description="Project description",
    )


# ============================================================================
# Data Provider Configuration
# ============================================================================


class DatabentoConfig(BaseModel):
    """Databento provider configuration."""

    dataset_equities: str = Field(default="DBEQ.BASIC", description="Dataset for equity bars")
    dataset_options: str = Field(default="OPRA.PILLAR", description="Dataset for options quotes")
    stype_in: str = Field(default="raw_symbol", description="Symbol type for ingestion")
    definition_query_days: int | None = Field(
        default=None,
        ge=1,
        le=30,
        description="Days to query for option definitions (null = auto-calculate from max_dte)",
    )


class DatabentoOptionsConfig(BaseModel):
    """Databento options ingestion configuration."""

    quote_schema: str = Field(default="cbbo-1m", description="Options quote schema")
    bar_schema: str = Field(default="ohlcv-1d", description="Options bar schema for volume data")
    statistics_schema: str = Field(default="statistics", description="Options statistics schema for OI data")


class DatabentoCostConfig(BaseModel):
    """Databento cost control configuration."""

    max_usd: float = Field(default=100.0, gt=0, description="Maximum USD cost per ingestion run")


class RetryConfig(BaseModel):
    """Retry configuration for API calls."""

    max_retries: int = Field(default=3, ge=0, description="Max retry attempts per API call")
    base_delay_seconds: float = Field(
        default=30.0, gt=0, description="Initial delay between retries (exponential backoff)"
    )
    max_delay_seconds: float = Field(
        default=300.0, gt=0, description="Cap on retry delay"
    )


class DatabentoIngestionConfig(BaseModel):
    """Databento ingestion sub-configuration."""

    options: DatabentoOptionsConfig = Field(default_factory=DatabentoOptionsConfig)
    cost: DatabentoCostConfig = Field(default_factory=DatabentoCostConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)


class FREDConfig(BaseModel):
    """FRED (Federal Reserve Economic Data) provider configuration."""
    pass


class ProvidersConfig(BaseModel):
    """Data providers configuration."""

    databento: DatabentoConfig = Field(default_factory=DatabentoConfig)
    fred: FREDConfig = Field(default_factory=FREDConfig)


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""

    start_date: date | None = Field(
        default=None,
        description="Default ingestion start date (CLI --start-date overrides)",
    )
    end_date: date | None = Field(
        default=None,
        description="Default ingestion end date (CLI --end-date overrides)",
    )
    databento: DatabentoIngestionConfig = Field(default_factory=DatabentoIngestionConfig)

    @model_validator(mode="after")
    def validate_date_ordering(self) -> "IngestionConfig":
        """Validate start_date < end_date when both are provided."""
        if self.start_date is not None and self.end_date is not None:
            if self.start_date >= self.end_date:
                raise ValueError(
                    f"ingestion start_date ({self.start_date}) must be before "
                    f"end_date ({self.end_date})"
                )
        return self


class DataConfig(BaseModel):
    """Data layer configuration."""

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, v: ProvidersConfig) -> ProvidersConfig:
        """Validate provider configuration."""
        # Ensure datasets are not empty
        if not v.databento.dataset_equities:
            raise ValueError("databento.dataset_equities must not be empty")
        if not v.databento.dataset_options:
            raise ValueError("databento.dataset_options must not be empty")
        return v


# ============================================================================
# Calendar Configuration
# ============================================================================


class MarketHoursConfig(BaseModel):
    """Market hours configuration."""

    market_open: time = Field(
        default=time(9, 30),
        description="Market open time (ET)",
    )
    market_close: time = Field(
        default=time(16, 0),
        description="Market close time (ET)",
    )
    timezone: str = Field(default="America/New_York", description="Market timezone")


class CalendarConfig(BaseModel):
    """Calendar and market hours configuration."""

    market_hours: MarketHoursConfig = Field(default_factory=MarketHoursConfig)


# ============================================================================
# Universe Configuration
# ============================================================================


class UniverseConfig(BaseModel):
    """Trading universe configuration."""

    underlying: str = Field(default="SPY", description="Primary underlying symbol")
    options_type: Literal["vanilla"] = Field(
        default="vanilla",
        description="Option types (vanilla only in v1)",
    )


# ============================================================================
# Surface Configuration
# ============================================================================


class DeltaBucketsConfig(BaseModel):
    """Delta bucket definitions."""

    P10: list[float] = Field(default=[0.0, -0.15], description="P10 bucket range")
    P25: list[float] = Field(default=[-0.15, -0.35], description="P25 bucket range")
    P40: list[float] = Field(default=[-0.35, -0.45], description="P40 bucket range")
    ATM: list[float] = Field(default=[-0.45, -0.55, 0.45, 0.55], description="ATM bucket ranges")
    C40: list[float] = Field(default=[0.35, 0.45], description="C40 bucket range")
    C25: list[float] = Field(default=[0.15, 0.35], description="C25 bucket range")
    C10: list[float] = Field(default=[0.0, 0.15], description="C10 bucket range")

    @field_validator("ATM")
    @classmethod
    def validate_atm(cls, v: list[float]) -> list[float]:
        """Validate ATM bucket has 4 elements."""
        if len(v) != 4:
            raise ValueError("ATM bucket must have exactly 4 elements")
        return v


class TenorBinsConfig(BaseModel):
    """Tenor bin definitions (in days)."""

    bins: list[int] = Field(
        default=list(SurfaceConstants.TENOR_DAYS_DEFAULT),
        description="Tenor bins in days",
    )


class BlackScholesConfig(BaseModel):
    """Black-Scholes solver configuration."""

    method: Literal["newton-raphson"] = Field(
        default="newton-raphson",
        description="IV solver method",
    )
    max_iterations: int = Field(default=100, gt=0, description="Maximum solver iterations")
    tolerance: float = Field(default=1e-6, gt=0, description="Convergence tolerance")
    min_iv: float = Field(default=0.001, gt=0, description="Minimum IV bound")
    max_iv: float = Field(default=5.0, gt=0, description="Maximum IV bound")


class SurfaceConfig(BaseModel):
    """Surface engine configuration."""

    delta_buckets: DeltaBucketsConfig = Field(default_factory=DeltaBucketsConfig)
    tenor_bins: TenorBinsConfig = Field(default_factory=TenorBinsConfig)
    black_scholes: BlackScholesConfig = Field(default_factory=BlackScholesConfig)
    max_iv_failure_ratio: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Maximum IV solver failure ratio before warning (0.35 = 35%)",
    )


# ============================================================================
# Features Configuration
# ============================================================================


class NodeFeaturesConfig(BaseModel):
    """Node-level feature configuration."""

    include_iv: bool = Field(default=True, description="Include implied volatility")
    include_volume: bool = Field(default=True, description="Include volume")
    include_open_interest: bool = Field(default=True, description="Include open interest")
    include_greeks: bool = Field(default=True, description="Include Greeks")
    lookback_bars: int = Field(default=60, gt=0, description="Lookback window in bars")


class GlobalFeaturesConfig(BaseModel):
    """Global surface-level feature configuration."""

    include_skew: bool = Field(default=True, description="Include skew metrics")
    include_term_structure: bool = Field(default=True, description="Include term structure")
    include_spot_features: bool = Field(default=True, description="Include spot price features")


class MacroFeaturesConfig(BaseModel):
    """Macro feature configuration."""

    series: list[str] = Field(
        default=list(MarketConstants.DEFAULT_FRED_SERIES),
        description="FRED series IDs",
    )
    lookback_days: int = Field(default=252, gt=0, description="Lookback window in days")


class MaskingConfig(BaseModel):
    """Quality-flag-based masking for excluding unreliable nodes from training."""

    enabled: bool = Field(default=False, description="Enable quality flag propagation to node masks")
    mask_low_volume: bool = Field(default=True, description="Mask nodes with FLAG_LOW_VOLUME")
    mask_crossed: bool = Field(default=True, description="Mask nodes with FLAG_CROSSED")
    mask_wide_spread: bool = Field(default=False, description="Mask nodes with FLAG_WIDE_SPREAD")
    mask_stale: bool = Field(default=False, description="Mask nodes with FLAG_STALE")
    mask_low_oi: bool = Field(default=False, description="Mask nodes with FLAG_LOW_OI")


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    node: NodeFeaturesConfig = Field(default_factory=NodeFeaturesConfig)
    global_: GlobalFeaturesConfig = Field(default_factory=GlobalFeaturesConfig, alias="global")
    macro: MacroFeaturesConfig = Field(default_factory=MacroFeaturesConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)


# ============================================================================
# Labels Configuration
# ============================================================================


class LabelsConfig(BaseModel):
    """Label generation configuration."""

    horizons: list[int] = Field(
        default=[5, 10, 21],
        description="Forecast horizons in trading days",
    )


# ============================================================================
# Dataset Configuration
# ============================================================================


class DatasetSplitsConfig(BaseModel):
    """Dataset split configuration."""

    train_start: date = Field(description="Training set start date")
    train_end: date = Field(description="Training set end date")
    val_start: date = Field(description="Validation set start date")
    val_end: date = Field(description="Validation set end date")
    test_start: date = Field(description="Test set start date")
    test_end: date = Field(description="Test set end date")

    @model_validator(mode="after")
    def validate_chronological_order(self) -> "DatasetSplitsConfig":
        """Validate splits are chronologically ordered."""
        dates = [
            ("train_start", self.train_start),
            ("train_end", self.train_end),
            ("val_start", self.val_start),
            ("val_end", self.val_end),
            ("test_start", self.test_start),
            ("test_end", self.test_end),
        ]

        for i in range(len(dates) - 1):
            if dates[i][1] >= dates[i + 1][1]:
                raise ValueError(
                    f"{dates[i][0]} ({dates[i][1]}) must be before {dates[i+1][0]} ({dates[i+1][1]})"
                )

        return self


class DatasetConfig(BaseModel):
    """Dataset builder configuration."""

    splits: DatasetSplitsConfig = Field(description="Train/val/test splits")
    min_dte: int = Field(default=7, gt=0, description="Minimum days-to-expiry")
    max_dte: int = Field(default=120, gt=0, description="Maximum days-to-expiry")
    moneyness_min: float = Field(default=0.8, description="Min strike/spot ratio for option filtering")
    moneyness_max: float = Field(default=1.2, description="Max strike/spot ratio for option filtering")


# ============================================================================
# Model Configuration
# ============================================================================


class PatchTSTConfig(BaseModel):
    """PatchTST model configuration."""

    patch_len: int = Field(default=12, gt=0, description="Patch length")
    stride: int = Field(default=6, gt=0, description="Patch stride")
    d_model: int = Field(default=128, gt=0, description="Model dimension")
    n_heads: int = Field(default=8, gt=0, description="Number of attention heads")
    d_ff: int = Field(default=256, gt=0, description="Feedforward dimension")
    n_layers: int = Field(default=3, gt=0, description="Number of transformer layers")
    dropout: float = Field(default=0.1, ge=0, le=1, description="Dropout rate")


class GNNConfig(BaseModel):
    """Graph Neural Network configuration."""

    model_type: Literal["GAT", "GCN"] = Field(default="GAT", description="GNN architecture")
    hidden_dim: int = Field(default=64, gt=0, description="Hidden dimension")
    n_layers: int = Field(default=2, gt=0, description="Number of GNN layers")
    heads: int = Field(default=4, gt=0, description="Number of attention heads (GAT only)")
    dropout: float = Field(default=0.1, ge=0, le=1, description="Dropout rate")


class ModelConfig(BaseModel):
    """ML model configuration."""

    patchtst: PatchTSTConfig = Field(default_factory=PatchTSTConfig)
    gnn: GNNConfig = Field(default_factory=GNNConfig)


# ============================================================================
# Training Configuration
# ============================================================================


class TrainingConfig(BaseModel):
    """Training configuration."""

    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto", description="Training device (auto detects best available)"
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size")
    max_epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    weight_decay: float = Field(default=1e-5, ge=0, description="Weight decay")
    early_stopping_patience: int = Field(
        default=10, gt=0, description="Early stopping patience (epochs)"
    )
    gradient_clip_val: float = Field(default=1.0, gt=0, description="Gradient clipping value")
    num_workers: int = Field(
        default=4, ge=0, description="DataLoader worker processes (0 for main thread)"
    )
    use_amp: bool = Field(
        default=True, description="Enable automatic mixed precision (CUDA only)"
    )


# ============================================================================
# Backtest Configuration
# ============================================================================


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    start_date: date = Field(description="Backtest start date")
    end_date: date = Field(description="Backtest end date")
    initial_capital: float = Field(default=100000.0, gt=0, description="Initial capital (USD)")


# ============================================================================
# Execution Configuration
# ============================================================================


class PricingConfig(BaseModel):
    """Order pricing configuration."""

    buy_at: Literal["ask"] = Field(
        default="ask",
        description="Buy execution price (ask only in v1)",
    )
    sell_at: Literal["bid"] = Field(
        default="bid",
        description="Sell execution price",
    )


class SlippageConfig(BaseModel):
    """Slippage model configuration."""

    model: Literal["fixed_bps"] = Field(default="fixed_bps", description="Slippage model type")
    fixed_bps: float = Field(default=5.0, ge=0, description="Fixed slippage in basis points")


class FeeConfig(BaseModel):
    """Trading fee configuration."""

    per_contract: float = Field(
        default=0.65, ge=0, description="Fee per contract (USD)"
    )
    per_trade_minimum: float = Field(
        default=0.0, ge=0, description="Minimum fee per trade (USD)"
    )


class SignalThresholdConfig(BaseModel):
    """Signal filtering thresholds."""

    min_edge: float = Field(default=0.01, ge=0, description="Minimum edge required to trade")
    max_uncertainty: float = Field(
        default=0.5, ge=0, le=1.0, description="Maximum uncertainty (1 - confidence) allowed"
    )
    min_confidence: float = Field(
        default=0.5, ge=0, le=1.0, description="Minimum confidence required"
    )


class SizingConfig(BaseModel):
    """Position sizing configuration."""

    method: Literal["fixed", "risk_parity", "kelly_fraction"] = Field(
        default="fixed", description="Sizing method"
    )
    base_contracts: int = Field(default=1, gt=0, description="Base number of contracts")
    max_contracts_per_trade: int = Field(
        default=10, gt=0, description="Max contracts per single trade"
    )
    max_loss_per_trade: float = Field(
        default=500.0, gt=0, description="Max loss per trade for sizing (USD)"
    )
    kelly_fraction: float = Field(
        default=0.25, gt=0, le=1.0, description="Kelly fraction (if using kelly method)"
    )
    scale_by_confidence: bool = Field(
        default=True, description="Scale size by signal confidence"
    )
    scale_by_liquidity: bool = Field(
        default=True, description="Scale size by liquidity (volume/OI)"
    )
    min_liquidity_contracts: int = Field(
        default=100, gt=0, description="Minimum daily volume for full sizing"
    )


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    pricing: PricingConfig = Field(default_factory=PricingConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    fees: FeeConfig = Field(default_factory=FeeConfig)
    signal_threshold: SignalThresholdConfig = Field(default_factory=SignalThresholdConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)


# ============================================================================
# Risk Configuration
# ============================================================================


class PerTradeRiskConfig(BaseModel):
    """Per-trade risk limits."""

    max_loss: float = Field(default=500.0, gt=0, description="Max loss per trade (USD)")
    max_contracts: int = Field(default=10, gt=0, description="Max contracts per trade")


class RiskCapsConfig(BaseModel):
    """Portfolio-level risk caps."""

    max_abs_delta: float = Field(default=100.0, gt=0, description="Max absolute portfolio delta")
    max_abs_gamma: float = Field(default=50.0, gt=0, description="Max absolute portfolio gamma")
    max_abs_vega: float = Field(default=1000.0, gt=0, description="Max absolute portfolio vega")
    max_daily_loss: float = Field(default=2000.0, gt=0, description="Max daily loss (USD)")
    max_position_size_usd: float = Field(
        default=10000.0, gt=0, description="Max single position size (USD)"
    )
    max_total_notional_usd: float = Field(
        default=50000.0, gt=0, description="Max total portfolio notional (USD)"
    )


class StressConfig(BaseModel):
    """Stress testing configuration."""

    enabled: bool = Field(default=True, description="Enable stress testing")
    underlying_shocks_pct: list[float] = Field(
        default=[-0.10, -0.05, -0.02, 0.02, 0.05, 0.10],
        description="Underlying price shocks as percentages",
    )
    iv_shocks_points: list[float] = Field(
        default=[-10.0, -5.0, 5.0, 10.0, 20.0],
        description="IV shocks in volatility points",
    )


class KillSwitchConfig(BaseModel):
    """Kill switch configuration."""

    halt_on_daily_loss: bool = Field(default=True, description="Halt on daily loss breach")
    halt_on_stress_breach: bool = Field(default=True, description="Halt on stress test breach")
    halt_on_liquidity_collapse: bool = Field(
        default=True, description="Halt on liquidity collapse"
    )
    max_daily_loss: float = Field(default=2000.0, gt=0, description="Max daily loss trigger (USD)")
    max_spread_pct: float = Field(
        default=0.10, gt=0, le=1.0, description="Max spread percentage trigger"
    )


class ExitSignalsConfig(BaseModel):
    """Exit signal generation configuration."""

    min_edge_retention: float = Field(
        default=0.3,
        ge=0,
        le=1.0,
        description="Minimum edge retention ratio to stay in position (exit if edge < entry_edge * this)",
    )
    stop_loss_pct: float = Field(
        default=0.8,
        gt=0,
        le=1.0,
        description="Stop-loss trigger as % of position max_loss (0.8 = exit at 80% of max loss)",
    )
    take_profit_pct: float = Field(
        default=0.5,
        gt=0,
        description="Take-profit trigger as % of max_loss (0.5 = exit at 50% profit of max_loss)",
    )
    min_dte_exit: int = Field(
        default=3,
        ge=0,
        description="Minimum days-to-expiry before forced exit (time decay protection)",
    )


class DriftBandsConfig(BaseModel):
    """Greek drift bands for rebalancing triggers."""

    delta_target: float = Field(
        default=0.0,
        description="Target portfolio delta (typically 0 for delta-neutral)",
    )
    delta_max_drift: float = Field(
        default=50.0,
        gt=0,
        description="Maximum delta drift before rebalancing",
    )
    vega_target: float = Field(
        default=0.0,
        description="Target portfolio vega",
    )
    vega_max_drift: float = Field(
        default=500.0,
        gt=0,
        description="Maximum vega drift before rebalancing",
    )
    gamma_max_drift: float = Field(
        default=25.0,
        gt=0,
        description="Maximum gamma drift before rebalancing",
    )


class RebalancingConfig(BaseModel):
    """Greek drift rebalancing configuration."""

    enabled: bool = Field(default=True, description="Enable automatic rebalancing")
    check_interval_seconds: int = Field(
        default=60,
        gt=0,
        description="Interval between rebalance checks (seconds)",
    )
    strategy: Literal["close_first", "add_hedge", "replace"] = Field(
        default="close_first",
        description="Rebalancing strategy: close drifted positions, add hedges, or replace",
    )
    max_trades_per_rebalance: int = Field(
        default=3,
        gt=0,
        description="Maximum number of trades per rebalance cycle",
    )


class PositionManagementConfig(BaseModel):
    """Position lifecycle and exit management configuration."""

    exit_signals: ExitSignalsConfig = Field(default_factory=ExitSignalsConfig)
    rebalancing: RebalancingConfig = Field(default_factory=RebalancingConfig)
    drift_bands: DriftBandsConfig = Field(default_factory=DriftBandsConfig)


class CovarianceConfig(BaseModel):
    """Marchenko-Pastur covariance cleaning configuration."""

    enabled: bool = Field(default=False, description="Enable covariance-aware sizing/risk (opt-in)")
    method: Literal["marchenko_pastur", "ledoit_wolf"] = Field(
        default="marchenko_pastur", description="Covariance cleaning method"
    )
    returns_source: Literal["iv_change_1d", "iv_change_5d", "dhr_5d"] = Field(
        default="iv_change_1d", description="Node returns column for covariance estimation"
    )
    window: int = Field(default=252, gt=0, description="Rolling window in observations")
    min_observations: int = Field(
        default=63, gt=0, description="Minimum observations before estimation (one quarter)"
    )
    refresh_frequency: Literal["daily", "weekly"] = Field(
        default="daily", description="How often to recompute the covariance matrix"
    )
    mp_eigenvalue_method: Literal["clip_to_mean", "clip_to_zero", "shrink_toward_identity"] = Field(
        default="clip_to_mean", description="How to treat noise eigenvalues"
    )
    condition_number_cap: float = Field(
        default=100.0, gt=0, description="Max condition number after cleaning; regularize if exceeded"
    )
    fallback_to_ledoit_wolf: bool = Field(
        default=True, description="Fall back to Ledoit-Wolf if MP cleaning produces degenerate result"
    )
    signal_max_correlation: float = Field(
        default=0.85, gt=0, le=1.0, description="Max pairwise correlation for signal diversity filter"
    )


class RiskConfig(BaseModel):
    """Risk management configuration."""

    per_trade: PerTradeRiskConfig = Field(default_factory=PerTradeRiskConfig)
    caps: RiskCapsConfig = Field(default_factory=RiskCapsConfig)
    stress: StressConfig = Field(default_factory=StressConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    position_management: PositionManagementConfig = Field(
        default_factory=PositionManagementConfig
    )
    covariance: CovarianceConfig = Field(default_factory=CovarianceConfig)


# ============================================================================
# Paper Trading Configuration
# ============================================================================


class PaperConfig(BaseModel):
    """Paper trading configuration."""

    broker_api_key_env: str = Field(
        default="PAPER_BROKER_API_KEY",
        description="Paper broker API key environment variable",
    )
    account_id_env: str = Field(
        default="PAPER_ACCOUNT_ID",
        description="Paper account ID environment variable",
    )
    enabled: bool = Field(default=False, description="Enable paper trading")

    # Trading loop settings
    loop_interval_seconds: int = Field(
        default=60,
        ge=0,
        description="Seconds between trading loop iterations (0 = no delay)",
    )
    halt_on_error: bool = Field(
        default=True,
        description="Halt trading on unexpected errors",
    )
    max_loop_iterations: int = Field(
        default=0,
        ge=0,
        description="Max loop iterations (0 = unlimited)",
    )

    # Data settings
    lookback_minutes: int = Field(
        default=5,
        gt=0,
        description="Minutes of recent data to fetch for surface building",
    )

    # State persistence
    state_dir: str = Field(
        default="data/paper/state",
        description="Directory for state snapshots",
    )
    save_state_interval: int = Field(
        default=1,
        gt=0,
        description="Save state every N loop iterations",
    )


# ============================================================================
# Logging Configuration
# ============================================================================


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    format: Literal["text", "json"] = Field(default="text", description="Log format")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    console_enabled: bool = Field(default=True, description="Enable console logging")


# ============================================================================
# Paths Configuration
# ============================================================================


class PathsConfig(BaseModel):
    """File paths configuration (relative to repo root)."""

    data_dir: str = Field(default="data", description="Data directory")
    db_path: str = Field(default="data/db.sqlite", description="SQLite database path")
    parquet_dir: str = Field(default="data/parquet", description="Parquet archive directory")
    manifest_dir: str = Field(default="data/manifest", description="Manifest directory")
    artifacts_dir: str = Field(default="artifacts", description="Artifacts directory")
    checkpoints_dir: str = Field(
        default="artifacts/checkpoints",
        description="Model checkpoints directory",
    )
    reports_dir: str = Field(default="artifacts/reports", description="Reports directory")
    logs_dir: str = Field(default="artifacts/logs", description="Logs directory")


# ============================================================================
# Root Configuration Schema
# ============================================================================


class ConfigSchema(BaseModel):
    """Root configuration schema for Valorem v2.0."""

    version: Literal["v1"] = Field(default="v1", description="Config schema version")
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    surface: SurfaceConfig = Field(default_factory=SurfaceConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    dataset: DatasetConfig = Field(description="Dataset configuration")
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    backtest: BacktestConfig = Field(description="Backtest configuration")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    model_config = {
        "extra": "forbid",  # Fail on unknown fields
        "validate_assignment": True,  # Validate on attribute assignment
    }
