"""Configuration schema with Pydantic models.

This module defines the complete configuration schema for Rhubarb v2.0.
All configuration is type-safe and validated at load time.
"""

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Project Configuration
# ============================================================================


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = Field(default="rhubarb", description="Project name")
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

    api_key_env: str = Field(default="DATABENTO_API_KEY", description="API key environment variable")
    dataset_equities: str = Field(default="GLBX.MDP3", description="Dataset for equity bars")
    dataset_options: str = Field(default="OPRA", description="Dataset for options quotes")
    stype_in: str = Field(default="raw_symbol", description="Symbol type for ingestion")
    options_schema: str = Field(default="cbbo-1m", description="Options schema (cbbo-1m)")


class DatabentoOptionsConfig(BaseModel):
    """Databento options ingestion configuration."""

    schema: str = Field(default="cbbo-1m", description="Options quote schema")


class DatabentoCostConfig(BaseModel):
    """Databento cost control configuration."""

    max_usd: float = Field(default=100.0, gt=0, description="Maximum USD cost per ingestion run")


class DatabentoIngestionConfig(BaseModel):
    """Databento ingestion sub-configuration."""

    options: DatabentoOptionsConfig = Field(default_factory=DatabentoOptionsConfig)
    cost: DatabentoCostConfig = Field(default_factory=DatabentoCostConfig)


class FREDConfig(BaseModel):
    """FRED (Federal Reserve Economic Data) provider configuration."""

    api_key_env: str = Field(default="FRED_API_KEY", description="API key environment variable")
    base_url: str = Field(
        default="https://api.stlouisfed.org/fred",
        description="FRED API base URL",
    )


class ProvidersConfig(BaseModel):
    """Data providers configuration."""

    databento: DatabentoConfig = Field(default_factory=DatabentoConfig)
    fred: FREDConfig = Field(default_factory=FREDConfig)


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""

    databento: DatabentoIngestionConfig = Field(default_factory=DatabentoIngestionConfig)


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

    P10: list[float] = Field(default=[-1.0, -0.15], description="P10 bucket range")
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
        default=[7, 14, 30, 60, 90, 120],
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
        default=["DGS10", "VIXCLS"],
        description="FRED series IDs",
    )
    lookback_days: int = Field(default=252, gt=0, description="Lookback window in days")


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    node: NodeFeaturesConfig = Field(default_factory=NodeFeaturesConfig)
    global_: GlobalFeaturesConfig = Field(default_factory=GlobalFeaturesConfig, alias="global")
    macro: MacroFeaturesConfig = Field(default_factory=MacroFeaturesConfig)


# ============================================================================
# Labels Configuration
# ============================================================================


class LabelsConfig(BaseModel):
    """Label generation configuration."""

    horizons: list[int] = Field(
        default=[5, 10, 21],
        description="Forecast horizons in trading days",
    )
    target_metric: Literal["variance_misprice"] = Field(
        default="variance_misprice",
        description="Target metric to predict",
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

    device: Literal["cpu", "cuda", "mps"] = Field(default="cuda", description="Training device")
    batch_size: int = Field(default=32, gt=0, description="Batch size")
    max_epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    weight_decay: float = Field(default=1e-5, ge=0, description="Weight decay")
    early_stopping_patience: int = Field(
        default=10, gt=0, description="Early stopping patience (epochs)"
    )
    gradient_clip_val: float = Field(default=1.0, gt=0, description="Gradient clipping value")
    checkpoint_every_n_epochs: int = Field(
        default=5, gt=0, description="Checkpoint frequency (epochs)"
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


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    pricing: PricingConfig = Field(default_factory=PricingConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)


# ============================================================================
# Risk Configuration
# ============================================================================


class RiskCapsConfig(BaseModel):
    """Risk limit caps."""

    max_portfolio_delta: float = Field(default=100.0, gt=0, description="Max portfolio delta")
    max_portfolio_vega: float = Field(default=1000.0, gt=0, description="Max portfolio vega")
    max_position_size_usd: float = Field(
        default=10000.0, gt=0, description="Max single position size (USD)"
    )
    max_total_notional_usd: float = Field(
        default=50000.0, gt=0, description="Max total portfolio notional (USD)"
    )


class RiskConfig(BaseModel):
    """Risk management configuration."""

    caps: RiskCapsConfig = Field(default_factory=RiskCapsConfig)
    stress_test_enabled: bool = Field(default=True, description="Enable stress testing")


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


# ============================================================================
# Harness Configuration
# ============================================================================


class HarnessConfig(BaseModel):
    """Research harness configuration."""

    enabled: bool = Field(default=False, description="Enable research harness")
    port: int = Field(default=8501, gt=0, lt=65536, description="Streamlit port")


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
    logs_dir: str = Field(default="logs", description="Logs directory")


# ============================================================================
# Root Configuration Schema
# ============================================================================


class ConfigSchema(BaseModel):
    """Root configuration schema for Rhubarb v2.0."""

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
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    model_config = {
        "extra": "forbid",  # Fail on unknown fields
        "validate_assignment": True,  # Validate on attribute assignment
    }
