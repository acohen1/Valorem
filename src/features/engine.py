"""Feature engine orchestrator for feature generation.

This module orchestrates all feature generators:
- Node features: IV features, microstructure features, surface features
- Global features: returns, realized volatility, drawdown
- Macro features: FRED series transforms with release-time alignment

The engine coordinates the feature pipeline, merges features,
validates for data leakage, and writes to the node_panel table.

Architecture:
    The FeatureEngine uses dependency injection for all components. Dependencies
    can be injected directly into the constructor for testing, or created via
    the `create_feature_engine()` factory for production use.

Example:
    # Testing with mocks
    engine = FeatureEngine(
        config=config,
        raw_repo=mock_raw_repo,
        derived_repo=mock_derived_repo,
        node_generator=mock_node_gen,
    )

    # Production with factory
    engine = create_feature_engine(config=config)
"""

import logging
from dataclasses import dataclass, field

from src.config.constants import MarketConstants
from src.config.schema import MaskingConfig
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy.engine import Engine

from src.data.storage.repository import DerivedRepository, RawRepository
from src.features.global_.realized_vol import RealizedVolConfig, RealizedVolGenerator
from src.features.global_.returns import ReturnsConfig, ReturnsGenerator
from src.features.macro.alignment import AlignmentConfig
from src.features.macro.transforms import MacroTransformConfig, MacroTransformGenerator
from src.features.node.iv_features import IVFeatureConfig, IVFeatureGenerator
from src.features.node.microstructure import (
    MicrostructureConfig,
    MicrostructureFeatureGenerator,
)
from src.features.node.surface import SurfaceFeatureConfig, SurfaceFeatureGenerator
from src.surface.quality.filters import QualityFilter


@dataclass
class NodeFeatureConfig:
    """Configuration for node feature generation.

    Attributes:
        iv_config: IV feature configuration
        microstructure_config: Microstructure feature configuration
        surface_config: Surface feature configuration
        include_iv: Whether to compute IV features
        include_microstructure: Whether to compute microstructure features
        include_surface: Whether to compute surface features
    """

    iv_config: IVFeatureConfig = field(default_factory=IVFeatureConfig)
    microstructure_config: MicrostructureConfig = field(
        default_factory=MicrostructureConfig
    )
    surface_config: SurfaceFeatureConfig = field(default_factory=SurfaceFeatureConfig)
    include_iv: bool = True
    include_microstructure: bool = True
    include_surface: bool = True


@dataclass
class FeatureResult:
    """Result of feature generation.

    Attributes:
        feature_version: Version string for this feature set
        row_count: Number of rows in output
        feature_count: Number of features generated
        start_ts: Start timestamp of data
        end_ts: End timestamp of data
        nodes_processed: Number of unique nodes
        features_generated: List of feature column names
    """

    feature_version: str
    row_count: int
    feature_count: int
    start_ts: datetime
    end_ts: datetime
    nodes_processed: int
    features_generated: list[str] = field(default_factory=list)


class NodeFeatureGenerator:
    """Orchestrate node-level feature generation.

    This class coordinates all node feature generators to produce
    a complete feature set from surface snapshots.

    The pipeline:
    1. Validate input data
    2. Compute IV features (longitudinal, per-node)
    3. Compute microstructure features (longitudinal, per-node)
    4. Compute surface features (cross-sectional)
    5. Return combined feature DataFrame

    All features use only past data (no future leakage).

    Example:
        config = NodeFeatureConfig()
        generator = NodeFeatureGenerator(config)
        result_df, result = generator.generate(
            surface_df,
            feature_version="v1.0"
        )
    """

    def __init__(
        self,
        config: NodeFeatureConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize node feature generator.

        Args:
            config: Feature configuration. Uses defaults if None.
            logger: Optional logger instance
        """
        self._config = config or NodeFeatureConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Initialize sub-generators
        self._iv_generator = IVFeatureGenerator(self._config.iv_config)
        self._micro_generator = MicrostructureFeatureGenerator(
            self._config.microstructure_config
        )
        self._surface_generator = SurfaceFeatureGenerator(self._config.surface_config)

    def generate(
        self,
        surface_df: pd.DataFrame,
        feature_version: str = "v1.0",
    ) -> tuple[pd.DataFrame, FeatureResult]:
        """Generate all node features from surface snapshots.

        Args:
            surface_df: Surface snapshot DataFrame with required columns:
                - ts_utc: Timestamp
                - tenor_days: Tenor in days
                - delta_bucket: Delta bucket name
                - iv_mid: Mid IV value
                - volume: Volume (optional, for microstructure)
                - open_interest: Open interest (optional, for microstructure)
                - delta: Actual delta (optional, for surface features)
            feature_version: Version string for output

        Returns:
            Tuple of (features_df, FeatureResult)
        """
        if surface_df.empty:
            return surface_df.copy(), FeatureResult(
                feature_version=feature_version,
                row_count=0,
                feature_count=0,
                start_ts=datetime.min,
                end_ts=datetime.min,
                nodes_processed=0,
            )

        # Track original columns for feature counting
        original_cols = set(surface_df.columns)

        # Validate required columns
        required_cols = ["ts_utc", "tenor_days", "delta_bucket", "iv_mid"]
        missing = [c for c in required_cols if c not in surface_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self._logger.debug(f"Generating features for {len(surface_df)} surface rows")

        # Start with a copy
        df = surface_df.copy()

        # Step 1: IV features (longitudinal)
        if self._config.include_iv:
            self._logger.debug("Computing IV features...")
            df = self._iv_generator.generate(df)
            self._logger.debug(f"IV features complete: {len(df.columns) - len(original_cols)} new columns")

        # Step 2: Microstructure features (longitudinal)
        has_micro_inputs = "volume" in df.columns or "open_interest" in df.columns
        if self._config.include_microstructure and has_micro_inputs:
            self._logger.debug("Computing microstructure features...")
            cols_before = len(df.columns)
            df = self._micro_generator.generate(df)
            self._logger.debug(f"Microstructure features complete: {len(df.columns) - cols_before} new columns")

        # Step 3: Surface features (cross-sectional)
        if self._config.include_surface:
            self._logger.debug("Computing surface features...")
            cols_before = len(df.columns)
            df = self._surface_generator.generate(df)
            self._logger.debug(f"Surface features complete: {len(df.columns) - cols_before} new columns")

        # Identify new feature columns
        new_cols = [c for c in df.columns if c not in original_cols]

        # Count unique nodes
        nodes = df.groupby(["tenor_days", "delta_bucket"]).ngroups

        # Build result
        result = FeatureResult(
            feature_version=feature_version,
            row_count=len(df),
            feature_count=len(new_cols),
            start_ts=df["ts_utc"].min(),
            end_ts=df["ts_utc"].max(),
            nodes_processed=nodes,
            features_generated=new_cols,
        )

        self._logger.info(
            f"Feature generation complete: {result.feature_count} features, "
            f"{result.nodes_processed} nodes, {result.row_count} rows"
        )

        return df, result

    def generate_iv_only(self, surface_df: pd.DataFrame) -> pd.DataFrame:
        """Generate only IV features (utility method).

        Args:
            surface_df: Surface snapshot DataFrame

        Returns:
            DataFrame with IV features added
        """
        return self._iv_generator.generate(surface_df)

    def generate_microstructure_only(self, surface_df: pd.DataFrame) -> pd.DataFrame:
        """Generate only microstructure features (utility method).

        Args:
            surface_df: Surface snapshot DataFrame

        Returns:
            DataFrame with microstructure features added
        """
        return self._micro_generator.generate(surface_df)

    def generate_surface_only(self, surface_df: pd.DataFrame) -> pd.DataFrame:
        """Generate only surface features (utility method).

        Args:
            surface_df: Surface snapshot DataFrame

        Returns:
            DataFrame with surface features added
        """
        return self._surface_generator.generate(surface_df)

    def get_feature_names(self) -> list[str]:
        """Get list of all feature column names that will be generated.

        Returns:
            List of feature column names
        """
        features = []

        if self._config.include_iv:
            # IV change features
            for period in self._config.iv_config.change_periods:
                features.append(f"iv_change_{period}d")

            # IV volatility features
            for window in self._config.iv_config.rolling_windows:
                features.append(f"iv_vol_{window}d")
                features.append(f"iv_zscore_{window}d")

        if self._config.include_microstructure:
            # Volume features
            for window in self._config.microstructure_config.volume_ratio_windows:
                features.append(f"volume_ratio_{window}d")
            features.append("log_volume")

            # OI features
            for period in self._config.microstructure_config.oi_change_periods:
                features.append(f"oi_change_{period}d")
            features.append("log_oi")

        if self._config.include_surface:
            features.extend([
                "skew_slope",
                "skew_convexity",
                "term_slope",
                "atm_spread",
                "curvature",
            ])

        return features


@dataclass
class GlobalFeatureConfig:
    """Configuration for global underlying features.

    Attributes:
        returns_config: Configuration for returns generator
        realized_vol_config: Configuration for realized vol generator
        include_returns: Whether to include return features
        include_realized_vol: Whether to include realized vol features
    """

    returns_config: ReturnsConfig = field(default_factory=ReturnsConfig)
    realized_vol_config: RealizedVolConfig = field(default_factory=RealizedVolConfig)
    include_returns: bool = True
    include_realized_vol: bool = True


@dataclass
class FeatureEngineConfig:
    """Configuration for the full feature engine.

    Attributes:
        node_config: Configuration for node features
        global_config: Configuration for global underlying features
        macro_config: Configuration for macro features
        underlying_symbol: Symbol for underlying (e.g., "SPY")
        fred_series: List of FRED series to include
        lookback_buffer_days: Extra days to load for rolling calculations
        include_node_features: Whether to compute node features
        include_global_features: Whether to compute global features
        include_macro_features: Whether to compute macro features
        fail_on_validation_issues: If True, raises ValueError when leakage or
            feature-range validation reports any issues (warnings or errors).
    """

    node_config: NodeFeatureConfig = field(default_factory=NodeFeatureConfig)
    global_config: GlobalFeatureConfig = field(default_factory=GlobalFeatureConfig)
    macro_config: MacroTransformConfig = field(default_factory=MacroTransformConfig)
    underlying_symbol: str = "SPY"
    fred_series: list[str] = field(
        default_factory=lambda: list(MarketConstants.DEFAULT_FRED_SERIES)
    )
    lookback_buffer_days: int = 60
    include_node_features: bool = True
    include_global_features: bool = True
    include_macro_features: bool = True
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    fail_on_validation_issues: bool = False


@dataclass
class FeatureEngineResult:
    """Result of full feature engine build.

    Attributes:
        feature_version: Version string for this feature set
        surface_version: Version of input surface data
        row_count: Number of rows in final panel
        feature_count: Number of features generated
        start_ts: Start timestamp of data
        end_ts: End timestamp of data
        nodes_processed: Number of unique nodes
        node_features_count: Number of node features
        global_features_count: Number of global features
        macro_features_count: Number of macro features
        validation_passed: Whether anti-leakage validation passed
    """

    feature_version: str
    surface_version: str
    row_count: int
    feature_count: int
    start_ts: datetime
    end_ts: datetime
    nodes_processed: int
    node_features_count: int = 0
    global_features_count: int = 0
    macro_features_count: int = 0
    validation_passed: bool = True


class FeatureEngine:
    """Orchestrate feature generation across all feature families.

    This class coordinates the full feature pipeline:
    1. Load surface snapshots from database
    2. Load underlying bars for global features
    3. Load FRED series for macro features
    4. Generate node-level features (IV, microstructure, surface)
    5. Generate global features (returns, realized vol, drawdown)
    6. Generate macro features (levels, changes, z-scores)
    7. Merge all features with time-aware joins
    8. Validate for data leakage
    9. Write node_panel to database

    Example:
        config = FeatureEngineConfig()
        engine = FeatureEngine(config, db_engine)
        result = engine.build_feature_panel(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            surface_version="v1.0",
            feature_version="v1.0",
        )
    """

    def __init__(
        self,
        config: FeatureEngineConfig | None = None,
        engine: Engine | None = None,
        raw_repo: RawRepository | None = None,
        derived_repo: DerivedRepository | None = None,
        logger: logging.Logger | None = None,
        # Dependency injection for sub-generators (all optional)
        node_generator: NodeFeatureGenerator | None = None,
        returns_generator: ReturnsGenerator | None = None,
        realized_vol_generator: RealizedVolGenerator | None = None,
        macro_generator: MacroTransformGenerator | None = None,
    ):
        """Initialize the feature engine.

        All component dependencies can be injected for testing. If not provided,
        they are created with default configurations.

        Args:
            config: Feature engine configuration. Uses defaults if None.
            engine: SQLAlchemy engine. Creates default if None.
            raw_repo: Raw data repository. Creates from engine if None.
            derived_repo: Derived data repository. Creates from engine if None.
            logger: Optional logger instance
            node_generator: Node feature generator (inject for testing)
            returns_generator: Returns feature generator (inject for testing)
            realized_vol_generator: Realized vol generator (inject for testing)
            macro_generator: Macro feature generator (inject for testing)
        """
        self._config = config or FeatureEngineConfig()
        self._logger = logger or logging.getLogger(__name__)

        # Set up database access
        if engine is None:
            raise ValueError("engine is required — pass a SQLAlchemy Engine or DatabaseEngine")
        else:
            # Handle both DatabaseEngine wrapper and raw SQLAlchemy Engine
            from src.data.storage.engine import DatabaseEngine

            if isinstance(engine, DatabaseEngine):
                sa_engine = engine.engine
            else:
                sa_engine = engine
        self._engine = sa_engine

        self._raw_repo = raw_repo or RawRepository(sa_engine)
        self._derived_repo = derived_repo or DerivedRepository(sa_engine)

        # Initialize node feature generator (use injected or create default)
        self._node_generator = node_generator or NodeFeatureGenerator(
            self._config.node_config, self._logger
        )

        # Initialize global feature generators (use injected or create defaults)
        self._returns_generator = returns_generator or ReturnsGenerator(
            self._config.global_config.returns_config
        )
        self._realized_vol_generator = realized_vol_generator or RealizedVolGenerator(
            self._config.global_config.realized_vol_config
        )

        # Initialize macro feature generator (use injected or create default)
        self._macro_generator = macro_generator or MacroTransformGenerator(
            self._config.macro_config
        )

    def build_feature_panel(
        self,
        start: datetime,
        end: datetime,
        surface_version: str,
        feature_version: str,
        write_to_db: bool = True,
    ) -> tuple[pd.DataFrame, FeatureEngineResult]:
        """Build node_panel with all features.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            surface_version: Version of surface snapshots to use
            feature_version: Version string for output features
            write_to_db: Whether to write results to database

        Returns:
            Tuple of (panel_df, FeatureEngineResult)
        """
        self._logger.info(
            f"Building feature panel: {start} to {end}, "
            f"surface={surface_version}, features={feature_version}"
        )

        # Calculate buffered start for lookback
        buffer_start = start - timedelta(days=self._config.lookback_buffer_days)

        # Step 1: Load surface snapshots
        self._logger.info("Loading surface snapshots...")
        surface_df = self._derived_repo.read_surface_snapshots(
            start=buffer_start,
            end=end,
            version=surface_version,
        )

        if surface_df.empty:
            self._logger.warning("No surface snapshots found")
            return pd.DataFrame(), FeatureEngineResult(
                feature_version=feature_version,
                surface_version=surface_version,
                row_count=0,
                feature_count=0,
                start_ts=start,
                end_ts=end,
                nodes_processed=0,
            )

        self._logger.info(f"Loaded {len(surface_df)} surface rows")

        # Step 2: Load underlying bars for global features
        underlying_df = None
        if self._config.include_global_features:
            self._logger.info("Loading underlying bars...")
            underlying_df = self._raw_repo.read_underlying_bars(
                symbol=self._config.underlying_symbol,
                start=buffer_start,
                end=end,
                timeframe="1m",
            )
            self._logger.info(f"Loaded {len(underlying_df)} underlying bars")

        # Step 3: Load FRED series for macro features
        fred_data = {}
        if self._config.include_macro_features:
            self._logger.info("Loading FRED series...")
            for series_id in self._config.fred_series:
                fred_df = self._raw_repo.read_fred_series(
                    series_id=series_id,
                    start=buffer_start - timedelta(days=365),  # Extra buffer for z-score
                    end=end,
                )
                if not fred_df.empty:
                    fred_data[series_id] = fred_df
                    self._logger.info(f"Loaded {len(fred_df)} rows for {series_id}")

        # Step 4: Generate node-level features
        node_features_df = surface_df.copy()
        node_feature_cols = []
        if self._config.include_node_features:
            self._logger.debug("Generating node features...")
            node_features_df, node_result = self._node_generator.generate(
                surface_df, feature_version
            )
            node_feature_cols = node_result.features_generated
            self._logger.debug(f"Generated {len(node_feature_cols)} node features")

        # Step 5: Generate global features
        global_features_df = None
        global_feature_cols = []
        if self._config.include_global_features and underlying_df is not None and not underlying_df.empty:
            self._logger.debug("Generating global features...")
            global_features_df = self._generate_global_features(underlying_df)
            global_feature_cols = [
                c for c in global_features_df.columns if c != "ts_utc"
            ]
            self._logger.debug(f"Generated {len(global_feature_cols)} global features")

        # Step 6: Generate macro features
        macro_features_df = None
        macro_feature_cols = []
        if self._config.include_macro_features and fred_data:
            self._logger.debug("Generating macro features...")
            macro_features_df = self._macro_generator.generate_multi(fred_data)
            macro_feature_cols = [
                c for c in macro_features_df.columns if c != "ts_utc"
            ]
            self._logger.debug(f"Generated {len(macro_feature_cols)} macro features")

        # Step 7: Merge all features
        self._logger.debug("Merging features...")
        panel_df = self._merge_features(
            node_features_df,
            global_features_df,
            macro_features_df,
        )

        # Filter to requested date range (after lookback buffer)
        panel_df = panel_df[panel_df["ts_utc"] >= start].copy()
        panel_df = panel_df.reset_index(drop=True)

        self._logger.debug(f"Merged panel has {len(panel_df)} rows")

        # Step 8: Validate for leakage and feature ranges
        validation_passed = True
        try:
            from src.features.validators import FeatureValidator
            validator = FeatureValidator()
            validation_result = validator.validate_no_future_leakage(
                panel_df, lookback_buffered=self._config.lookback_buffer_days > 0
            )
            validation_passed = validation_result.passed
            if not validation_passed:
                self._logger.warning(
                    f"Leakage validation failed: {validation_result.issues}"
                )
            elif validation_result.issues:
                for issue in validation_result.issues:
                    self._logger.warning(
                        f"Leakage validation warning: {issue.message}"
                    )

            range_result = validator.validate_feature_ranges(panel_df)
            if not range_result.passed:
                self._logger.warning(
                    f"Feature range validation failed: {range_result.issues}"
                )
            elif range_result.issues:
                for issue in range_result.issues:
                    self._logger.warning(f"Feature range warning: {issue.message}")

            if self._config.fail_on_validation_issues:
                all_issues = validation_result.issues + range_result.issues
                if all_issues:
                    summary = "; ".join(
                        f"{issue.severity.value}: {issue.message}" for issue in all_issues
                    )
                    raise ValueError(
                        "Feature validation failed by policy "
                        "(fail_on_validation_issues=True): "
                        f"{summary}"
                    )
        except ImportError:
            self._logger.debug("FeatureValidator not available, skipping validation")

        # Step 8b: Apply quality-flag-based masking
        if self._config.masking.enabled:
            panel_df = self._apply_quality_masks(panel_df, surface_df)

        # Step 9: Write to database
        if write_to_db and not panel_df.empty:
            self._logger.info(f"Writing {len(panel_df)} rows to node_panel...")
            self._derived_repo.write_node_panel(panel_df, feature_version)

        # Count unique nodes
        nodes_processed = 0
        if not panel_df.empty and "tenor_days" in panel_df.columns and "delta_bucket" in panel_df.columns:
            nodes_processed = panel_df.groupby(["tenor_days", "delta_bucket"]).ngroups

        # Calculate total feature count
        all_feature_cols = set(node_feature_cols + global_feature_cols + macro_feature_cols)

        # Build result
        result = FeatureEngineResult(
            feature_version=feature_version,
            surface_version=surface_version,
            row_count=len(panel_df),
            feature_count=len(all_feature_cols),
            start_ts=panel_df["ts_utc"].min() if not panel_df.empty else start,
            end_ts=panel_df["ts_utc"].max() if not panel_df.empty else end,
            nodes_processed=nodes_processed,
            node_features_count=len(node_feature_cols),
            global_features_count=len(global_feature_cols),
            macro_features_count=len(macro_feature_cols),
            validation_passed=validation_passed,
        )

        self._logger.info(
            f"Feature panel complete: {result.row_count} rows, "
            f"{result.feature_count} features, {result.nodes_processed} nodes"
        )

        return panel_df, result

    def _generate_global_features(self, underlying_df: pd.DataFrame) -> pd.DataFrame:
        """Generate global features from underlying bars.

        Args:
            underlying_df: DataFrame with underlying bars (ts_utc, close, etc.)

        Returns:
            DataFrame with ts_utc and global feature columns
        """
        if underlying_df.empty:
            return pd.DataFrame(columns=["ts_utc"])

        df = underlying_df.copy()

        # Generate returns
        if self._config.global_config.include_returns:
            df = self._returns_generator.generate(df)

        # Generate realized vol
        if self._config.global_config.include_realized_vol:
            df = self._realized_vol_generator.generate(df)

        # Select only ts_utc and feature columns (not raw OHLCV)
        feature_cols = ["ts_utc"]
        for col in df.columns:
            if any(prefix in col for prefix in [
                "returns_", "log_returns_",
                "rv_", "realized_vol_", "vol_of_vol_",
                "drawdown", "max_drawdown_"
            ]):
                feature_cols.append(col)

        result = df[feature_cols].drop_duplicates(subset=["ts_utc"])

        # Rename rv_* to underlying_rv_* to match node_panel schema
        rename_map = {c: f"underlying_{c}" for c in result.columns if c.startswith("rv_")}
        if rename_map:
            result = result.rename(columns=rename_map)

        return result

    def _merge_features(
        self,
        node_df: pd.DataFrame,
        global_df: pd.DataFrame | None,
        macro_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Merge node, global, and macro features with time-aware joins.

        Uses merge_asof for global and macro features to ensure
        proper forward-fill behavior (no future data leakage).

        Args:
            node_df: DataFrame with node features (per node per timestamp)
            global_df: DataFrame with global features (one row per timestamp)
            macro_df: DataFrame with macro features (one row per timestamp)

        Returns:
            Merged DataFrame with all features
        """
        result = node_df.copy()

        # Ensure ts_utc is datetime
        result["ts_utc"] = pd.to_datetime(result["ts_utc"])

        # Merge global features using merge_asof (forward-fill)
        if global_df is not None and not global_df.empty:
            global_df = global_df.copy()
            global_df["ts_utc"] = pd.to_datetime(global_df["ts_utc"])
            global_df = global_df.sort_values("ts_utc")

            # Sort result for merge_asof
            result = result.sort_values("ts_utc")

            result = pd.merge_asof(
                result,
                global_df,
                on="ts_utc",
                direction="backward",  # Only use data from the past
            )

        # Merge macro features using merge_asof (forward-fill)
        if macro_df is not None and not macro_df.empty:
            macro_df = macro_df.copy()
            macro_df["ts_utc"] = pd.to_datetime(macro_df["ts_utc"])
            macro_df = macro_df.sort_values("ts_utc")

            # Sort result for merge_asof
            result = result.sort_values("ts_utc")

            result = pd.merge_asof(
                result,
                macro_df,
                on="ts_utc",
                direction="backward",  # Only use data from the past
            )

        return result

    def _apply_quality_masks(
        self,
        panel_df: pd.DataFrame,
        surface_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Propagate surface quality flags to node panel is_masked/mask_reason.

        For each row in the panel, looks up the corresponding surface snapshot
        flags and sets is_masked=True with an appropriate mask_reason when
        quality issues are detected.

        Args:
            panel_df: Node panel DataFrame with features
            surface_df: Surface snapshot DataFrame with 'flags' column

        Returns:
            Panel DataFrame with is_masked and mask_reason updated
        """
        if "flags" not in surface_df.columns:
            self._logger.debug("No flags column in surface_df, skipping quality masking")
            return panel_df

        # Build list of (flag_value, reason) pairs to check
        flags_to_mask: list[tuple[int, str]] = []
        if self._config.masking.mask_low_volume:
            flags_to_mask.append((QualityFilter.FLAG_LOW_VOLUME, "low_volume"))
        if self._config.masking.mask_crossed:
            flags_to_mask.append((QualityFilter.FLAG_CROSSED, "crossed_quote"))
        if self._config.masking.mask_wide_spread:
            flags_to_mask.append((QualityFilter.FLAG_WIDE_SPREAD, "wide_spread"))
        if self._config.masking.mask_stale:
            flags_to_mask.append((QualityFilter.FLAG_STALE, "stale_quote"))
        if self._config.masking.mask_low_oi:
            flags_to_mask.append((QualityFilter.FLAG_LOW_OI, "low_oi"))

        if not flags_to_mask:
            return panel_df

        # Build flag lookup: aggregate flags per node key
        # Surface has one row per representative per (ts_utc, tenor_days, delta_bucket)
        join_cols = ["ts_utc", "tenor_days", "delta_bucket"]
        if not all(c in surface_df.columns for c in join_cols):
            self._logger.warning("Surface DF missing join columns for quality masking")
            return panel_df

        flag_lookup = (
            surface_df.groupby(join_cols)["flags"]
            .first()
            .reset_index()
        )

        # Merge flags onto panel
        panel_df = panel_df.merge(
            flag_lookup, on=join_cols, how="left", suffixes=("", "_surf")
        )

        flags_col_name = "flags_surf" if "flags_surf" in panel_df.columns else "flags"
        flags_col = panel_df[flags_col_name].fillna(0).astype(int)

        # Initialize mask columns
        if "is_masked" not in panel_df.columns:
            panel_df["is_masked"] = False
        if "mask_reason" not in panel_df.columns:
            panel_df["mask_reason"] = None

        # Apply each flag
        for flag_value, reason in flags_to_mask:
            flagged = (flags_col & flag_value) != 0
            panel_df.loc[flagged, "is_masked"] = True
            # Only set reason if not already set (first flag wins)
            panel_df.loc[flagged & panel_df["mask_reason"].isna(), "mask_reason"] = reason

        # Clean up temporary column
        if "flags_surf" in panel_df.columns:
            panel_df.drop(columns=["flags_surf"], inplace=True)
        if "flags" in panel_df.columns and flags_col_name == "flags":
            panel_df.drop(columns=["flags"], inplace=True)

        n_masked = panel_df["is_masked"].sum()
        self._logger.info(
            f"Quality masking: {n_masked}/{len(panel_df)} rows masked "
            f"({n_masked / max(len(panel_df), 1) * 100:.1f}%)"
        )

        return panel_df

    def generate_node_features_only(
        self,
        surface_df: pd.DataFrame,
        feature_version: str = "v1.0",
    ) -> tuple[pd.DataFrame, FeatureResult]:
        """Generate only node features (utility method).

        Args:
            surface_df: Surface snapshot DataFrame
            feature_version: Version string for output

        Returns:
            Tuple of (features_df, FeatureResult)
        """
        return self._node_generator.generate(surface_df, feature_version)

    def generate_global_features_only(
        self,
        underlying_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate only global features (utility method).

        Args:
            underlying_df: Underlying bars DataFrame

        Returns:
            DataFrame with global features
        """
        return self._generate_global_features(underlying_df)

    def generate_macro_features_only(
        self,
        fred_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate only macro features (utility method).

        Args:
            fred_data: Dict mapping series_id to FRED DataFrame

        Returns:
            DataFrame with macro features
        """
        return self._macro_generator.generate_multi(fred_data)


def create_feature_engine(
    config: FeatureEngineConfig | None = None,
    engine: Engine | None = None,
) -> FeatureEngine:
    """Factory function for creating a FeatureEngine with default dependencies.

    This function creates all dependencies internally, which is the recommended
    approach for production use. For testing, inject dependencies directly
    into the FeatureEngine constructor.

    Args:
        config: Feature engine configuration. Uses defaults if None.
        engine: SQLAlchemy engine. Creates default if None.

    Returns:
        Configured FeatureEngine instance

    Example:
        engine = create_feature_engine(config=my_config)
        panel_df, result = engine.build_feature_panel(
            start=start_dt,
            end=end_dt,
            surface_version="v1.0",
            feature_version="v1.0",
        )
    """
    return FeatureEngine(config=config, engine=engine)
