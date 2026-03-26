#!/usr/bin/env python3
"""Run paper trading loop.

Usage:
    python scripts/run_paper_trading.py --config config/config.yaml
    python scripts/run_paper_trading.py --mock --max-iterations 10
    python scripts/run_paper_trading.py --dry-run  # Validate config only

The trading loop runs continuously until stopped via Ctrl+C or kill switch.
"""

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.config.constants import SurfaceConstants
from src.config.loader import ConfigLoader
from src.config.env import (
    EnvironmentConfig,
    TradingMode,
    print_validation_results,
    validate_cli_config,
)
from src.config.schema import (
    ExecutionConfig,
    PaperConfig,
    PerTradeRiskConfig,
    RiskCapsConfig,
    RiskConfig,
)
from src.live import (
    DatabaseSurfaceProvider,
    DatabentoIngestionService,
    DatabentoSymbolProvider,
    Fill,
    LoopState,
    MockIngestionService,
    ModelSignalGenerator,
    PaperOrderRouter,
    RuleBasedSignalGenerator,
    save_symbols_manifest,
    SignalGenerator,
    SurfaceProvider,
    TradingLoop,
)
from loguru import logger

from src.config.logging import add_file_handler, setup_logging
from src.data.storage.engine import create_engine
from src.data.storage.repository import DerivedRepository, RawRepository
from src.strategy.types import Signal, SignalType


class MockSurfaceProvider(SurfaceProvider):
    """Mock surface provider for testing - generates synthetic surfaces."""

    def __init__(
        self,
        underlying_price: float = 450.0,
        base_iv: float = 0.20,
        random_walk: bool = True,
    ) -> None:
        """Initialize mock surface provider.

        Args:
            underlying_price: Starting underlying price
            base_iv: Base implied volatility
            random_walk: Whether to apply random walk to underlying
        """
        self._underlying_price = underlying_price
        self._base_iv = base_iv
        self._random_walk = random_walk
        self._call_count = 0

    def get_latest_surface(self) -> pd.DataFrame:
        """Generate synthetic options surface."""
        self._call_count += 1

        # Apply random walk to underlying
        if self._random_walk:
            self._underlying_price *= 1 + np.random.normal(0, 0.002)

        return self._generate_surface(self._underlying_price, self._base_iv)

    def _generate_surface(
        self, underlying_price: float, base_iv: float
    ) -> pd.DataFrame:
        """Generate synthetic options surface."""
        tenors = list(SurfaceConstants.TENOR_DAYS_DEFAULT)
        strike_pcts = [0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10]
        as_of_date = date.today()

        records = []
        for tenor in tenors:
            expiry = as_of_date + timedelta(days=tenor)

            for strike_pct in strike_pcts:
                strike = round(underlying_price * strike_pct, 2)

                for right in ["C", "P"]:
                    # Compute synthetic delta
                    moneyness = underlying_price / strike
                    if right == "C":
                        delta = max(0.05, min(0.95, 0.5 + 0.4 * (moneyness - 1)))
                    else:
                        delta = -max(0.05, min(0.95, 0.5 - 0.4 * (moneyness - 1)))

                    # Assign delta bucket
                    abs_delta = abs(delta)
                    if abs_delta >= 0.45:
                        delta_bucket = "ATM"
                    elif abs_delta >= 0.35:
                        delta_bucket = "C40" if right == "C" else "P40"
                    elif abs_delta >= 0.15:
                        delta_bucket = "C25" if right == "C" else "P25"
                    else:
                        delta_bucket = "C10" if right == "C" else "P10"

                    # IV with skew
                    iv = base_iv * (1 + 0.1 * (1 - moneyness))

                    # Synthetic price
                    time_value = (
                        iv * np.sqrt(tenor / 365) * underlying_price * 0.4
                    )
                    intrinsic = max(
                        0,
                        (underlying_price - strike)
                        if right == "C"
                        else (strike - underlying_price),
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
                            "tenor_days": tenor,
                            "delta_bucket": delta_bucket,
                            "strike": strike,
                            "expiry": expiry,
                            "right": right,
                            "bid": round(bid, 2),
                            "ask": round(ask, 2),
                            "mid_price": round(mid_price, 2),
                            "delta": round(delta, 4),
                            "gamma": round(0.02 / (1 + abs(moneyness - 1) * 10), 4),
                            "vega": round(0.3 * np.sqrt(tenor / 30), 4),
                            "theta": round(-0.02 * (30 / tenor), 4),
                            "iv": round(iv, 4),
                            "underlying_price": underlying_price,
                        }
                    )

        return pd.DataFrame(records)


class MockSignalGenerator(SignalGenerator):
    """Mock signal generator for testing - generates random signals."""

    def __init__(
        self,
        signal_probability: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize mock signal generator.

        Args:
            signal_probability: Probability of generating a signal each call
            seed: Random seed for reproducibility
        """
        self._signal_probability = signal_probability
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """Generate random signals."""
        self._call_count += 1

        if self._rng.random() > self._signal_probability:
            return []

        # Pick random node from surface
        valid_nodes = surface[
            surface["delta_bucket"].isin(["P25", "ATM", "C25"])
        ]
        if valid_nodes.empty:
            return []

        node = valid_nodes.sample(1, random_state=self._call_count).iloc[0]

        # Random signal type
        signal_types = [
            SignalType.TERM_ANOMALY,
            SignalType.DIRECTIONAL_VOL,
            SignalType.SKEW_ANOMALY,
            SignalType.ELEVATED_IV,
        ]
        signal_type = signal_types[self._rng.integers(0, len(signal_types))]

        return [
            Signal(
                signal_type=signal_type,
                edge=float(self._rng.uniform(0.02, 0.08)),
                confidence=float(self._rng.uniform(0.6, 0.9)),
                tenor_days=int(node["tenor_days"]),
                delta_bucket=str(node["delta_bucket"]),
                timestamp=datetime.now(timezone.utc),
            )
        ]


def on_fill(fill: Fill) -> None:
    """Callback when an order is filled."""
    logger.info(
        f"[FILL] order={fill.order_id}, "
        f"gross={fill.gross_premium:.2f}, "
        f"fees={fill.fees:.2f}, "
        f"net={fill.net_premium:.2f}"
    )


def on_iteration(state: LoopState) -> None:
    """Callback after each iteration."""
    logger.info(
        f"[ITERATION {state.iteration}] "
        f"signals={state.last_signal_count}, "
        f"orders={state.last_order_count}, "
        f"fills={state.last_fill_count}"
    )


def run_paper_trading(
    paper_config: PaperConfig,
    execution_config: ExecutionConfig,
    risk_config: RiskConfig,
    surface_config=None,
    universe_config=None,
    use_mock: bool = False,
    model_checkpoint: Optional[str] = None,
    option_symbols: Optional[list[str]] = None,
    db_path: Optional[str] = None,
) -> None:
    """Run the paper trading loop.

    Args:
        paper_config: Paper trading configuration
        execution_config: Execution configuration
        risk_config: Risk configuration
        surface_config: Surface configuration (required for real mode)
        universe_config: Universe configuration (required for real mode)
        use_mock: Whether to use mock data providers
        model_checkpoint: Path to trained model checkpoint (for real mode)
        option_symbols: List of option symbols to subscribe to (for real mode)
        db_path: Path to database file (required for real mode)
    """
    surface_provider: SurfaceProvider
    signal_generator: SignalGenerator

    ingestion_service = None

    if use_mock:
        logger.info("Using mock data providers (in-memory, no DB)")
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.4)
    else:
        # Validate required configs for real mode
        if surface_config is None or universe_config is None:
            logger.error(
                "Surface and universe configuration required for real data mode. "
                "Ensure config.yaml has surface and universe sections."
            )
            return

        if not option_symbols:
            logger.error(
                "Option symbols required for real data mode. "
                "Use --symbols-file to provide a manifest."
            )
            return

        logger.info("Initializing unified data pipeline...")

        # Create database engine and repositories
        try:
            db_engine = create_engine(db_path)
            raw_repo = RawRepository(db_engine.engine)
            derived_repo = DerivedRepository(db_engine.engine)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return

        # Create ingestion service (streams data to DB)
        try:
            ingestion_service = DatabentoIngestionService(
                raw_repo=raw_repo,
                derived_repo=derived_repo,
                surface_config=surface_config,
                universe_config=universe_config,
                option_symbols=option_symbols,
            )
            ingestion_service.start()
            logger.info(f"DatabentoIngestionService started with {len(option_symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to start ingestion service: {e}")
            return

        # Create surface provider (reads from DB)
        surface_provider = DatabaseSurfaceProvider(
            derived_repo=derived_repo,
            version="live",
        )

        # Create signal generator
        if model_checkpoint:
            try:
                signal_generator = ModelSignalGenerator.from_checkpoint(
                    checkpoint_path=model_checkpoint,
                )
                logger.info(f"Loaded model from {model_checkpoint}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                ingestion_service.stop()
                return
        else:
            logger.warning("No model checkpoint provided, using rule-based signals")
            signal_generator = RuleBasedSignalGenerator()

    # Create order router
    order_router = PaperOrderRouter(execution_config)

    # Create and configure trading loop
    loop = TradingLoop(
        paper_config=paper_config,
        execution_config=execution_config,
        risk_config=risk_config,
        surface_provider=surface_provider,
        signal_generator=signal_generator,
        order_router=order_router,
        on_fill_callback=on_fill,
        on_iteration_callback=on_iteration,
    )

    logger.info("=" * 60)
    logger.info("Starting paper trading loop")
    logger.info(f"  Loop interval: {paper_config.loop_interval_seconds}s")
    logger.info(f"  Max iterations: {paper_config.max_loop_iterations or 'unlimited'}")
    logger.info(f"  Halt on error: {paper_config.halt_on_error}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        loop.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        # Stop ingestion service if running
        if ingestion_service is not None:
            ingestion_service.stop()

        # Print final metrics
        metrics = loop.metrics
        print("\n" + "=" * 60)
        print("PAPER TRADING SUMMARY")
        print("=" * 60)
        print(f"Total iterations: {metrics.total_iterations}")
        print(f"Total signals: {metrics.total_signals}")
        print(f"Total orders: {metrics.total_orders}")
        print(f"Total fills: {metrics.total_fills}")
        print(f"Total rejections: {metrics.total_rejections}")
        print(f"Daily P&L: ${metrics.daily_pnl:,.2f}")

        if metrics.start_time and metrics.end_time:
            duration = metrics.end_time - metrics.start_time
            print(f"Duration: {duration}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run paper trading loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock mode (synthetic data):
  python scripts/run_paper_trading.py --mode mock --max-iterations 10

  # Paper trading with live data:
  python scripts/run_paper_trading.py --mode paper_live \\
      --symbols-file data/manifests/spy_options.json

  # Discover symbols and save manifest:
  python scripts/run_paper_trading.py --discover-symbols --underlying SPY \\
      --output data/manifests/spy_options.json

  # Validate configuration:
  python scripts/run_paper_trading.py --validate --mode paper_live

  # Live trading (real execution - use with caution!):
  python scripts/run_paper_trading.py --mode live \\
      --symbols-file data/manifests/spy_options.json
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mock", "paper_live", "paper_db", "live"],
        default="mock",
        help="Trading mode: mock (synthetic), paper_live (live data, simulated exec), "
             "paper_db (database replay), live (real execution)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="Path to JSON manifest with option symbols",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Maximum loop iterations (0 = unlimited)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Seconds between loop iterations",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit",
    )
    parser.add_argument(
        "--discover-symbols",
        action="store_true",
        help="Discover option symbols from Databento and save to manifest",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying symbol for --discover-symbols (default: SPY)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for --discover-symbols manifest",
    )
    parser.add_argument(
        "--min-dte",
        type=int,
        default=7,
        help="Minimum DTE for --discover-symbols (default: 7)",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=90,
        help="Maximum DTE for --discover-symbols (default: 90)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # Handle --discover-symbols mode
    if args.discover_symbols:
        _run_symbol_discovery(args)
        return

    # Handle --validate mode
    if args.validate:
        results = validate_cli_config(
            mode=args.mode,
            symbols_file=args.symbols_file,
            checkpoint=args.checkpoint,
        )
        print_validation_results(results)
        sys.exit(0 if results["valid"] else 1)

    # Load environment config
    env_config = EnvironmentConfig.from_env(mode=args.mode)

    # Validate environment config
    try:
        env_config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # Load YAML configuration
    try:
        config = ConfigLoader.load(Path(args.config))
        logger.info(f"Loaded configuration from {args.config}")
        add_file_handler(
            workflow="live",
            logs_dir=config.paths.logs_dir,
            level=config.logging.level,
            fmt=config.logging.format,
            enabled=config.logging.file_enabled,
        )
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        config = None

    # Build configs
    if config:
        paper_config = config.paper
        execution_config = config.execution
        risk_config = config.risk
    else:
        # Use permissive defaults for mock testing
        paper_config = PaperConfig()
        execution_config = ExecutionConfig()
        risk_config = RiskConfig(
            per_trade=PerTradeRiskConfig(max_loss=2000.0, max_contracts=20),
            caps=RiskCapsConfig(max_abs_delta=500.0, max_abs_vega=5000.0),
        )

    # Apply command-line overrides
    paper_config = PaperConfig(
        **{
            **paper_config.model_dump(),
            "max_loop_iterations": args.max_iterations or paper_config.max_loop_iterations,
            "loop_interval_seconds": args.interval,
        }
    )

    logger.info("Trading configuration:")
    logger.info(f"  Mode: {env_config.mode.value}")
    logger.info(f"  Loop interval: {paper_config.loop_interval_seconds}s")
    logger.info(f"  Max iterations: {paper_config.max_loop_iterations or 'unlimited'}")

    # Warn about LIVE mode
    if env_config.mode == TradingMode.LIVE:
        logger.warning("=" * 60)
        logger.warning("⚠️  LIVE MODE - REAL MONEY EXECUTION ⚠️")
        logger.warning("Orders will be sent to broker for real execution!")
        logger.warning("=" * 60)

    # Load option symbols for non-mock modes
    option_symbols = None
    if args.symbols_file:
        import json
        try:
            with open(args.symbols_file) as f:
                manifest = json.load(f)
                option_symbols = manifest.get("symbols", manifest)
                if isinstance(option_symbols, dict):
                    option_symbols = manifest.get("symbols", [])
            logger.info(f"Loaded {len(option_symbols)} symbols from {args.symbols_file}")
        except Exception as e:
            logger.error(f"Failed to load symbols file: {e}")
            return

    # Get surface and universe config for real mode
    surface_config = getattr(config, "surface", None) if config else None
    universe_config = getattr(config, "universe", None) if config else None

    # Determine if using mock based on mode
    use_mock = env_config.mode == TradingMode.MOCK

    run_paper_trading(
        paper_config=paper_config,
        execution_config=execution_config,
        risk_config=risk_config,
        surface_config=surface_config,
        universe_config=universe_config,
        use_mock=use_mock,
        model_checkpoint=args.checkpoint,
        option_symbols=option_symbols,
        db_path=config.paths.db_path if config else "data/db.sqlite",
    )


def _run_symbol_discovery(args) -> None:
    """Run symbol discovery mode.

    Fetches option symbols from Databento and saves to manifest file.
    """
    if not args.output:
        logger.error("--output required for --discover-symbols")
        sys.exit(1)

    logger.info(f"Discovering option symbols for {args.underlying}...")
    logger.info(f"  DTE range: {args.min_dte} - {args.max_dte}")

    try:
        provider = DatabentoSymbolProvider()
        symbols = provider.get_option_symbols(
            underlying=args.underlying,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
        )

        if not symbols:
            logger.warning("No symbols found matching criteria")
            sys.exit(1)

        # Save manifest
        save_symbols_manifest(
            symbols=symbols,
            output_path=args.output,
            underlying=args.underlying,
            metadata={
                "min_dte": args.min_dte,
                "max_dte": args.max_dte,
            },
        )

        logger.info(f"Saved {len(symbols)} symbols to {args.output}")

    except Exception as e:
        logger.error(f"Symbol discovery failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
