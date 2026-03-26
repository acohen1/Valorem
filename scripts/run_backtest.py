#!/usr/bin/env python3
"""Run a backtest over historical data.

Usage:
    python scripts/run_backtest.py --start-date 2024-01-01 --end-date 2024-01-31
    python scripts/run_backtest.py --config config/config.yaml --dry-run
    python scripts/run_backtest.py --mock  # Use synthetic data for testing
"""

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.backtest import BacktestEngine, BacktestResult
from src.config.constants import SurfaceConstants
from src.config.loader import ConfigLoader
from src.config.logging import add_file_handler, setup_logging
from src.config.schema import (
    ExecutionConfig,
    PositionManagementConfig,
    RiskConfig,
)
from src.strategy.types import Signal, SignalType


def generate_mock_surface(
    as_of_date: date,
    underlying_price: float = 450.0,
    base_iv: float = 0.20,
) -> pd.DataFrame:
    """Generate synthetic options surface for testing.

    Args:
        as_of_date: Surface date
        underlying_price: Underlying price
        base_iv: Base implied volatility

    Returns:
        DataFrame with surface columns
    """
    tenors = list(SurfaceConstants.TENOR_DAYS_DEFAULT)
    strike_pcts = [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]

    records = []
    for tenor in tenors:
        expiry = as_of_date + pd.Timedelta(days=tenor)

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

                # Synthetic price using delta proxy
                time_value = iv * np.sqrt(tenor / 365) * underlying_price * 0.4
                intrinsic = max(0, (underlying_price - strike) if right == "C" else (strike - underlying_price))
                mid_price = intrinsic + time_value * abs_delta

                bid = max(0.01, mid_price * 0.98)
                ask = mid_price * 1.02

                # Generate OSI symbol
                expiry_str = expiry.strftime("%y%m%d")
                strike_str = f"{int(strike * 1000):08d}"
                symbol = f"SPY{expiry_str}{right}{strike_str}"

                records.append({
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
                })

    return pd.DataFrame(records)


def generate_mock_signals(surface: pd.DataFrame, period: int) -> list[Signal]:
    """Generate synthetic signals for testing.

    Args:
        surface: Current surface
        period: Period number (for randomization)

    Returns:
        List of signals (0-2 per period)
    """
    np.random.seed(period)

    if np.random.random() > 0.3:  # 70% chance of signal
        return []

    # Pick a random node
    node = surface.sample(1).iloc[0]

    signal_types = [SignalType.TERM_ANOMALY, SignalType.DIRECTIONAL_VOL]
    signal_type = signal_types[np.random.choice([0, 1], p=[0.4, 0.6])]

    return [
        Signal(
            signal_type=signal_type,
            edge=np.random.uniform(0.02, 0.08),
            confidence=np.random.uniform(0.6, 0.9),
            tenor_days=int(node["tenor_days"]),
            delta_bucket=str(node["delta_bucket"]),
            timestamp=datetime.now(timezone.utc),
        )
    ]


def run_mock_backtest(
    start_date: date,
    end_date: date,
    initial_capital: float,
    execution_config: ExecutionConfig,
    risk_config: RiskConfig,
    pos_mgmt_config: PositionManagementConfig,
) -> BacktestResult:
    """Run backtest with mock/synthetic data.

    Args:
        start_date: Backtest start
        end_date: Backtest end
        initial_capital: Starting capital
        execution_config: Execution config
        risk_config: Risk config
        pos_mgmt_config: Position management config

    Returns:
        BacktestResult
    """
    # Generate surfaces for each trading day
    surfaces = {}
    signals_by_date = {}

    current = start_date
    period = 0
    underlying_price = 450.0

    while current <= end_date:
        # Skip weekends
        if current.weekday() >= 5:
            current += pd.Timedelta(days=1)
            continue

        # Random walk for underlying
        underlying_price *= 1 + np.random.normal(0, 0.01)

        surface = generate_mock_surface(current, underlying_price)
        surfaces[current] = surface

        signals = generate_mock_signals(surface, period)
        if signals:
            signals_by_date[current] = signals

        current += pd.Timedelta(days=1)
        period += 1

    logger.info(f"Generated {len(surfaces)} trading days of mock data")
    logger.info(f"Generated signals for {len(signals_by_date)} days")

    # Create and run backtest engine
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        execution_config=execution_config,
        risk_config=risk_config,
        position_management_config=pos_mgmt_config,
    )

    result = engine.run_backtest(
        surfaces=surfaces,
        signals_by_date=signals_by_date,
    )

    return result


def print_summary(result: BacktestResult) -> None:
    """Print backtest summary to console."""
    print(result.summary())

    # Additional details
    m = result.metrics
    print("\nRisk Metrics")
    print("-" * 40)
    print(f"95% VaR: ${m.var_95:,.2f}")
    print(f"95% CVaR: ${m.cvar_95:,.2f}")
    print(f"Max Concurrent Positions: {m.max_concurrent_positions}")
    print(f"Avg Concurrent Positions: {m.avg_concurrent_positions:.1f}")


def export_results(result: BacktestResult, output_dir: Path) -> None:
    """Export backtest results to files.

    Args:
        result: Backtest result
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export trades
    trades_path = output_dir / f"{result.run_id}_trades.csv"
    result.export_trades_csv(str(trades_path))
    logger.info(f"Exported trades to {trades_path}")

    # Export portfolio history
    portfolio_path = output_dir / f"{result.run_id}_portfolio.csv"
    result.export_portfolio_csv(str(portfolio_path))
    logger.info(f"Exported portfolio history to {portfolio_path}")

    # Export summary
    summary_path = output_dir / f"{result.run_id}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(result.summary())
    logger.info(f"Exported summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--start-date",
        type=lambda s: date.fromisoformat(s),
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: date.fromisoformat(s),
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=None,
        help="Initial capital in USD (default: from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        help="Environment overlay name (loads config/environments/{env}.yaml)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock/synthetic data for testing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running backtest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/reports",
        help="Directory for output files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best_model.pt",
        help="Path to trained model checkpoint (real data mode)",
    )
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v1.0",
        help="Feature version in node_panel table (real data mode)",
    )
    parser.add_argument(
        "--surface-version",
        type=str,
        default="v1.0",
        help="Surface snapshot version (real data mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model inference: auto, cpu, cuda, mps (default: from config)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # Load configuration
    try:
        config = ConfigLoader.load(Path(args.config), env=args.env)
        logger.info(f"Loaded configuration from {args.config} (env={args.env})")
        add_file_handler(
            workflow="backtest",
            logs_dir=config.paths.logs_dir,
            level=config.logging.level,
            fmt=config.logging.format,
            enabled=config.logging.file_enabled,
        )
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Use defaults if config fails
        config = None

    # Get dates from args or config
    if args.start_date:
        start_date = args.start_date
    elif config:
        start_date = config.backtest.start_date
    else:
        start_date = date(2024, 1, 1)

    if args.end_date:
        end_date = args.end_date
    elif config:
        end_date = config.backtest.end_date
    else:
        end_date = date(2024, 1, 31)

    if args.initial_capital is not None:
        initial_capital = args.initial_capital
    elif config:
        initial_capital = config.backtest.initial_capital
    else:
        initial_capital = 100000.0

    # Get configs
    if config:
        execution_config = config.execution
        risk_config = config.risk
        pos_mgmt_config = config.risk.position_management
    else:
        # Use permissive defaults for mock testing
        from src.config.schema import PerTradeRiskConfig, RiskCapsConfig

        execution_config = ExecutionConfig()
        risk_config = RiskConfig(
            per_trade=PerTradeRiskConfig(max_loss=2000.0, max_contracts=20),
            caps=RiskCapsConfig(max_abs_delta=500.0, max_abs_vega=5000.0),
        )
        pos_mgmt_config = PositionManagementConfig()

    logger.info(f"Backtest period: {start_date} to {end_date}")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")

    if args.dry_run:
        logger.info("Dry run complete - configuration is valid")
        return

    if args.mock:
        logger.info("Running with mock data...")
        result = run_mock_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            execution_config=execution_config,
            risk_config=risk_config,
            pos_mgmt_config=pos_mgmt_config,
        )
    else:
        # Real data backtest
        from src.backtest.data_pipeline import BacktestDataConfig, BacktestDataPipeline
        from src.data.storage.engine import create_engine as create_db_engine
        from src.data.storage.repository import DerivedRepository, RawRepository

        if not config:
            logger.error("Real data backtest requires a valid config file.")
            return

        db_engine = create_db_engine(config.paths.db_path)
        raw_repo = RawRepository(db_engine.engine)
        derived_repo = DerivedRepository(db_engine.engine)

        device = args.device if args.device is not None else config.training.device

        pipeline_config = BacktestDataConfig(
            checkpoint_path=args.checkpoint,
            feature_version=args.feature_version,
            surface_version=args.surface_version,
            underlying_symbol=config.universe.underlying,
            device=device,
        )

        logger.info("Loading real data and running model inference...")
        logger.info(f"  Checkpoint: {args.checkpoint}")
        logger.info(f"  Feature version: {args.feature_version}")
        logger.info(f"  Surface version: {args.surface_version}")
        logger.info(f"  Device: {device}")

        pipeline = BacktestDataPipeline(raw_repo, derived_repo, pipeline_config)
        backtest_data = pipeline.load(start_date, end_date)

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            pricer=backtest_data.pricer,
        )

        result = engine.run_backtest(
            surfaces=backtest_data.surfaces,
            signals_by_date=backtest_data.signals_by_date,
        )

    # Print summary
    print_summary(result)

    # Export results
    output_dir = Path(args.output_dir)
    export_results(result, output_dir)


if __name__ == "__main__":
    main()
