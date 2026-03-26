#!/usr/bin/env python
"""CLI script for building feature panel.

This script orchestrates feature generation from surface snapshots:
- Node features: IV changes, microstructure, cross-sectional surface features
- Global features: Returns, realized volatility, drawdown
- Macro features: FRED series transforms with release-time alignment

Usage:
    # Build features for a date range
    python scripts/build_features.py --start-date 2024-01-01 --end-date 2024-01-31 --surface-version v1.0 --feature-version v1.0

    # Skip writing to database (dry run)
    python scripts/build_features.py --start-date 2024-01-01 --end-date 2024-01-31 --surface-version v1.0 --feature-version v1.0 --dry-run

    # Only generate node features
    python scripts/build_features.py --start-date 2024-01-01 --end-date 2024-01-31 --surface-version v1.0 --feature-version v1.0 --node-only

    # Skip validation
    python scripts/build_features.py --start-date 2024-01-01 --end-date 2024-01-31 --surface-version v1.0 --feature-version v1.0 --skip-validation
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.config.loader import ConfigLoader
from src.config.logging import add_file_handler, setup_logging
from src.data.storage.engine import create_engine
from src.data.storage.repository import DerivedRepository, RawRepository
from src.features.engine import FeatureEngine, FeatureEngineConfig
from src.surface.builder import SurfaceBuilder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build feature panel from surface snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--surface-version",
        type=str,
        required=True,
        help="Version of surface snapshots to use (e.g., v1.0)",
    )

    parser.add_argument(
        "--feature-version",
        type=str,
        required=True,
        help="Version string for output features (e.g., v1.0)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate features but don't write to database",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip anti-leakage validation",
    )

    parser.add_argument(
        "--fail-on-validation-issues",
        action="store_true",
        help=(
            "Fail the build if feature validation reports any issues "
            "(warnings or errors)"
        ),
    )

    parser.add_argument(
        "--node-only",
        action="store_true",
        help="Only generate node features (skip global and macro)",
    )

    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Only generate global features (skip node and macro)",
    )

    parser.add_argument(
        "--macro-only",
        action="store_true",
        help="Only generate macro features (skip node and global)",
    )

    parser.add_argument(
        "--underlying",
        type=str,
        default=None,
        help="Underlying symbol (default: from config)",
    )

    parser.add_argument(
        "--fred-series",
        type=str,
        nargs="+",
        default=None,
        help="FRED series to include (default: from config)",
    )

    parser.add_argument(
        "--lookback-buffer",
        type=int,
        default=None,
        help="Days of lookback buffer for rolling calculations (default: from config)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        help="Environment overlay name (loads config/environments/{env}.yaml)",
    )

    parser.add_argument(
        "--skip-surfaces",
        action="store_true",
        help="Skip surface building (assumes surfaces already exist in DB)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save features as CSV",
    )

    return parser.parse_args()



def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        datetime object at midnight

    Raises:
        ValueError: If date string is invalid
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD.") from e


def main() -> int:
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    try:
        # Parse dates
        start = parse_date(args.start_date)
        end = parse_date(args.end_date)

        if start >= end:
            logger.error("Start date must be before end date")
            return 1

        if args.skip_validation and args.fail_on_validation_issues:
            logger.error(
                "--fail-on-validation-issues cannot be used with --skip-validation"
            )
            return 1

        # Load configuration
        logger.info(f"Loading config from {args.config} (env={args.env})")
        config = ConfigLoader.load(Path(args.config), env=args.env)
        add_file_handler(
            workflow="surface",
            logs_dir=config.paths.logs_dir,
            level=config.logging.level,
            fmt=config.logging.format,
            enabled=config.logging.file_enabled,
        )

        # Create database engine and repositories
        db_engine = create_engine(config.paths.db_path)
        sa_engine = db_engine.engine  # Raw SQLAlchemy Engine for repositories
        raw_repo = RawRepository(sa_engine)
        derived_repo = DerivedRepository(sa_engine)

        # Step 1: Build surfaces (unless --skip-surfaces)
        if not args.skip_surfaces:
            logger.info("Building surface snapshots...")
            surface_builder = SurfaceBuilder(
                config=config.surface,
                universe=config.universe,
                raw_repo=raw_repo,
                derived_repo=derived_repo,
            )

            # Process in monthly chunks for progress and memory
            chunk_starts = pd.date_range(start, end, freq="MS").tolist()
            if not chunk_starts or chunk_starts[0] > pd.Timestamp(start):
                chunk_starts.insert(0, pd.Timestamp(start))
            chunk_ends = chunk_starts[1:] + [pd.Timestamp(end)]

            total_snapshots = 0
            total_iv_failures = 0
            total_quotes_processed = 0
            for i, (cs, ce) in enumerate(zip(chunk_starts, chunk_ends)):
                cs_dt = cs.to_pydatetime()
                ce_dt = ce.to_pydatetime()
                logger.info(
                    f"Building surfaces chunk {i+1}/{len(chunk_starts)}: "
                    f"{cs_dt.date()} to {ce_dt.date()}"
                )
                build_result = surface_builder.build_surface(
                    start=cs_dt,
                    end=ce_dt,
                    version=args.surface_version,
                )
                total_snapshots += build_result.row_count
                total_iv_failures += build_result.iv_failures
                total_quotes_processed += build_result.quotes_processed
                logger.info(
                    f"  Chunk {i+1}: {build_result.row_count} snapshots, "
                    f"{build_result.iv_failures} IV failures "
                    f"(of {build_result.quotes_processed} quotes)"
                )

            iv_failure_ratio = (
                total_iv_failures / total_quotes_processed
                if total_quotes_processed > 0 else 0.0
            )
            print(
                f"\nSurface build complete: {total_snapshots} snapshots, "
                f"{total_iv_failures} IV failures "
                f"({iv_failure_ratio:.1%} of {total_quotes_processed} quotes)"
            )

            max_ratio = config.surface.max_iv_failure_ratio
            if iv_failure_ratio > max_ratio:
                logger.warning(
                    f"IV failure ratio {iv_failure_ratio:.1%} exceeds threshold "
                    f"{max_ratio:.0%} (config: surface.max_iv_failure_ratio)"
                )
                print(
                    f"\n*** WARNING: IV failure ratio ({iv_failure_ratio:.1%}) "
                    f"exceeds configured threshold ({max_ratio:.0%}) ***"
                )
        else:
            logger.info("Skipping surface build (--skip-surfaces)")

        # Resolve parameters: CLI arg → config → schema default
        underlying = args.underlying if args.underlying is not None else config.universe.underlying
        fred_series = args.fred_series if args.fred_series is not None else list(config.features.macro.series)
        lookback_buffer = args.lookback_buffer if args.lookback_buffer is not None else config.features.node.lookback_bars

        # Step 2: Configure feature engine
        engine_config = FeatureEngineConfig(
            underlying_symbol=underlying,
            fred_series=fred_series,
            lookback_buffer_days=lookback_buffer,
            include_node_features=not (args.global_only or args.macro_only),
            include_global_features=not (args.node_only or args.macro_only),
            include_macro_features=not (args.node_only or args.global_only),
            fail_on_validation_issues=args.fail_on_validation_issues,
        )

        # Create feature engine
        feature_engine = FeatureEngine(
            config=engine_config,
            engine=sa_engine,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        # Step 3: Build features (chunked monthly to bound memory)
        logger.info(
            f"Building features: {start.date()} to {end.date()}, "
            f"surface={args.surface_version}, features={args.feature_version}"
        )

        # Monthly chunks for feature generation
        feat_chunk_starts = pd.date_range(start, end, freq="MS").tolist()
        if not feat_chunk_starts or feat_chunk_starts[0] > pd.Timestamp(start):
            feat_chunk_starts.insert(0, pd.Timestamp(start))
        feat_chunk_ends = feat_chunk_starts[1:] + [pd.Timestamp(end)]

        total_rows = 0
        total_feature_count = 0
        total_node_features = 0
        total_global_features = 0
        total_macro_features = 0
        total_nodes = 0
        all_validation_passed = True
        csv_header_written = False

        for i, (fcs, fce) in enumerate(zip(feat_chunk_starts, feat_chunk_ends)):
            fcs_dt = fcs.to_pydatetime()
            fce_dt = fce.to_pydatetime()
            logger.info(
                f"Building features chunk {i+1}/{len(feat_chunk_starts)}: "
                f"{fcs_dt.date()} to {fce_dt.date()}"
            )

            panel_df, result = feature_engine.build_feature_panel(
                start=fcs_dt,
                end=fce_dt,
                surface_version=args.surface_version,
                feature_version=args.feature_version,
                write_to_db=not args.dry_run,
            )

            # Append to CSV per chunk if requested
            if args.output_csv and not panel_df.empty:
                output_path = Path(args.output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                panel_df.to_csv(
                    output_path,
                    mode="a" if csv_header_written else "w",
                    header=not csv_header_written,
                    index=False,
                )
                csv_header_written = True

            # Accumulate totals
            total_rows += result.row_count
            total_feature_count = max(total_feature_count, result.feature_count)
            total_node_features = max(total_node_features, result.node_features_count)
            total_global_features = max(total_global_features, result.global_features_count)
            total_macro_features = max(total_macro_features, result.macro_features_count)
            total_nodes = max(total_nodes, result.nodes_processed)
            if not result.validation_passed:
                all_validation_passed = False

            logger.info(
                f"  Chunk {i+1}: {result.row_count} rows, "
                f"{result.feature_count} features"
            )

            del panel_df

        # Print result summary
        print("\n" + "=" * 60)
        print("FEATURE BUILD RESULT")
        print("=" * 60)
        print(f"Feature version: {args.feature_version}")
        print(f"Surface version: {args.surface_version}")
        print(f"Date range: {start.date()} to {end.date()}")
        print(f"\nTotal rows: {total_rows}")
        print(f"Total features: {total_feature_count}")
        print(f"  - Node features: {total_node_features}")
        print(f"  - Global features: {total_global_features}")
        print(f"  - Macro features: {total_macro_features}")
        print(f"Unique nodes: {total_nodes}")
        print(f"\nValidation: {'PASSED' if all_validation_passed else 'FAILED'}")

        if args.dry_run:
            print("\n[DRY RUN] Features not written to database.")
        else:
            print(f"\nFeatures written to node_panel table (version={args.feature_version})")

        if args.output_csv:
            print(f"Features saved to: {args.output_csv}")

        return 0 if all_validation_passed else 1

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nERROR: {e}")
        return 1

    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        print(f"\nERROR: {e}")
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nERROR: Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
