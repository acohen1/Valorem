#!/usr/bin/env python
"""CLI script for raw data ingestion.

This script coordinates ingestion of market data, option quotes, and macro
data from configured providers into the database.

Usage:
    # Preview mode (estimate costs only)
    python scripts/ingest_raw.py --preview-only --start-date 2024-01-01 --end-date 2024-01-02

    # Full ingestion (interactive cost confirmation)
    python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-01-02

    # Full ingestion (skip confirmation for automation)
    python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-01-02 --yes

    # Skip validation
    python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-01-02 --skip-validation

    # Use existing manifest
    python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-01-02 --manifest data/manifest/manifest_SPY_2024-01-01.json
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging import add_file_handler, setup_logging

from dotenv import load_dotenv

load_dotenv()

from src.config.loader import ConfigLoader
from src.data.ingest.orchestrator import (
    CostExceededException,
    DataQualityException,
    create_orchestrator,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest raw market data from providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides config value.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Overrides config value.",
    )

    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only estimate costs without fetching data",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data quality validation",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to existing manifest file (optional)",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock providers (for testing)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all data even if already ingested (default: skip existing)",
    )

    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts (for automated/non-interactive use)",
    )

    parser.add_argument(
        "--fred-only",
        action="store_true",
        help="Only fetch FRED macro series (skip Databento market data entirely)",
    )

    parser.add_argument(
        "--quotes-only",
        action="store_true",
        help="Only fetch option quotes (skip underlying bars, option bars, statistics, and FRED)",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only fetch option statistics/OI (skip underlying bars, option bars, quotes, and FRED)",
    )

    args = parser.parse_args()

    only_flags = [args.fred_only, args.quotes_only, args.stats_only]
    if sum(only_flags) > 1:
        parser.error("--fred-only, --quotes-only, and --stats-only are mutually exclusive")

    return args



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
        # Load configuration first (dates may come from config)
        logger.info(f"Loading config from {args.config} (env={args.env})")
        config = ConfigLoader.load(Path(args.config), env=args.env)
        add_file_handler(
            workflow="ingest",
            logs_dir=config.paths.logs_dir,
            level=config.logging.level,
            fmt=config.logging.format,
            enabled=config.logging.file_enabled,
        )

        # Resolve dates: CLI args override config values
        start_str = args.start_date
        end_str = args.end_date

        if start_str is None and config.data.ingestion.start_date is not None:
            start_str = config.data.ingestion.start_date.isoformat()
            logger.info(f"Using ingestion start_date from config: {start_str}")

        if end_str is None and config.data.ingestion.end_date is not None:
            end_str = config.data.ingestion.end_date.isoformat()
            logger.info(f"Using ingestion end_date from config: {end_str}")

        if start_str is None or end_str is None:
            logger.error(
                "start_date and end_date are required. "
                "Set them in config (data.ingestion.start_date/end_date) "
                "or pass --start-date/--end-date on the CLI."
            )
            return 1

        start = parse_date(start_str)
        end = parse_date(end_str)

        if start >= end:
            logger.error("Start date must be before end date")
            return 1

        # Preview mode confirmation prompt
        if args.preview_only and not args.fred_only:
            # Check if manifests already exist on disk for this date range
            # If they do, no API calls needed — skip the cost warning
            has_manifest = args.manifest is not None
            if not has_manifest:
                from src.data.ingest.manifest import ManifestGenerator
                mg = ManifestGenerator(config=config, logger=logger)
                chunk_starts = pd.date_range(start, end, freq="MS").tolist()
                if not chunk_starts or chunk_starts[0] > pd.Timestamp(start):
                    chunk_starts.insert(0, pd.Timestamp(start))
                has_manifest = all(
                    mg.get_manifest_path(cs.date()).exists() for cs in chunk_starts
                )

            if not has_manifest:
                num_days = (end - start).days
                # Heuristic: ~$0.0044 per day for SPY manifest generation
                # Calibrated from 05/2018-02/2026 ingestion: $12.46 for 2860 days
                estimated_manifest_cost = num_days * 0.0044

                print("\n" + "=" * 60)
                print("PREVIEW MODE WARNING")
                print("=" * 60)
                print("Generating manifests requires API calls and will cost money.")
                print(f"\nDate range: {start.date()} to {end.date()} ({num_days} days)")
                print(f"Estimated manifest generation cost: ~${estimated_manifest_cost:.2f}")
                print("\nNote: This is ONLY the cost to generate manifests.")
                print("      Actual data ingestion will cost additional.")
                print("=" * 60)

                response = input("\nProceed with preview? [y/N]: ").strip().lower()
                if response not in ("y", "yes"):
                    logger.info("Preview cancelled by user")
                    print("\nPreview cancelled.")
                    return 0
            else:
                logger.info("All manifests already cached on disk — no generation cost")

        # Create orchestrator
        if args.mock:
            # Use mock providers for testing
            from src.data.providers.mock import MockMacroDataProvider, MockMarketDataProvider
            from src.data.storage.engine import create_engine
            from src.data.storage.repository import RawRepository

            market_provider = MockMarketDataProvider()
            macro_provider = MockMacroDataProvider()
            db_engine = create_engine(config.paths.db_path)
            repository = RawRepository(db_engine.engine)

            from src.data.ingest.orchestrator import IngestionOrchestrator

            orchestrator = IngestionOrchestrator(
                market_provider=market_provider,
                macro_provider=macro_provider,
                repository=repository,
                config=config,
            )
            logger.info("Using mock providers (--mock flag)")
        else:
            orchestrator = create_orchestrator(config)

        # Run ingestion
        manifest_path = Path(args.manifest) if args.manifest else None

        # For non-preview mode: prompt for cost confirmation if interactive
        skip_cost_check = False
        if not args.preview_only and not args.fred_only and not args.stats_only and sys.stdin.isatty() and not args.yes:
            logger.info("Running cost preview before actual ingestion...")
            preview_result = orchestrator.run_ingestion(
                start=start,
                end=end,
                preview_only=True,
                skip_validation=args.skip_validation,
                manifest_path=manifest_path,
                force=args.force,
                quotes_only=getattr(args, "quotes_only", False),
                stats_only=getattr(args, "stats_only", False),
            )

            # Show cost breakdown and prompt for confirmation
            print("\n" + "=" * 60)
            print("COST CONFIRMATION")
            print("=" * 60)
            print(f"Date range: {start.date()} to {end.date()}")
            if preview_result.adjusted_end_date:
                print(f"Adjusted to: {start.date()} to {preview_result.adjusted_end_date.date()}")
                print("  (Data availability limited)")
            print(f"\nEstimated ingestion cost: ${preview_result.estimated_cost:.2f}")

            if preview_result.cost_limit_exceeded:
                max_cost = config.data.ingestion.databento.cost.max_usd
                print(f"Configured limit: ${max_cost:.2f}")
                print("\n" + "!" * 60)
                print("WARNING: Cost exceeds configured limit!")
                print("!" * 60)
                print("Ingestion will be blocked unless you:")
                print("  1. Reduce the date range, OR")
                print("  2. Increase cost.max_usd in your config")
                print("=" * 60)
                return 1

            print("=" * 60)
            response = input("\nProceed with ingestion? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Ingestion cancelled by user")
                print("\nIngestion cancelled.")
                return 0

            # Proceed with actual ingestion - skip redundant cost check
            logger.info("User confirmed, proceeding with ingestion...")
            skip_cost_check = True

        logger.info(
            f"{'Preview' if args.preview_only else 'Starting ingestion'}: "
            f"{start.date()} to {end.date()}"
        )

        result = orchestrator.run_ingestion(
            start=start,
            end=end,
            preview_only=args.preview_only,
            skip_validation=args.skip_validation,
            manifest_path=manifest_path,
            force=args.force,
            skip_cost_check=skip_cost_check,
            fred_only=getattr(args, "fred_only", False),
            quotes_only=getattr(args, "quotes_only", False),
            stats_only=getattr(args, "stats_only", False),
        )

        # Print result
        print("\n" + "=" * 60)
        print("INGESTION RESULT")
        print("=" * 60)
        print(result.summary())

        # Show warning if date range was adjusted due to data availability
        if result.adjusted_end_date:
            print("\n" + "!" * 60)
            print("DATE RANGE ADJUSTED")
            print("!" * 60)
            print(f"Requested end date: {end.date()}")
            print(f"Actual data available until: {result.adjusted_end_date.date()}")
            print("\nThe date range was automatically truncated due to:")
            print("  - Historical data availability limits, OR")
            print("  - Live data license requirements for recent data")
            print("!" * 60)

        if result.preview_only:
            if result.adjusted_end_date:
                print(f"\nEstimated cost: ${result.estimated_cost:.2f}")
                print(f"  (for adjusted date range: {start.date()} to {result.adjusted_end_date.date()})")
            else:
                print(f"\nEstimated cost: ${result.estimated_cost:.2f}")

            # Show warning if cost limit exceeded
            if result.cost_limit_exceeded:
                print("\n" + "!" * 60)
                print("WARNING: COST LIMIT EXCEEDED")
                print("!" * 60)
                print("The estimated cost exceeds your configured limit.")
                print("Actual ingestion will be blocked unless you:")
                print("  1. Reduce the date range, OR")
                print("  2. Increase cost.max_usd in your config")
                print("!" * 60)

            print("\nRun without --preview-only to execute ingestion.")
            return 0

        print(f"\nRun ID: {result.run_id}")
        print(f"Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
        print(f"Total rows: {result.total_rows}")
        print(f"  - Underlying bars: {result.underlying_rows}")
        print(f"  - Option quotes: {result.option_rows}")
        print(f"  - Option statistics: {result.statistics_rows}")
        print(f"  - Macro observations: {result.macro_rows}")

        if result.skipped_underlying or result.skipped_chunks or result.skipped_series:
            print("\nSkipped (already ingested):")
            if result.skipped_underlying:
                print("  - Underlying bars: loaded from DB")
            if result.skipped_chunks:
                print(f"  - Option quote chunks: {result.skipped_chunks}")
            if result.skipped_series:
                print(f"  - FRED series: {result.skipped_series}")
            print("  (use --force to re-fetch)")

        if result.failed_chunks:
            print(f"\nFailed chunks ({len(result.failed_chunks)}):")
            for chunk_desc in result.failed_chunks:
                print(f"  - {chunk_desc}")
            print("  (re-run without --force to retry only failed chunks)")

        if result.validation_results:
            print("\nValidation results:")
            for name, vr in result.validation_results.items():
                status = "PASS" if vr.passed else "FAIL"
                print(f"  - {name}: {status} ({vr.total_rows} rows)")

        if result.errors:
            print("\nWarnings/Errors:")
            for error in result.errors:
                print(f"  - {error}")

        if result.success:
            print("\nIngestion completed successfully.")
            return 0
        elif result.partial_success:
            print("\nIngestion completed with warnings (data was written).")
            return 0
        else:
            print("\nIngestion failed — no data written.")
            return 1

    except CostExceededException as e:
        logger.error(f"Cost exceeded: {e}")
        print(f"\nERROR: {e}")
        print("Use --preview-only to see estimated costs before running.")
        return 1

    except DataQualityException as e:
        logger.error(f"Data quality check failed: {e}")
        print(f"\nERROR: {e}")
        print("Use --skip-validation to bypass quality checks (not recommended).")
        return 1

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
