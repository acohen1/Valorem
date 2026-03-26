#!/usr/bin/env python
"""CLI tool for inspecting and managing database tables.

Subcommands:
    list    Show all tables with row counts and disk usage
    clear   Delete rows from specific tables or table groups

Usage:
    # List all tables and row counts
    python scripts/manage_data.py list

    # Clear derived tables (surfaces + node_panel) — keeps raw ingested data
    python scripts/manage_data.py clear --derived

    # Clear a specific table
    python scripts/manage_data.py clear surface_snapshots

    # Clear multiple tables
    python scripts/manage_data.py clear surface_snapshots node_panel

    # Clear all tables (requires --yes to skip confirmation)
    python scripts/manage_data.py clear --all

    # Skip confirmation prompt
    python scripts/manage_data.py clear --derived --yes
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import inspect, text

from src.config.loader import ConfigLoader
from src.config.logging import setup_logging
from src.data.storage.engine import create_engine

# Table groupings
RAW_TABLES = [
    "raw_underlying_bars",
    "raw_option_quotes",
    "raw_option_bars",
    "raw_fred_series",
    "raw_ingestion_log",
]

DERIVED_TABLES = [
    "surface_snapshots",
    "node_panel",
]

ALL_TABLES = RAW_TABLES + DERIVED_TABLES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect and manage database tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- list ---
    subparsers.add_parser("list", help="Show tables with row counts")

    # --- clear ---
    clear_parser = subparsers.add_parser("clear", help="Delete rows from tables")
    clear_parser.add_argument(
        "tables",
        nargs="*",
        help="Table names to clear (e.g. surface_snapshots node_panel)",
    )
    clear_group = clear_parser.add_mutually_exclusive_group()
    clear_group.add_argument(
        "--derived",
        action="store_true",
        help="Clear derived tables (surface_snapshots, node_panel)",
    )
    clear_group.add_argument(
        "--raw",
        action="store_true",
        help="Clear raw ingested tables (USE WITH CAUTION — re-ingestion costs money)",
    )
    clear_group.add_argument(
        "--all",
        action="store_true",
        help="Clear all tables",
    )
    clear_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    return parser.parse_args()


def get_table_row_count(engine, table_name: str) -> int:
    """Get row count for a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM [{table_name}]"))
        return result.scalar()


def get_db_size(db_path: Path) -> str:
    """Get database file size as human-readable string."""
    if not db_path.exists():
        return "N/A"
    size_bytes = db_path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def cmd_list(engine, db_path: Path) -> int:
    """List all tables with row counts."""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if not existing_tables:
        print("No tables found in database.")
        return 0

    print(f"\nDatabase: {db_path} ({get_db_size(db_path)})")
    print(f"{'Table':<25} {'Rows':>12}  {'Group':<10}")
    print("-" * 52)

    total_rows = 0
    for table in ALL_TABLES:
        if table not in existing_tables:
            continue
        count = get_table_row_count(engine, table)
        total_rows += count
        group = "raw" if table in RAW_TABLES else "derived"
        count_str = f"{count:,}"
        print(f"  {table:<23} {count_str:>12}  {group:<10}")

    # Show any tables not in our known list
    unknown = set(existing_tables) - set(ALL_TABLES)
    for table in sorted(unknown):
        count = get_table_row_count(engine, table)
        total_rows += count
        count_str = f"{count:,}"
        print(f"  {table:<23} {count_str:>12}  {'unknown':<10}")

    print("-" * 52)
    print(f"  {'Total':<23} {total_rows:>12,}")
    print()

    return 0


def cmd_clear(engine, db_path: Path, args: argparse.Namespace) -> int:
    """Clear rows from specified tables."""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    # Determine which tables to clear
    if args.all:
        targets = [t for t in ALL_TABLES if t in existing_tables]
    elif args.raw:
        targets = [t for t in RAW_TABLES if t in existing_tables]
    elif args.derived:
        targets = [t for t in DERIVED_TABLES if t in existing_tables]
    elif args.tables:
        # Validate table names
        for t in args.tables:
            if t not in existing_tables:
                print(f"Error: table '{t}' does not exist. Available: {', '.join(existing_tables)}")
                return 1
        targets = args.tables
    else:
        print("Error: specify table names, or use --derived / --raw / --all")
        return 1

    if not targets:
        print("No matching tables found.")
        return 0

    # Show what will be cleared
    print(f"\nDatabase: {db_path}")
    print(f"\nTables to clear:")
    total_rows = 0
    for table in targets:
        count = get_table_row_count(engine, table)
        total_rows += count
        print(f"  {table:<25} {count:>12,} rows")
    print(f"  {'Total':<25} {total_rows:>12,} rows")

    # Warn about raw tables
    has_raw = any(t in RAW_TABLES for t in targets)
    if has_raw:
        print("\n  WARNING: Clearing raw tables requires re-ingestion (costs money)!")

    # Confirm
    if not args.yes:
        response = input("\nProceed? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    # Drop and let create_all() recreate with current schema
    with engine.begin() as conn:
        for table in targets:
            conn.execute(text(f"DROP TABLE IF EXISTS [{table}]"))
            logger.info(f"Dropped {table}")

    # Recreate tables with current schema
    from src.data.storage.schema import Base
    Base.metadata.create_all(engine)
    logger.info("Recreated tables with current schema")

    # VACUUM to reclaim disk space
    print("Reclaiming disk space (VACUUM)...")
    with engine.connect() as conn:
        conn.execute(text("VACUUM"))

    print(f"\nCleared {len(targets)} table(s). DB size: {get_db_size(db_path)}")
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    config = ConfigLoader.load(Path(args.config), env=args.env)
    db_path = Path(config.paths.db_path).resolve()

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    db_engine = create_engine(db_path)
    engine = db_engine.engine

    if args.command == "list":
        return cmd_list(engine, db_path)
    elif args.command == "clear":
        return cmd_clear(engine, db_path, args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
