#!/usr/bin/env python
"""Restore raw_option_quotes from backup database."""

import sqlite3
import time

BACKUP_PATH = "/path/to/backup/db.sqlite"  # Set to your backup location
CURRENT_PATH = "data/db.sqlite"
BATCH_SIZE = 500_000


def main():
    backup = sqlite3.connect(BACKUP_PATH)
    current = sqlite3.connect(CURRENT_PATH)
    current.execute("PRAGMA journal_mode=WAL")
    current.execute("PRAGMA synchronous=NORMAL")

    total = backup.execute("SELECT COUNT(*) FROM raw_option_quotes").fetchone()[0]
    print(f"Backup has {total:,} rows to copy.")

    print("Deleting current option quotes...")
    current.execute("DELETE FROM raw_option_quotes")
    current.commit()
    print("Done.\n")

    copied = 0
    t0 = time.monotonic()

    cursor = backup.execute(
        """SELECT dataset, schema, stype_in, instrument_id, publisher_id,
                  ts_utc, ts_recv_utc, option_symbol, exp_date, strike, "right",
                  bid, ask, bid_size, ask_size, volume, open_interest,
                  source_ingested_at, ingest_run_id
           FROM raw_option_quotes"""
    )

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        current.executemany(
            """INSERT INTO raw_option_quotes
               (dataset, schema, stype_in, instrument_id, publisher_id,
                ts_utc, ts_recv_utc, option_symbol, exp_date, strike, "right",
                bid, ask, bid_size, ask_size, volume, open_interest,
                source_ingested_at, ingest_run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        current.commit()

        copied += len(rows)
        elapsed = time.monotonic() - t0
        pct = copied / total * 100
        rate = copied / elapsed if elapsed > 0 else 0
        eta = (total - copied) / rate if rate > 0 else 0
        print(f"  {copied:>12,} / {total:,}  ({pct:5.1f}%)  "
              f"{rate:,.0f} rows/s  ETA {eta:.0f}s")

    elapsed = time.monotonic() - t0
    print(f"\nCopied {copied:,} rows in {elapsed:.0f}s.")

    # Verify
    final = current.execute("SELECT COUNT(*) FROM raw_option_quotes").fetchone()[0]
    print(f"Verification: {final:,} rows ({'✓ match' if final == total else 'MISMATCH'})")

    backup.close()
    current.close()


if __name__ == "__main__":
    main()
