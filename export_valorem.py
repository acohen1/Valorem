#!/usr/bin/env python
"""
export_valorem.py
-----------------
Dump *all* tables in valorem.db into:
  valorem_dump.xlsx   (one worksheet per table)
  valorem_master_dump.csv   (all rows stacked, plus __table__ column)

Requires only pandas
"""

from __future__ import annotations
import sqlite3
import pandas as pd
import pathlib
import sys

DB_PATH   = pathlib.Path("valorem.db")
XLSX_OUT  = pathlib.Path("valorem_master_dump.xlsx")
CSV_OUT   = pathlib.Path("valorem_master_dump.csv")

def main() -> None:
    if not DB_PATH.exists():
        sys.exit(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as con, pd.ExcelWriter(XLSX_OUT, engine="xlsxwriter") as xls:
        master_frames: list[pd.DataFrame] = []

        # Pull table list
        tables = [row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )]

        if not tables:
            sys.exit("No tables found in database.")

        for tbl in tables:
            df = pd.read_sql(f"SELECT * FROM {tbl};", con)
            df.to_excel(xls, sheet_name=tbl[:31], index=False)  # Excel sheet names ≤31 chars
            df.insert(0, "__table__", tbl)      # tag for master CSV
            master_frames.append(df)
            print(f"✓ wrote {tbl:<25} → sheet, {len(df):>10,} rows")

        # Concatenate all DataFrames (outer join on columns)
        master_df = pd.concat(master_frames, ignore_index=True, sort=False)
        master_df.to_csv(CSV_OUT, index=False)
        print(f"\n✓ master CSV: {CSV_OUT}  ({len(master_df):,} total rows)")

        print(f"✓ Excel workbook: {XLSX_OUT}  ({len(tables)} sheets)")

if __name__ == "__main__":
    main()
