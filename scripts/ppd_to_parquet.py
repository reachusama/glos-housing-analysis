#!/usr/bin/env python3
"""
Price Paid Data (PPD) (headerless .txt/.csv) -> Optimized Parquet via DuckDB

- Streams via DuckDB (low memory)
- Assigns canonical headers (no header in source)
- Robust date parse (handles time suffixes)
- Normalizes Postcode (upper, no spaces)
- Optional partitioning by sale_year/sale_month
"""

from __future__ import annotations
import argparse
from pathlib import Path
import duckdb
from typing import Optional

PPD_CANONICAL_COLUMNS = [
    "Transaction ID","Price","Date of Transfer","Postcode","Property Type","Old/New",
    "Duration","PAON","SAON","Street","Locality","Town/City","District","County",
    "PPD Category Type","Record Status",
]

DUCK_TYPES = {
    "Transaction ID": "VARCHAR",
    "Price": "BIGINT",
    "Date of Transfer": "VARCHAR",  # parse later
    "Postcode": "VARCHAR",
    "Property Type": "VARCHAR",
    "Old/New": "VARCHAR",
    "Duration": "VARCHAR",
    "PAON": "VARCHAR",
    "SAON": "VARCHAR",
    "Street": "VARCHAR",
    "Locality": "VARCHAR",
    "Town/City": "VARCHAR",
    "District": "VARCHAR",
    "County": "VARCHAR",
    "PPD Category Type": "VARCHAR",
    "Record Status": "VARCHAR",
}

def _columns_map_sql() -> str:
    return "{" + ", ".join([f"'{k}':'{v}'" for k, v in DUCK_TYPES.items()]) + "}"

def _esc(p: str) -> str:
    return p.replace("'", "''")

def main(input_path: str,
         output_dir: str,
         delimiter: Optional[str] = None,
         compression: str = "zstd",
         partition: bool = True,
         threads: int = 0) -> None:

    inp = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    delim = delimiter or ","
    columns_map = _columns_map_sql()
    out_base = str(out / "ppd")

    # Take the part before the first space; then try formats safely (no exceptions)
    DATE_EXPR = r"""
    COALESCE(
      try_strptime(split_part("Date of Transfer", ' ', 1), '%Y-%m-%d'),
      try_strptime(split_part("Date of Transfer", ' ', 1), '%d/%m/%Y')
    )
    """

    select_sql = f"""
    WITH raw AS (
      SELECT *
      FROM read_csv_auto('{_esc(str(inp))}',
                         header = FALSE,
                         delim = '{_esc(delim)}',
                         quote = '"',
                         columns = {columns_map},
                         sample_size = -1)
    ),
    casted AS (
      SELECT
        "Transaction ID"::VARCHAR                                     AS "Transaction ID",
        "Price"::BIGINT                                               AS "Price",
        ({DATE_EXPR})::DATE                                           AS "Date of Transfer",
        replace(upper("Postcode"), ' ', '')                           AS "Postcode",
        "Property Type"::VARCHAR                                      AS "Property Type",
        "Old/New"::VARCHAR                                            AS "Old/New",
        "Duration"::VARCHAR                                           AS "Duration",
        "PAON"::VARCHAR                                               AS "PAON",
        "SAON"::VARCHAR                                               AS "SAON",
        "Street"::VARCHAR                                             AS "Street",
        "Locality"::VARCHAR                                           AS "Locality",
        "Town/City"::VARCHAR                                          AS "Town/City",
        "District"::VARCHAR                                           AS "District",
        "County"::VARCHAR                                             AS "County",
        "PPD Category Type"::VARCHAR                                  AS "PPD Category Type",
        "Record Status"::VARCHAR                                      AS "Record Status",
        strftime(({DATE_EXPR}), '%Y')                                  AS sale_year,
        strftime(({DATE_EXPR}), '%m')                                  AS sale_month
      FROM raw
    )
    SELECT * FROM casted
    """

    con = duckdb.connect()
    if threads > 0:
        con.execute(f"PRAGMA threads={threads};")

    if partition:
        copy_sql = f"""
        COPY (
          {select_sql}
        )
        TO '{_esc(out_base)}'
        (FORMAT PARQUET,
         COMPRESSION {compression.upper()},
         PARTITION_BY (sale_year, sale_month));
        """
    else:
        copy_sql = f"""
        COPY (
          {select_sql}
        )
        TO '{_esc(out_base)}.parquet'
        (FORMAT PARQUET,
         COMPRESSION {compression.upper()});
        """

    print(f"[INFO] Reading: {inp}")
    print(f"[INFO] Writing: {out} (compression={compression}, partition={partition})")
    con.execute(copy_sql)
    print("[DONE]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PPD (headerless) -> Parquet via DuckDB")
    ap.add_argument("--input", required=True, help="Path to PPD .txt/.csv (no header)")
    ap.add_argument("--output", required=True, help="Output folder for Parquet")
    ap.add_argument("--delimiter", default=None, help="Override delimiter (default: ',')")
    ap.add_argument("--compression", default="zstd",
                    choices=["zstd","snappy","gzip","uncompressed"])
    ap.add_argument("--no-partition", action="store_true", help="Write single Parquet instead")
    ap.add_argument("--threads", type=int, default=0, help="DuckDB threads (0 uses default)")
    args = ap.parse_args()

    main(input_path=args.input,
         output_dir=args.output,
         delimiter=args.delimiter,
         compression=args.compression,
         partition=not args.no_partition,
         threads=args.threads)
