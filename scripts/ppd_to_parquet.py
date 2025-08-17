#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
from typing import Optional, Tuple, List
import duckdb

PPD_CANONICAL_COLUMNS = [
    "Transaction ID","Price","Date of Transfer","Postcode","Property Type","Old/New",
    "Duration","PAON","SAON","Street","Locality","Town/City","District","County",
    "PPD Category Type","Record Status",
]

DUCK_TYPES = {
    "Transaction ID": "VARCHAR",
    "Price": "BIGINT",
    "Date of Transfer": "VARCHAR",
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

PARTITION_CHOICES = {
    "outcode2_year_month": ("outcode2", "sale_year", "sale_month"),
    "year_month_outcode2": ("sale_year", "sale_month", "outcode2"),
    "outcode2_year": ("outcode2", "sale_year"),
    "year_month": ("sale_year", "sale_month"),
    "outcode2_only": ("outcode2",),
    "none": tuple(),
}

def _columns_map_sql() -> str:
    return "{" + ", ".join([f"'{k}':'{v}'" for k, v in DUCK_TYPES.items()]) + "}"

def _esc(s: str) -> str: return s.replace("'", "''")

def _partition_clause(name: str) -> Tuple[str, bool]:
    cols = PARTITION_CHOICES[name]
    if not cols: return "", False
    return f"PARTITION_BY ({', '.join(cols)})", True

def _detect_format(path: Path, user_choice: str) -> str:
    if user_choice != "auto": return user_choice
    if path.is_dir(): return "parquet" if any(path.rglob("*.parquet")) else "csv"
    return "parquet" if str(path).lower().endswith(".parquet") else "csv"

def _resolve_col(actual_cols: List[str], candidates: List[str]) -> Optional[str]:
    ac = {c.lower(): c for c in actual_cols}
    for cand in candidates:
        if cand.lower() in ac:
            return ac[cand.lower()]
    return None

def _consolidate_leaves(root: Path, compression: str = "zstd"):
    """Ensure exactly one file per leaf partition directory."""
    con = duckdb.connect()
    con.execute("PRAGMA threads=1; PRAGMA preserve_insertion_order=false;")
    # A leaf dir is any dir that contains parquet files and no subdirs with parquet
    for leaf in sorted({p.parent for p in root.rglob("*.parquet")}):
        files = sorted(leaf.glob("*.parquet"))
        if len(files) <= 1:
            continue
        tmp = leaf / "_tmp.parquet"
        con.execute(f"""
          COPY (
            SELECT * FROM read_parquet('{_esc(str(leaf))}/*.parquet')
          )
          TO '{_esc(str(tmp))}'
          (FORMAT PARQUET, COMPRESSION {compression.upper()});
        """)
        for f in files:
            if f.name != "_tmp.parquet":
                f.unlink()
        tmp.rename(leaf / "data_0.parquet")
        print(f"[consolidate] {leaf} -> 1 file")

def main(input_path: str,
         output_dir: str,
         delimiter: Optional[str] = None,
         compression: str = "zstd",
         partition: str = "year_month_outcode2",
         threads: int = 1,                           # default 1 to prefer single-file leaves
         overwrite: bool = False,
         memory_limit: Optional[str] = None,
         input_format: str = "auto",
         postcode_col: Optional[str] = None,
         date_col: Optional[str] = None,
         year_col: Optional[str] = None,
         month_col: Optional[str] = None,
         parquet_positional: bool = False,
         outcode2_na: str = "ZZ",                    # sentinel for missing/invalid outcode2
         single_file_per_partition: bool = True) -> None:

    inp = Path(input_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    fmt = _detect_format(inp, input_format)
    delim = delimiter or ","
    out_base_dir = out / "ppd"

    if overwrite:
        if out_base_dir.exists() and out_base_dir.is_dir(): shutil.rmtree(out_base_dir)
        out_base_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA preserve_insertion_order=false;")
    if threads > 0: con.execute(f"PRAGMA threads={threads};")
    if memory_limit: con.execute(f"PRAGMA memory_limit='{_esc(memory_limit)}';")

    # ---------- Source + schema ----------
    if fmt == "csv":
        columns_map = _columns_map_sql()
        src_cte = f"""
        src AS (
          SELECT *
          FROM read_csv_auto('{_esc(str(inp))}',
                             header = FALSE,
                             delim = '{_esc(delim)}',
                             quote = '"',
                             columns = {columns_map},
                             sample_size = -1)
        )
        """
        base_from = "src"
        schema_cols = [d[0] for d in con.execute(
            f"SELECT * FROM read_csv_auto('{_esc(str(inp))}', header=FALSE, delim='{_esc(delim)}', quote='\"', columns={columns_map}, sample_size=1) LIMIT 0"
        ).description]
        resolved_date_col = date_col or "Date of Transfer"
        resolved_postcode_col = postcode_col or "Postcode"
        select_base = ""
    else:
        pq = str(inp / "**/*.parquet") if inp.is_dir() else str(inp)
        src_cte = f"src AS (SELECT * FROM read_parquet('{_esc(pq)}', hive_partitioning=1))"
        base_from = "src"
        schema_cols = [d[0] for d in con.execute(
            f"SELECT * FROM read_parquet('{_esc(pq)}', hive_partitioning=1) LIMIT 0"
        ).description]

        need_positional = parquet_positional or (
            len(schema_cols) == len(PPD_CANONICAL_COLUMNS) and
            set(c.lower() for c in schema_cols) != set(c.lower() for c in PPD_CANONICAL_COLUMNS)
        )

        if need_positional:
            proj = ",\n            ".join(
                f"\"{_esc(schema_cols[i])}\" AS \"{_esc(PPD_CANONICAL_COLUMNS[i])}\""
                for i in range(len(PPD_CANONICAL_COLUMNS))
            )
            select_base = f"""
            base AS (
              SELECT
                {proj}
              FROM src
            )
            """
            base_from = "base"
            resolved_date_col = date_col or "Date of Transfer"
            resolved_postcode_col = postcode_col or "Postcode"
            print("[INFO] Parquet → applied positional rename to canonical headers.")
        else:
            select_base = ""
            resolved_date_col = date_col or _resolve_col(schema_cols, [
                "Date of Transfer","date_of_transfer","DATE_OF_TRANSFER","Date","Sale Date","Transaction Date","TransferDate","Transfer Date"
            ])
            resolved_postcode_col = postcode_col or _resolve_col(schema_cols, [
                "Postcode","postcode","POSTCODE","PCDS","pcds","Post Code"
            ])

    # Date fallback from year/month
    fallback_make_date = ""
    if not resolved_date_col:
        if year_col and month_col:
            y_res = _resolve_col(schema_cols, [year_col]); m_res = _resolve_col(schema_cols, [month_col])
            if not y_res or not m_res:
                print(f"[ERROR] Year/Month columns not found: {year_col}, {month_col}\nSchema: {schema_cols}"); sys.exit(2)
            fallback_make_date = f", make_date(try_cast(\"{y_res}\" AS INTEGER), try_cast(\"{m_res}\" AS INTEGER), 1)"
            resolved_date_col = y_res
            print(f"[INFO] No date column; constructing from {y_res}/{m_res}.")
        else:
            print("[ERROR] No date column detected. Pass --date-col or --year-col/--month-col."); sys.exit(2)

    pcode_expr = f'replace(upper(CAST("{resolved_postcode_col}" AS VARCHAR)), \' \', \'\')' if resolved_postcode_col else "NULL::VARCHAR"

    DATE_EXPR = f"""
    COALESCE(
      try_strptime(split_part(CAST("{resolved_date_col}" AS VARCHAR), ' ', 1), '%Y-%m-%d'),
      try_strptime(split_part(CAST("{resolved_date_col}" AS VARCHAR), ' ', 1), '%d/%m/%Y')
      {fallback_make_date}
    )
    """

    # --------- build SELECT (only essential derived cols) ----------
    enriched_cte = f"""
    enriched AS (
      SELECT
        {base_from}.*
        REPLACE (
          ({DATE_EXPR})::DATE AS "Date of Transfer",
          ({pcode_expr})      AS "Postcode"
        ),
        -- minimal date parts for partitioning
        strftime(({DATE_EXPR}), '%Y') AS sale_year,
        strftime(({DATE_EXPR}), '%m') AS sale_month,

        -- postcode pieces
        regexp_extract("Postcode", '^([A-Z]{{1,2}}[0-9]{{1,2}}[A-Z]?)', 1) AS "Outward",
        regexp_extract("Postcode", '^([A-Z]{{1,2}}[0-9]{{1,2}}[A-Z]?)([0-9])', 2) AS "Sector",
        concat_ws(' ', "Outward", "Sector") AS "Postcode Sector",

        -- outcode2 with sentinel to avoid NULL partitions
        COALESCE(
          CASE WHEN length("Outward") >= 2 THEN substr("Outward",1,2) END,
          '{_esc(outcode2_na)}'
        ) AS outcode2
      FROM {base_from}
    )
    """

    cte_parts = [src_cte.strip()]
    if select_base.strip(): cte_parts.append(select_base.strip())
    select_sql = "WITH " + ",\n".join(cte_parts + [enriched_cte.strip()]) + """
SELECT * FROM enriched
WHERE "Date of Transfer" IS NOT NULL AND "Price" IS NOT NULL
"""

    part_clause, is_partitioned = _partition_clause(partition)
    if is_partitioned:
        if not overwrite and out_base_dir.exists() and any(out_base_dir.iterdir()):
            raise RuntimeError(f"Output dir {out_base_dir} is not empty. Use --overwrite.")
        copy_sql = f"""
        COPY ({select_sql})
        TO '{_esc(str(out_base_dir))}'
        (FORMAT PARQUET, COMPRESSION {compression.upper()}, {part_clause});
        """
    else:
        out_file = out_base_dir.with_suffix(".parquet")
        if overwrite and out_file.exists(): out_file.unlink()
        copy_sql = f"""
        COPY ({select_sql})
        TO '{_esc(str(out_file))}'
        (FORMAT PARQUET, COMPRESSION {compression.upper()});
        """

    print(f"[INFO] Input: {inp} (detected: {fmt})")
    print(f"[INFO] Output: {out_base_dir} (partition={partition}, compression={compression})")
    if fmt == "parquet":
        print(f"[INFO] Schema cols: {schema_cols}")
        print(f"[INFO] Using date={resolved_date_col!r} postcode={resolved_postcode_col!r} outcode2_na='{outcode2_na}'")
    con.execute(copy_sql)

    # consolidate to 1 file per leaf (if requested)
    if is_partitioned and single_file_per_partition:
        _consolidate_leaves(out_base_dir, compression=compression)

    print("[DONE]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PPD → Parquet (CSV/Parquet; positional support; single-file leaves)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--delimiter", default=None)
    ap.add_argument("--compression", default="zstd", choices=["zstd","snappy","gzip","uncompressed"])
    ap.add_argument("--partition", default="year_month_outcode2", choices=list(PARTITION_CHOICES.keys()))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--memory-limit", default=None)
    ap.add_argument("--input-format", default="auto", choices=["auto","csv","parquet"])
    ap.add_argument("--postcode-col", default=None)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--year-col", default=None)
    ap.add_argument("--month-col", default=None)
    ap.add_argument("--parquet-positional", action="store_true")
    ap.add_argument("--outcode2-na", default="ZZ", help="Sentinel value for missing/invalid outcode2")
    ap.add_argument("--no-consolidate", dest="single_file_per_partition", action="store_false",
                    help="Skip post-pass that enforces one file per partition")
    args = ap.parse_args()

    main(args.input, args.output, args.delimiter, args.compression, args.partition,
         args.threads, args.overwrite, args.memory_limit, args.input_format,
         args.postcode_col, args.date_col, args.year_col, args.month_col,
         args.parquet_positional, args.outcode2_na, args.single_file_per_partition)
