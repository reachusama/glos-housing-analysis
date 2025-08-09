#!/usr/bin/env python3
"""
CSV/TXT -> Parquet (as-is), low memory, no partitions.
- One Parquet per input file (same basename)
- Streams via DuckDB (handles very large files)
- Supports .csv, .txt, .csv.gz, .txt.gz

CLI:
  python csv_to_parquet.py --input /path/to/file_or_folder --output /path/to/out [--all-varchar]

Import:
  from csv_to_parquet import main
  main(input_path="...", output_path="...", all_varchar=True)
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional
import duckdb

DEFAULT_GLOBS = ["*.csv", "*.txt", "*.csv.gz", "*.txt.gz"]

def _escape_literal(path: str) -> str:
    # SQL string literal escape for DuckDB (single quotes doubled)
    return path.replace("'", "''")

def convert_one(con: duckdb.DuckDBPyConnection, in_path: Path, out_dir: Path,
                compression: str = "zstd", all_varchar: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (in_path.stem + ".parquet")

    # Build read_csv_auto options
    opts = ["union_by_name=true", "sample_size=-1"]
    if all_varchar:
        opts.append("all_varchar=1")
    opt_str = ", ".join(opts)

    in_lit  = _escape_literal(str(in_path))
    out_lit = _escape_literal(str(out_path))

    sql = f"""
    COPY (
      SELECT * FROM read_csv_auto('{in_lit}', {opt_str})
    )
    TO '{out_lit}' (FORMAT PARQUET, COMPRESSION {compression.upper()});
    """
    print(f"[INFO] {in_path} -> {out_path}  (compression={compression}, all_varchar={all_varchar})")
    con.execute(sql)

def _find_files(in_path: Path, patterns: List[str]) -> List[Path]:
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        files: List[Path] = []
        for pat in patterns:
            files.extend(in_path.rglob(pat))
        return sorted(set(files))
    raise FileNotFoundError(f"Input path not found: {in_path}")

def main(input_path: str,
         output_path: str,
         patterns: Optional[List[str]] = None,
         threads: int = 0,
         compression: str = "zstd",
         all_varchar: bool = False) -> None:
    in_path = Path(input_path)
    out_dir = Path(output_path)
    patterns = patterns or DEFAULT_GLOBS

    con = duckdb.connect()
    if threads > 0:
        con.execute(f"PRAGMA threads={threads};")

    files = _find_files(in_path, patterns)
    if not files:
        print("[WARN] No input files matched.")
        return

    for f in files:
        try:
            convert_one(con, f, out_dir, compression, all_varchar)
        except Exception as e:
            print(f"[ERROR] Failed on {f}: {e}", file=sys.stderr)

    print("[DONE] Wrote Parquet to", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input file OR folder")
    ap.add_argument("--output", required=True, help="Output folder for Parquet")
    ap.add_argument("--glob", default=",".join(DEFAULT_GLOBS),
                    help="Comma-separated patterns inside folder (default: %(default)s)")
    ap.add_argument("--threads", type=int, default=0, help="DuckDB threads (0=auto)")
    ap.add_argument("--compression", choices=["zstd","snappy","gzip","uncompressed"],
                    default="zstd", help="Parquet compression")
    ap.add_argument("--all-varchar", action="store_true",
                    help="Force all columns to TEXT (avoid type inference surprises)")
    args = ap.parse_args()

    pats = [p.strip() for p in args.glob.split(",") if p.strip()]
    main(input_path=args.input,
         output_path=args.output,
         patterns=pats,
         threads=args.threads,
         compression=args.compression,
         all_varchar=args.all_varchar)
