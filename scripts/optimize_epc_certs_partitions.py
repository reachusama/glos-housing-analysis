#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import duckdb
from typing import Tuple

def _esc(s: str) -> str: return s.replace("'", "''")
def _rm(p: Path):
    if p.exists(): shutil.rmtree(p)
def _mk(p: Path): p.mkdir(parents=True, exist_ok=True)

def _year_bounds(con, certs_in: str, lodg_expr: str) -> Tuple[int,int]:
    q = f"SELECT min({lodg_expr}) AS dmin, max({lodg_expr}) AS dmax FROM read_parquet('{_esc(certs_in)}')"
    dmin, dmax = con.execute(q).fetchone()
    if dmin is None or dmax is None:
        raise RuntimeError("Could not detect lodgement date range (no dates parsed).")
    return int(dmin.strftime("%Y")), int(dmax.strftime("%Y"))

def main(certs_in: str,
         out_dir: str,
         compression: str = "zstd",
         threads: int = 1,
         overwrite: bool = False,
         memory_limit: str = "4GB",
         temp_dir: str = "/tmp/duckdb_spill",
         start_year: int | None = None,
         end_year: int | None = None):

    out = Path(out_dir)
    final_dir = out / "certificates"            # final: outcode2 first
    tmp_years = out / "_tmp_certs_by_year"      # temp: per-year writes
    link_out  = out / "link_keys"               # tiny helper (optional)

    if overwrite:
        for p in (final_dir, tmp_years, link_out): _rm(p)
    for p in (final_dir, tmp_years, link_out): _mk(p)

    con = duckdb.connect()
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    if threads and threads > 0: con.execute(f"PRAGMA threads={threads};")
    _mk(Path(temp_dir))
    con.execute(f"PRAGMA temp_directory='{_esc(str(temp_dir))}';")

    # Introspect columns (we keep ALL of them)
    cols = [c[0] for c in con.execute(f"SELECT * FROM read_parquet('{_esc(certs_in)}') LIMIT 0").description]
    up = {c.upper(): c for c in cols}

    # Normalizations
    postcode_col = up.get("POSTCODE") or up.get("POST_CODE") or up.get("POSTAL_CODE") or up.get("POSTCODE_TEXT")
    postcode_expr = f"replace(upper(CAST({postcode_col} AS VARCHAR)), ' ', '')" if postcode_col else "NULL::VARCHAR"

    addr_candidates = ["ADDRESS1","ADDRESS2","ADDRESS3","ADDRESS_LINE1","ADDRESS_LINE2","ADDRESS_LINE3",
                       "BUILDING_NUMBER","BUILDING_NAME","FLAT_NUMBER","STREET"]
    addr_actual = [up[a] for a in addr_candidates if a in up]
    if addr_actual:
        pieces = " || ' ' || ".join([f"coalesce(CAST({c} AS VARCHAR), '')" for c in addr_actual])
        addr_expr = f"regexp_replace(upper({pieces}), '[^A-Z0-9]', '', 'g')"
    else:
        addr_expr = "NULL::VARCHAR"

    lodg_expr = """
      COALESCE(
        try_strptime(CAST(LODGEMENT_DATE AS VARCHAR), '%Y-%m-%d'),
        try_strptime(CAST(LODGEMENT_DATE AS VARCHAR), '%Y-%m-%d %H:%M'),
        try_strptime(CAST(LODGEMENT_DATE AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
        try_strptime(CAST(INSPECTION_DATE AS VARCHAR), '%Y-%m-%d'),
        try_strptime(CAST(INSPECTION_DATE AS VARCHAR), '%Y-%m-%d %H:%M'),
        try_strptime(CAST(INSPECTION_DATE AS VARCHAR), '%Y-%m-%d %H:%M:%S')
      )::DATE
    """

    # Discover year range (constant memory)
    y0, y1 = _year_bounds(con, certs_in, lodg_expr)
    if start_year is not None: y0 = max(y0, start_year)
    if end_year   is not None: y1 = min(y1, end_year)

    # Base SELECT (adds normalized fields; keeps all original columns via src.*)
    base_sql = f"""
    WITH src AS (SELECT * FROM read_parquet('{_esc(certs_in)}')),
    norm AS (
      SELECT src.*, {postcode_expr} AS postcode_norm, {lodg_expr} AS lodgement_date
      FROM src
    ),
    keyed AS (
      SELECT
        *,
        CASE WHEN postcode_norm IS NOT NULL AND length(postcode_norm) >= 4
             THEN substr(postcode_norm, 1, length(postcode_norm)-3) END AS outcode,
        CASE WHEN postcode_norm IS NOT NULL AND length(postcode_norm) >= 2
             THEN substr(postcode_norm, 1, 2) END AS outcode2,
        strftime(lodgement_date, '%Y') AS lodgement_year,
        {addr_expr} AS addr_key
      FROM norm
    )
    SELECT * FROM keyed
    """

    # 1) Per-year writes → tmp tree: lodgement_year=YYYY/outcode2=XX/...
    for y in range(y0, y1 + 1):
        print(f"[certificates] temp write for year {y} …")
        year_dir = tmp_years / f"lodgement_year={y}"
        _mk(year_dir)
        year_sql = f"""
        {base_sql}
        WHERE lodgement_date >= DATE '{y}-01-01' AND lodgement_date < DATE '{y+1}-01-01'
        """
        con.execute(f"""
          COPY ({year_sql})
          TO '{_esc(str(year_dir))}'
          (FORMAT PARQUET, COMPRESSION {compression.upper()},
           PARTITION_BY (outcode2));
        """)

    # 2) Single final write → outcode2 first, then year
    print("[certificates] consolidating to outcode2-first layout …")
    _rm(final_dir); _mk(final_dir)
    con.execute(f"""
      COPY (
        SELECT * FROM read_parquet('{_esc(str(tmp_years))}/lodgement_year=*/outcode2=*/*.parquet')
      )
      TO '{_esc(str(final_dir))}'
      (FORMAT PARQUET, COMPRESSION {compression.upper()},
       PARTITION_BY (outcode2, lodgement_year));
    """)

    # 3) Optional tiny link_keys (handy later; safe to keep now)
    lmk_present = "LMK_KEY" in up
    lmk_col = up.get("LMK_KEY")
    lmk_select = f"{lmk_col}, " if lmk_present else ""
    con.execute(f"""
      COPY (
        SELECT {lmk_select} postcode_norm, addr_key, outcode2, lodgement_year
        FROM read_parquet('{_esc(str(final_dir))}/outcode2=*/lodgement_year=*/*.parquet')
      )
      TO '{_esc(str(link_out / "epc_cert_link_keys.parquet"))}'
      (FORMAT PARQUET, COMPRESSION {compression.upper()});
    """)

    # 4) Clean temp
    _rm(tmp_years)

    con.close()
    print(f"[DONE] Wrote {final_dir} (partitioned by outcode2 → lodgement_year).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EPC certificates → Parquet (outcode2 first, low RAM, keep all columns)")
    ap.add_argument("--certs-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compression", default="zstd", choices=["zstd","snappy","gzip","uncompressed"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--memory-limit", default="4GB", help="e.g., 2GB, 6GB, 80%")
    ap.add_argument("--temp-dir", default="/tmp/duckdb_spill")
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    args = ap.parse_args()
    main(args.certs_in, args.out, args.compression, args.threads, args.overwrite,
         args.memory_limit, args.temp_dir, args.start_year, args.end_year)
