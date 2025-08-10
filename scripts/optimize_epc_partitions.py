#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import duckdb


def _esc(s: str) -> str: return s.replace("'", "''")


def _nuke_dir(p: Path):
    if p.exists(): shutil.rmtree(p)


def main(certs_in: str,
         recs_in: str,
         out_dir: str,
         compression: str = "zstd",
         threads: int = 0,
         recs_postcode_col: str | None = None,
         overwrite: bool = False,
         memory_limit: str | None = None,
         temp_dir: str | None = None,
         chunked: bool = False):
    out = Path(out_dir)
    cert_out = out / "certificates"
    rec_out = out / "recommendations"
    link_out = out / "link_keys"

    if overwrite:
        for p in (cert_out, rec_out, link_out):
            _nuke_dir(p)

    for p in (cert_out, rec_out, link_out):
        p.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    # Safety/tuning pragmas
    if threads and threads > 0: con.execute(f"PRAGMA threads={threads};")
    con.execute("PRAGMA preserve_insertion_order=false;")
    if memory_limit: con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{_esc(str(temp_dir))}';")

    # ---------- Certificates -> partition (outcode2, lodgement_year) ----------
    certs_sql = f"""
    WITH src AS (SELECT * FROM read_parquet('{_esc(certs_in)}')),
    norm AS (
      SELECT
        CAST(LMK_KEY AS VARCHAR)                                      AS LMK_KEY,
        replace(upper(CAST(POSTCODE AS VARCHAR)), ' ', '')            AS postcode_norm,
        CAST(ADDRESS1 AS VARCHAR)                                     AS address1,
        CAST(ADDRESS2 AS VARCHAR)                                     AS address2,
        CAST(ADDRESS3 AS VARCHAR)                                     AS address3,
        CAST(PROPERTY_TYPE AS VARCHAR)                                AS property_type,
        CAST(BUILT_FORM AS VARCHAR)                                   AS built_form,
        CAST(BUILDING_REFERENCE_NUMBER AS VARCHAR)                    AS building_reference_number,
        CAST(CURRENT_ENERGY_RATING AS VARCHAR)                        AS current_energy_rating,
        CAST(POTENTIAL_ENERGY_RATING AS VARCHAR)                      AS potential_energy_rating,
        CAST(CURRENT_ENERGY_EFFICIENCY AS BIGINT)                     AS current_energy_efficiency,
        CAST(POTENTIAL_ENERGY_EFFICIENCY AS BIGINT)                   AS potential_energy_efficiency,
        COALESCE(
          try_strptime(CAST(LODGEMENT_DATE AS VARCHAR), '%Y-%m-%d'),
          try_strptime(CAST(LODGEMENT_DATE AS VARCHAR), '%Y-%m-%d %H:%M'),
          try_strptime(CAST(LODGEMENT_DATE AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
          try_strptime(CAST(INSPECTION_DATE AS VARCHAR), '%Y-%m-%d'),
          try_strptime(CAST(INSPECTION_DATE AS VARCHAR), '%Y-%m-%d %H:%M'),
          try_strptime(CAST(INSPECTION_DATE AS VARCHAR), '%Y-%m-%d %H:%M:%S')
        )::DATE                                                       AS lodgement_date
      FROM src
    ),
    keyed AS (
      SELECT
        *,
        regexp_replace(
          upper(coalesce(address1,'') || ' ' || coalesce(address2,'') || ' ' || coalesce(address3,'')),
          '[^A-Z0-9]', '', 'g'
        ) AS addr_key,
        strftime(lodgement_date, '%Y') AS lodgement_year,
        CASE WHEN length(postcode_norm) >= 4 THEN substr(postcode_norm, 1, length(postcode_norm)-3) END AS outcode,
        CASE WHEN length(postcode_norm) >= 2 THEN substr(postcode_norm, 1, 2) END                      AS outcode2
      FROM norm
      WHERE LMK_KEY IS NOT NULL AND lodgement_date IS NOT NULL
    ),
    ordered AS (
      SELECT * FROM keyed
      ORDER BY outcode2, outcode, postcode_norm, lodgement_date, LMK_KEY
    )
    SELECT
      LMK_KEY, postcode_norm, outcode, outcode2,
      address1, address2, address3,
      property_type, built_form,
      building_reference_number,
      current_energy_rating, potential_energy_rating,
      current_energy_efficiency, potential_energy_efficiency,
      lodgement_date, lodgement_year,
      addr_key
    FROM ordered
    """
    con.execute(f"""
      COPY ({certs_sql})
      TO '{_esc(str(cert_out))}'
      (FORMAT PARQUET, COMPRESSION {compression.upper()},
       PARTITION_BY (outcode2, lodgement_year));
    """)
    # Build slim link_keys for joins
    con.execute(f"""
      COPY (
        SELECT LMK_KEY, postcode_norm, addr_key, lodgement_year, outcode2
        FROM read_parquet('{_esc(str(cert_out))}/outcode2=*/lodgement_year=*/*.parquet')
      )
      TO '{_esc(str(link_out / "epc_cert_link_keys.parquet"))}'
      (FORMAT PARQUET, COMPRESSION {compression.upper()});
    """)

    # ---------- Recommendations -> join to link_keys, then partition ----------
    # detect optional postcode column in recs
    if recs_postcode_col is None:
        cols = [c[0] for c in con.execute(
            f"SELECT * FROM read_parquet('{_esc(recs_in)}') LIMIT 0"
        ).description]
        if "LMK_KEY_POSTCODE" in cols:
            recs_postcode_col = "LMK_KEY_POSTCODE"
        elif "POSTCODE" in cols:
            recs_postcode_col = "POSTCODE"

    postcode_coalesce = (
        f"COALESCE(k.postcode_norm, replace(upper(CAST(r.{recs_postcode_col} AS VARCHAR)),' ',''))"
        if recs_postcode_col else "k.postcode_norm"
    )

    link_path = str(link_out / "epc_cert_link_keys.parquet")

    if not chunked:
        # Single-pass (should work now that we join against tiny link_keys)
        recs_sql = f"""
        WITH r AS (SELECT * FROM read_parquet('{_esc(recs_in)}')),
        k AS (SELECT * FROM read_parquet('{_esc(link_path)}'))
        SELECT
          CAST(r.LMK_KEY AS VARCHAR) AS LMK_KEY,
          {postcode_coalesce}        AS postcode_norm,
          COALESCE(k.outcode2, CASE WHEN length({postcode_coalesce}) >= 2 THEN substr({postcode_coalesce},1,2) END) AS outcode2,
          k.lodgement_year,
          *
        FROM r
        LEFT JOIN k ON r.LMK_KEY = k.LMK_KEY
        """
        con.execute(f"""
          COPY ({recs_sql})
          TO '{_esc(str(rec_out))}'
          (FORMAT PARQUET, COMPRESSION {compression.upper()},
           PARTITION_BY (outcode2, lodgement_year));
        """)
    else:
        # Ultra-low-memory: process by outcode2 buckets
        oc2s = [row[0] for row in con.execute(
            f"SELECT DISTINCT outcode2 FROM read_parquet('{_esc(link_path)}') WHERE outcode2 IS NOT NULL ORDER BY outcode2"
        ).fetchall()]
        for oc2 in oc2s:
            recs_sql = f"""
            WITH r AS (SELECT * FROM read_parquet('{_esc(recs_in)}')),
            k AS (SELECT * FROM read_parquet('{_esc(link_path)}') WHERE outcode2='{_esc(oc2)}')
            SELECT
              CAST(r.LMK_KEY AS VARCHAR) AS LMK_KEY,
              {postcode_coalesce}        AS postcode_norm,
              '{_esc(oc2)}'              AS outcode2,
              k.lodgement_year,
              *
            FROM r
            LEFT JOIN k ON r.LMK_KEY = k.LMK_KEY
            """
            con.execute(f"""
              COPY ({recs_sql})
              TO '{_esc(str(rec_out))}'
              (FORMAT PARQUET, COMPRESSION {compression.upper()},
               PARTITION_BY (outcode2, lodgement_year));
            """)

    con.close()
    print("[DONE] EPC optimized without blowing memory.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optimize EPC Parquet for search & PPD-ready linking (DuckDB, low-memory)")
    ap.add_argument("--certs-in", required=True)
    ap.add_argument("--recs-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "uncompressed"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--recs-postcode-col", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--memory-limit", default=None, help="e.g., 4GB or 80%")
    ap.add_argument("--temp-dir", default=None, help="Enable spilling, e.g., /tmp/duckdb_spill")
    ap.add_argument("--chunked", action="store_true",
                    help="Process recommendations per outcode2 bucket (slowest, lowest memory)")
    args = ap.parse_args()
    main(args.certs_in, args.recs_in, args.out, args.compression, args.threads,
         args.recs_postcode_col, args.overwrite, args.memory_limit, args.temp_dir, args.chunked)
