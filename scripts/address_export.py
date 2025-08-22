import os
import duckdb

con = duckdb.connect(database=':memory:')

ppd_root = "../data/hf_release/v1/ppd"
epc_root = "../data/hf_release/v1/epc"

con.execute(f"""
    CREATE OR REPLACE VIEW ppd_src AS
    SELECT *
    FROM parquet_scan('{ppd_root}/**/*.parquet', hive_partitioning=1)
""")

con.execute(f"""
    CREATE OR REPLACE VIEW epc_src AS
    SELECT *
    FROM parquet_scan('{epc_root}/**/*.parquet', hive_partitioning=1)
""")

out_dir = "../machine_learning/address_parser/data/raw/ppd_addresses"
os.makedirs(out_dir, exist_ok=True)

if __name__ == '__main__':
    years = [r[0] for r in con.execute("""
        SELECT DISTINCT sale_year
        FROM ppd_src
        ORDER BY sale_year
    """).fetchall()]
    print(years)

    for y in years:
        out_path = os.path.join(out_dir, f"ppd_{y}.parquet")
        print(f"Exporting {y} → {out_path}")

        # Escape any single quotes in the path for SQL
        out_path_sql = out_path.replace("'", "''")

        con.execute(f"""
          COPY (
            SELECT
                "Property Type" AS property_type,
                "Postcode"      AS postcode,
                PAON            AS paon,
                SAON            AS saon,
                Street          AS street,
                Locality        AS locality,
                "Town/City"     AS town_city,
                "District"      AS district,
                "County"        AS county
            FROM ppd_src
            WHERE sale_year = {int(y)}
          )
          TO '{out_path_sql}'
          (FORMAT 'parquet', COMPRESSION 'zstd')
        """)
