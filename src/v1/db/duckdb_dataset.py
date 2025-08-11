#!/usr/bin/env python3
"""
DuckDBDataset: quick schema scan + scalable queries over Parquet folders.

- Works with wildcard patterns: '/path/to/data/**/*.parquet'
- Understands Hive partitions (set hive_partitioning=True)
- Returns pandas or pyarrow DataFrames
- Low memory settings by default (spilling + memory cap)

pip install duckdb pandas pyarrow
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple, Any, Union
import re
import duckdb

try:
    import pandas as pd
except Exception:
    pd = None
try:
    import pyarrow as pa  # type: ignore
except Exception:
    pa = None


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier safely (double quotes)."""
    if not isinstance(name, str):
        raise TypeError("identifier must be a string")
    # minimal validation; allow letters, numbers, underscore, slash, equals (for partitions), star, dash
    return '"' + name.replace('"', '""') + '"'


def _escape_str(val: str) -> str:
    return val.replace("'", "''")


@dataclass
class DuckDBConfig:
    threads: int = 1
    memory_limit: str = "4GB"  # or "80%"
    temp_directory: Optional[str] = "/tmp/duckdb_spill"
    preserve_insertion_order: bool = False


class DuckDBDataset:
    """
    Wraps a DuckDB connection and a Parquet dataset (folder or glob pattern).

    Example:
        ds = DuckDBDataset("../data/epc_optimized/certificates", hive_partitioning=True)
        print(ds.schema())                       # list columns and types
        print(ds.partition_summary())            # show partition columns + values
        df = ds.select(
            columns=["LMK_KEY","postcode_norm","outcode2","lodgement_year"],
            where={"outcode2": "GL", "lodgement_year__gte": "2018"},
            limit=20,
        )
    """

    def __init__(
            self,
            path_or_glob: Union[str, Path, Iterable[Union[str, Path]]],
            *,
            hive_partitioning: bool = True,
            config: Optional[DuckDBConfig] = None,
            dataset_name: Optional[str] = None,
    ):
        self.paths: List[str] = self._normalize_paths(path_or_glob)
        if not self.paths:
            raise FileNotFoundError("No matching files/dirs for the given path/glob.")
        self.hive_partitioning = hive_partitioning
        self.config = config or DuckDBConfig()
        self.name = dataset_name or "dataset"

        # One connection for the object
        self.con = duckdb.connect()
        self._apply_pragmas()

        # Register a view that unions all matching files/dirs
        self.view_name = f"v_{re.sub(r'[^A-Za-z0-9_]', '_', self.name)}"
        self._register_view()

    # --------------- setup ---------------

    def _normalize_paths(self, path_or_glob) -> List[str]:
        if isinstance(path_or_glob, (str, Path)):
            candidates = [str(path_or_glob)]
        else:
            candidates = [str(p) for p in path_or_glob]

        files: List[str] = []
        for pat in candidates:
            p = Path(pat)
            # If it's a directory, read all parquet under it recursively
            if p.exists() and p.is_dir():
                files.extend([str(x) for x in p.rglob("*.parquet")])
            else:
                # Treat as glob pattern
                files.extend([str(x) for x in Path().glob(pat)])
        # Deduplicate, keep order
        seen = set()
        out = []
        for f in files:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def _apply_pragmas(self):
        c = self.config
        if not c.preserve_insertion_order:
            self.con.execute("PRAGMA preserve_insertion_order=false;")
        if c.threads and c.threads > 0:
            self.con.execute(f"PRAGMA threads={c.threads};")
        if c.memory_limit:
            self.con.execute(f"PRAGMA memory_limit='{_escape_str(c.memory_limit)}';")
        if c.temp_directory:
            Path(c.temp_directory).mkdir(parents=True, exist_ok=True)
            self.con.execute(f"PRAGMA temp_directory='{_escape_str(c.temp_directory)}';")

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    # --------------- core helpers ---------------

    def _read_parquet_sql(
            self,
            columns: Optional[List[str]] = None,
            where_sql: Optional[str] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            filename: bool = False,
    ) -> str:
        """
        Builds a SELECT over read_parquet that unions all paths.
        We pass a list of files to keep expansion explicit.
        """
        files_sql = ", ".join([f"'{_escape_str(p)}'" for p in self.paths])
        options = []
        if self.hive_partitioning:
            options.append("hive_partitioning=1")
        if filename:
            options.append("filename=1")

        opts = ""
        if options:
            opts = ", " + ", ".join(options)

        # columns projection (string list)
        col_sql = "*"
        if columns:
            col_sql = ", ".join(_quote_ident(c) for c in columns)

        base = f"SELECT {col_sql} FROM read_parquet([{files_sql}]{opts})"

        if where_sql:
            base += f" WHERE {where_sql}"
        if order_by:
            base += f" ORDER BY {order_by}"
        if limit and limit > 0:
            base += f" LIMIT {int(limit)}"
        return base

    def _build_where(self, where: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Turn a dict like:
          {"outcode2": "GL", "lodgement_year__gte": "2018", "price__lt": 1_000_000}
        into SQL that pushes down cleanly.
        Supported suffixes: __eq (default), __ne, __lt, __lte, __gt, __gte, __like, __in
        """
        if not where:
            return None
        clauses = []
        for key, val in where.items():
            if "__" in key:
                col, op = key.split("__", 1)
            else:
                col, op = key, "eq"
            col_q = _quote_ident(col)

            if op == "in":
                if not isinstance(val, (list, tuple, set)):
                    raise ValueError(f"{key} expects an iterable")
                vals = ", ".join([f"'{_escape_str(str(v))}'" for v in val])
                clauses.append(f"{col_q} IN ({vals})")
            elif op == "like":
                clauses.append(f"{col_q} LIKE '{_escape_str(str(val))}'")
            else:
                # scalar compare
                if isinstance(val, (int, float)):
                    v_sql = str(val)
                else:
                    v_sql = f"'{_escape_str(str(val))}'"

                mapping = {
                    "eq": "=",
                    "ne": "!=",
                    "lt": "<",
                    "lte": "<=",
                    "gt": ">",
                    "gte": ">=",
                }
                if op not in mapping:
                    raise ValueError(f"Unsupported operator: {op}")
                clauses.append(f"{col_q} {mapping[op]} {v_sql}")

        return " AND ".join(clauses)

    # --------------- public API ---------------

    def schema(self) -> List[Dict[str, str]]:
        """
        Return a [{name, type}] list without scanning data.
        """
        sql = self._read_parquet_sql(limit=0)
        cur = self.con.execute(sql)
        # duckdb .description -> [(name, type_code, ...)]
        cols = []
        for name, *rest in cur.description:
            # detect type with a DESCRIBE SELECT round-trip
            cols.append({"name": name, "type": "UNKNOWN"})
        # Better: DESCRIBE the relation to get DuckDB types
        dtypes = self.con.execute(f"DESCRIBE {sql}").fetchall()
        # dtypes cols: ['column_name','column_type',...]
        # protect against version differences
        for i, row in enumerate(dtypes):
            try:
                cols[i]["type"] = row[1]
            except Exception:
                pass
        return cols

    def partition_summary(self, sample_limit_per_key: int = 50) -> Dict[str, List[str]]:
        """
        Inspect Hive partition columns and give a few example values.
        """
        # When hive_partitioning=1, partition columns appear as real columns
        sql = self._read_parquet_sql(limit=0)
        cols = [row[0] for row in self.con.execute(f"DESCRIBE {sql}").fetchall()]
        # Heuristic: partition columns are commonly small-int/string + named like key=value in paths,
        # but here we just show distinct values for likely partition-like columns.
        # If you know them, call distinct_values(["outcode2","lodgement_year"])
        likely_parts = [c for c in cols if c in ("outcode2", "lodgement_year", "sale_year", "sale_month")]
        out: Dict[str, List[str]] = {}
        for c in likely_parts:
            vals = self.con.execute(
                self._read_parquet_sql(
                    columns=[c],
                    where_sql=None,
                    order_by=f"{_quote_ident(c)}",
                    limit=0,  # no limit here; we'll use GROUP BY with LIMIT
                ).replace(
                    f"SELECT {_quote_ident(c)}",
                    f"SELECT {_quote_ident(c)} AS v"
                ).replace(
                    " LIMIT 0", ""
                )
                + f" GROUP BY v ORDER BY v LIMIT {int(sample_limit_per_key)}"
            ).fetchall()
            out[c] = [str(v[0]) for v in vals]
        return out

    def head(
            self,
            n: int = 5,
            columns: Optional[List[str]] = None,
            where: Optional[Dict[str, Any]] = None,
            to: str = "pandas",
    ):
        """Preview rows with optional projection and filters."""
        where_sql = self._build_where(where)
        sql = self._read_parquet_sql(columns=columns, where_sql=where_sql, limit=n)
        return self._fetch(sql, to=to)

    def select(
            self,
            columns: Optional[List[str]] = None,
            where: Optional[Dict[str, Any]] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            to: str = "pandas",
    ):
        """General purpose SELECT with pushdown."""
        where_sql = self._build_where(where)
        sql = self._read_parquet_sql(
            columns=columns, where_sql=where_sql, order_by=order_by, limit=limit
        )
        return self._fetch(sql, to=to)

    def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        where_sql = self._build_where(where)
        sql = self._read_parquet_sql(columns=None, where_sql=where_sql)
        cnt = self.con.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0]
        return int(cnt)

    def distinct_values(
            self,
            columns: List[str],
            where: Optional[Dict[str, Any]] = None,
            limit: int = 1000,
            to: str = "pandas",
    ):
        """Distinct combos of columns (handy to inspect partition keys)."""
        if not columns:
            raise ValueError("columns must be non-empty")
        where_sql = self._build_where(where)
        base = self._read_parquet_sql(columns=columns, where_sql=where_sql, limit=None)
        sel = f"SELECT {', '.join(_quote_ident(c) for c in columns)} FROM ({base}) GROUP BY {', '.join(_quote_ident(c) for c in columns)}"
        sel += f" LIMIT {int(limit)}"
        return self._fetch(sel, to=to)

    def value_counts(
            self,
            column: str,
            where: Optional[Dict[str, Any]] = None,
            limit: int = 50,
            to: str = "pandas",
    ):
        """Top values with counts (sorted desc)."""
        where_sql = self._build_where(where)
        base = self._read_parquet_sql(columns=[column], where_sql=where_sql)
        q = f"""
        SELECT {_quote_ident(column)} AS value, COUNT(*) AS n
        FROM ({base})
        GROUP BY value
        ORDER BY n DESC
        LIMIT {int(limit)}
        """
        return self._fetch(q, to=to)

    def sql(self, query: str, params: Optional[Tuple[Any, ...]] = None, to: str = "pandas"):
        """Run raw SQL against the dataset view (self.view_name is available)."""
        # We expose a view self.view_name that selects from read_parquet([...])
        if params:
            cur = self.con.execute(query, params)
        else:
            cur = self.con.execute(query)
        return self._convert(cur, to)

    # --------------- view registration ---------------

    def _register_view(self):
        """Create/replace a view over the dataset as self.view_name."""
        files_sql = ", ".join([f"'{_escape_str(p)}'" for p in self.paths])
        opts = "hive_partitioning=1" if self.hive_partitioning else ""
        opts = ", " + opts if opts else ""
        self.con.execute(f"""
            CREATE OR REPLACE VIEW {_quote_ident(self.view_name)} AS
            SELECT * FROM read_parquet([{files_sql}]{opts});
        """)

    # --------------- fetch helpers ---------------

    def _fetch(self, sql: str, to: str = "pandas"):
        cur = self.con.execute(sql)
        return self._convert(cur, to)

    def _convert(self, cur: duckdb.DuckDBPyConnection, to: str):
        to = (to or "pandas").lower()
        if to == "pandas":
            if pd is None:
                raise RuntimeError("pandas is not installed")
            return cur.fetch_df()
        elif to in ("arrow", "pyarrow"):
            if pa is None:
                raise RuntimeError("pyarrow is not installed")
            return cur.fetch_arrow_table()
        else:
            # raw tuples
            return cur.fetchall()


# -------------------------- Convenience factories -------------------------- #

def open_epc_certs(root: Union[str, Path], config: Optional[DuckDBConfig] = None) -> DuckDBDataset:
    """
    Opens an EPC certificates dataset that was written as:
      root/certificates/outcode2=*/lodgement_year=*/*.parquet
    """
    root = Path(root)
    pattern = str(root / "certificates" / "outcode2=*" / "lodgement_year=*" / "*.parquet")
    return DuckDBDataset(pattern, hive_partitioning=True, config=config, dataset_name="epc_certs")


def open_epc_recs(root: Union[str, Path], config: Optional[DuckDBConfig] = None) -> DuckDBDataset:
    pattern = str(Path(root) / "recommendations" / "outcode2=*" / "lodgement_year=*" / "*.parquet")
    return DuckDBDataset(pattern, hive_partitioning=True, config=config, dataset_name="epc_recs")


def open_ppd(root: Union[str, Path], config: Optional[DuckDBConfig] = None) -> DuckDBDataset:
    pattern = str(Path(root) / "sale_year=*" / "outcode2=*" / "*.parquet")
    return DuckDBDataset(pattern, hive_partitioning=True, config=config, dataset_name="ppd")
