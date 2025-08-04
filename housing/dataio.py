# housing/dataio.py — loading, schema mapping, caching
import io
import socket
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

hostname = socket.gethostname()
if "local" in hostname:
    DATASET_TO_USE = "pp_2020_2025_combined.csv"
else:
    DATASET_TO_USE = "pp_2024_2025_combined.csv"

PPD_CANONICAL_COLUMNS = [
    "Transaction ID",
    "Price",
    "Date of Transfer",
    "Postcode",
    "Property Type",
    "Old/New",
    "Duration",
    "PAON",
    "SAON",
    "Street",
    "Locality",
    "Town/City",
    "District",
    "County",
    "PPD Category Type",
    "Record Status",
]

PPD_HEADER_ALIASES = {
    # id
    "transaction unique identifier": "Transaction ID",
    "transaction id": "Transaction ID",
    # price
    "price": "Price",
    "amount": "Price",
    # date
    "date of transfer": "Date of Transfer",
    "transfer date": "Date of Transfer",
    "date": "Date of Transfer",
    # postcode
    "postcode": "Postcode",
    "post code": "Postcode",
    # property type
    "property type": "Property Type",
    "property_type": "Property Type",
    # new/old
    "old/new": "Old/New",
    "new build": "Old/New",
    "is new": "Old/New",
    # tenure/duration
    "duration": "Duration",
    "tenure": "Duration",
    # address parts
    "paon": "PAON",
    "primary addressable object name": "PAON",
    "saon": "SAON",
    "secondary addressable object name": "SAON",
    "street": "Street",
    "thoroughfare": "Street",
    "locality": "Locality",
    "town/city": "Town/City",
    "town": "Town/City",
    "city": "Town/City",
    "district": "District",
    "local authority": "District",
    "county": "County",
    # categories/status
    "ppd category type": "PPD Category Type",
    "record status - monthly file only": "Record Status",
    "record status": "Record Status",
}

PPD_DEFAULTS = {
    "Transaction ID": None,
    "Price": np.nan,
    "Date of Transfer": pd.NaT,
    "Postcode": np.nan,
    "Property Type": np.nan,
    "Old/New": np.nan,
    "Duration": np.nan,
    "PAON": np.nan,
    "SAON": np.nan,
    "Street": np.nan,
    "Locality": np.nan,
    "Town/City": np.nan,
    "District": np.nan,
    "County": np.nan,
    "PPD Category Type": np.nan,
    "Record Status": np.nan,
}


def _normalise_headers(cols):
    norm, hits = [], 0
    for c in cols:
        if c is None:
            norm.append(None)
            continue
        key = str(c).strip().lower()
        mapped = PPD_HEADER_ALIASES.get(key)
        if mapped:
            hits += 1
        norm.append(mapped or c)
    return norm if hits >= 4 else None


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: PPD_HEADER_ALIASES.get(str(c).strip().lower(), c) for c in df.columns}
    df = df.rename(columns=renamed)
    for c in PPD_CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = PPD_DEFAULTS[c]
    df = df[PPD_CANONICAL_COLUMNS]
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Date of Transfer"] = pd.to_datetime(df["Date of Transfer"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_data(uploaded: Optional[bytes]) -> pd.DataFrame:
    if uploaded is not None:
        raw = io.BytesIO(uploaded)
    else:
        raw = f"data/{DATASET_TO_USE}"
    try:
        df = pd.read_csv(raw, dtype=str)
        norm = _normalise_headers(df.columns)
        if norm is not None:
            df.columns = norm
        else:
            raw.seek(0) if hasattr(raw, "seek") else None
            df = pd.read_csv(raw, header=None)
            if df.shape[1] >= len(PPD_CANONICAL_COLUMNS):
                df = df.iloc[:, : len(PPD_CANONICAL_COLUMNS)]
            else:
                for _ in range(len(PPD_CANONICAL_COLUMNS) - df.shape[1]):
                    df[df.shape[1]] = np.nan
            df.columns = PPD_CANONICAL_COLUMNS
    except Exception:
        raw = io.BytesIO(uploaded) if uploaded is not None else f"data/{DATASET_TO_USE}"
        df = pd.read_csv(raw, header=None)
        if df.shape[1] >= len(PPD_CANONICAL_COLUMNS):
            df = df.iloc[:, : len(PPD_CANONICAL_COLUMNS)]
        else:
            for _ in range(len(PPD_CANONICAL_COLUMNS) - df.shape[1]):
                df[df.shape[1]] = np.nan
        df.columns = PPD_CANONICAL_COLUMNS

    from .cleaning import addr_id  # lazy import to avoid cycles
    from .cleaning import coerce_datetime, expand_hmlr_columns, standardise_strings

    df = _coerce_schema(df)
    df = coerce_datetime(df)
    df = standardise_strings(df)
    df = expand_hmlr_columns(df)

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    if "Postcode" in df:
        pp = df["Postcode"].astype(str)
        df["Outward"] = pp.str.extract(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)", expand=False)
        df["Sector"] = pp.str.extract(
            r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d)", expand=True
        )[1]
        df["Postcode Sector"] = (
            df["Outward"].fillna("") + " " + df["Sector"].fillna("")
        ).str.strip()
    df["ADDRESS_ID"] = addr_id(df)
    return df.dropna(subset=["Price", "Date of Transfer"]).copy()
