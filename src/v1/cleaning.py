# cleaning.py — data cleaning + helpers
import numpy as np
import pandas as pd
import streamlit as st

__all__ = [
    "coerce_datetime",
    "standardise_strings",
    "addr_id",
    "expand_hmlr_columns",
    "winsorise",
]


def coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df["Date of Transfer"].dtype, np.datetime64):
        df["Date of Transfer"] = pd.to_datetime(df["Date of Transfer"], errors="coerce")
    if "Year" not in df:
        df["Year"] = df["Date of Transfer"].dt.year
    if "Month" not in df:
        df["Month"] = df["Date of Transfer"].dt.month
    if "Quarter" not in df:
        df["Quarter"] = df["Date of Transfer"].dt.quarter
    if "Month Name" not in df:
        df["Month Name"] = df["Date of Transfer"].dt.month_name()
    if "Day" not in df:
        df["Day"] = df["Date of Transfer"].dt.day
    return df


def standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "Postcode",
        "Street",
        "Locality",
        "Town/City",
        "District",
        "County",
        "PAON",
        "SAON",
    ]:
        if col in df:
            df[col] = (
                df[col].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
            )
    return df


def addr_id(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["PAON", "SAON", "Street", "Postcode"]:
        parts.append(df.get(col, pd.Series(index=df.index, dtype=object)).fillna(""))
    addr = (
        parts[0].astype(str)
        + "|"
        + parts[1].astype(str)
        + "|"
        + parts[2].astype(str)
        + "|"
        + parts[3].astype(str)
    )
    return addr.str.replace(r"\s+", " ", regex=True).str.strip()


def _expand_codes(series, mapping):
    if series is None:
        return series
    s = series.astype("string")
    return s.str.upper().map(mapping).fillna(s)


def expand_hmlr_columns(df, inplace=True):
    PROPERTY_TYPE_MAP = {
        "D": "Detached",
        "S": "Semi-Detached",
        "T": "Terraced",
        "F": "Flat/Maisonette",
        "O": "Other",
    }
    OLD_NEW_MAP = {"Y": "New", "N": "Old"}
    DURATION_MAP = {"F": "Freehold", "L": "Leasehold"}
    PPD_CATEGORY_MAP = {
        "A": "Standard price paid entry",
        "B": "Additional price paid entry",
    }
    RECORD_STATUS_MAP = {"A": "Addition", "C": "Change", "D": "Delete"}

    target = df if inplace else df.copy()
    if "Property Type" in target:
        target["Property Type"] = _expand_codes(
            target["Property Type"], PROPERTY_TYPE_MAP
        )
    if "Old/New" in target:
        target["Old/New"] = _expand_codes(target["Old/New"], OLD_NEW_MAP)
    if "Duration" in target:
        target["Duration"] = _expand_codes(target["Duration"], DURATION_MAP)
    if "PPD Category Type" in target:
        target["PPD Category Type"] = _expand_codes(
            target["PPD Category Type"], PPD_CATEGORY_MAP
        )
    if "Record Status" in target:
        target["Record Status"] = _expand_codes(
            target["Record Status"], RECORD_STATUS_MAP
        )
    return target


@st.cache_data(show_spinner=False)
def winsorise(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)
