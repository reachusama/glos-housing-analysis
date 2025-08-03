# app.py
# Streamlit portfolio app for UK housing data (glas_wrangled.csv)
# ---------------------------------------------------------------
# Features
# - Global sidebar filters (date/area/type/price/tenure etc.)
# - KPIs (median/mean price, volume, mix)
# - Time series (monthly trend, rolling median, seasonal heatmap)
# - Distributions (histogram, box/violin)
# - Area rankings (median, YoY, volume)
# - Outliers & data quality
# - Optional map if postcode centroids are provided
# - Simple hedonic regression (log-price) with effects chart
# - Repeat-sales returns if an address transacts more than once
#
# How to run
#   streamlit run app.py
# Place glas_wrangled.csv in the same folder, or use the file uploader.

import io
import math
import textwrap
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="Gloucestershire Housing – Portfolio Dashboard", layout="wide")


# -------------------------
# Canonical schema + normaliser
# -------------------------

# Canonical HMLR column order/names your app expects
PPD_CANONICAL_COLUMNS = [
    "Transaction ID","Price","Date of Transfer","Postcode","Property Type","Old/New","Duration",
    "PAON","SAON","Street","Locality","Town/City","District","County","PPD Category Type","Record Status"
]

# Common header variants seen in Price Paid downloads (past & present)
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

# Defaults for columns that might be missing
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
    """
    Map incoming headers to canonical names.
    If a file has no header (or nonsense), return None to trigger manual assignment.
    """
    norm = []
    hits = 0
    for c in cols:
        if c is None:
            norm.append(None)
            continue
        key = str(c).strip().lower()
        mapped = PPD_HEADER_ALIASES.get(key)
        if mapped:
            hits += 1
        norm.append(mapped or c)
    # Heuristic: if we mapped at least 4 typical columns, accept; else return None
    return norm if hits >= 4 else None

def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all canonical columns exist, correctly named.
    Add missing columns with defaults.
    """
    # Rename headers using aliases where possible
    renamed = {}
    for c in df.columns:
        key = str(c).strip().lower()
        renamed[c] = PPD_HEADER_ALIASES.get(key, c)
    df = df.rename(columns=renamed)

    # Create any missing canonical columns
    for c in PPD_CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = PPD_DEFAULTS[c]

    # Reorder to canonical
    df = df[PPD_CANONICAL_COLUMNS]

    # Type fixes
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    # date parsing may already happen in _coerce_datetime, but do a first pass here
    df["Date of Transfer"] = pd.to_datetime(df["Date of Transfer"], errors="coerce")

    return df


# -------------------------
# Helpers
# -------------------------

def _coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
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


def _standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    # Trim/upper some keys used for joins/grouping
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
            df[col] = df[col].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    return df


def _addr_id(df: pd.DataFrame) -> pd.Series:
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
    """
    Expand one-letter codes to their full names.
    - Keeps existing full strings unchanged.
    - Handles NaN and mixed case inputs.
    """
    if series is None:
        return series
    s = series.astype("string")
    # Only replace exact single-letter codes that exist in the mapping
    return s.str.upper().map(mapping).fillna(s)


def _expand_hmlr_columns(df, inplace=True):
    """
    Convert code columns in an HM Land Registry dataframe to their full names.

    Columns handled (if present):
      - 'Property Type' (D/S/T/F/O)
      - 'Old/New' (Y/N)
      - 'Duration' (F/L)
      - 'PPD Category Type' (A/B)
      - 'Record Status' (A/C/D)

    Parameters
    ----------
    df : pandas.DataFrame
    inplace : bool, default True
        If True, modify df in place and return df.
        If False, return a copy with expanded columns.

    Returns
    -------
    pandas.DataFrame
    """
    # --- dictionaries from HM Land Registry price-paid data ---
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
        target["Property Type"] = _expand_codes(target["Property Type"], PROPERTY_TYPE_MAP)

    if "Old/New" in target:
        target["Old/New"] = _expand_codes(target["Old/New"], OLD_NEW_MAP)

    if "Duration" in target:
        target["Duration"] = _expand_codes(target["Duration"], DURATION_MAP)

    if "PPD Category Type" in target:
        target["PPD Category Type"] = _expand_codes(target["PPD Category Type"], PPD_CATEGORY_MAP)

    if "Record Status" in target:
        target["Record Status"] = _expand_codes(target["Record Status"], RECORD_STATUS_MAP)

    return target


@st.cache_data(show_spinner=False)
def load_data(uploaded: Optional[bytes]) -> pd.DataFrame:
    # Step 1: read raw CSV safely
    if uploaded is not None:
        raw = io.BytesIO(uploaded)
    else:
        raw = "data/gloucestershire.csv"

    # Try reading with header row first
    try:
        df = pd.read_csv(raw, dtype=str)  # read as strings first; coerce later
        # Attempt to normalise headers
        norm = _normalise_headers(df.columns)
        if norm is not None:
            df.columns = norm
        else:
            # If headers look wrong, re-read without header and assign canonical names
            raw.seek(0) if hasattr(raw, "seek") else None
            df = pd.read_csv(raw, header=None)
            # If column count matches canonical, assign; else pad/truncate
            if df.shape[1] >= len(PPD_CANONICAL_COLUMNS):
                df = df.iloc[:, :len(PPD_CANONICAL_COLUMNS)]
            else:
                # pad missing columns
                for _ in range(len(PPD_CANONICAL_COLUMNS) - df.shape[1]):
                    df[df.shape[1]] = np.nan
            df.columns = PPD_CANONICAL_COLUMNS
    except Exception:
        # Fallback: try with no header directly
        raw = io.BytesIO(uploaded) if uploaded is not None else "data/gloucestershire.csv"
        df = pd.read_csv(raw, header=None)
        if df.shape[1] >= len(PPD_CANONICAL_COLUMNS):
            df = df.iloc[:, :len(PPD_CANONICAL_COLUMNS)]
        else:
            for _ in range(len(PPD_CANONICAL_COLUMNS) - df.shape[1]):
                df[df.shape[1]] = np.nan
        df.columns = PPD_CANONICAL_COLUMNS

    # Step 2: enforce schema + defaults
    df = _coerce_schema(df)
    df = _coerce_datetime(df)
    df = _standardise_strings(df)
    df = _expand_hmlr_columns(df)

    # price numeric
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    # outward code / sector for coarse geography
    if "Postcode" in df:
        pp = df["Postcode"].astype(str)
        df["Outward"] = pp.str.extract(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)", expand=False)
        df["Sector"] = pp.str.extract(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d)", expand=True)[1]
        df["Postcode Sector"] = (
                df["Outward"].fillna("")
                + " "
                + df["Sector"].fillna("")
        ).str.strip()
    # address id
    df["ADDRESS_ID"] = _addr_id(df)
    return df.dropna(subset=["Price", "Date of Transfer"]).copy()


@st.cache_data(show_spinner=False)
def winsorise(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def kpi_card(label: str, value, delta: Optional[str] = None, help: Optional[str] = None):
    c = st.container()
    with c:
        st.metric(label, value, delta=delta, help=help)


# -------------------------
# Sidebar: Data, Navigation & Filters
# -------------------------

st.title("UK Housing Data Trends Analysis")
st.markdown(
    "By [Usama Shahid - LinkedIn](https://www.linkedin.com/in/reach-usama/) \n"
)

st.info(
    "Use this dashboard to understand housing trends, compare areas, and inspect streets/postcodes.\n"
    "Use the filters on the left to focus on a time period, geography, and property characteristics.\n\n"
    "Data Source: [UK Government Price Paid Data]"
    "(https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads). "
)

st.warning("Current data filtered to show housing trends for **Gloucestershire**. Feel free to download any file from uk price paid housing data, and upload via sidebar for analysis.")

with st.sidebar:
    st.header("Data")
    file = st.file_uploader("Upload dataset (optional)", type=["csv"])  # noqa: E231
    df = load_data(file.read() if file else None)

    st.caption(
        f"Loaded **{len(df):,}** rows • {df['Date of Transfer'].min().date()} → {df['Date of Transfer'].max().date()}")

    st.header("Sections")
    nav = st.sidebar.radio(
        "Please select a section for relevant analysis.",
        [
            "KPIs",
            "Trends & Seasonality",
            "Distributions",
            "Area Rankings",
            "Street / Postcode detail",
            "Outliers & Data quality",
            "Hedonic model",
            "Repeat-sales",
            "Map (optional)",
        ],
    )

    st.header("Global filters")
    # Date range
    min_date, max_date = df["Date of Transfer"].min(), df["Date of Transfer"].max()
    date_range = st.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    # Area filters
    county = st.multiselect("County", sorted(df["County"].dropna().unique().tolist()))
    district = st.multiselect("District", sorted(df["District"].dropna().unique().tolist()))
    town = st.multiselect("Town/City", sorted(df["Town/City"].dropna().unique().tolist()))
    outward = st.multiselect("Outward code (e.g., GL52)", sorted(df["Outward"].dropna().unique().tolist()))

    # Property/tenure
    prop_types = st.multiselect("Property Type", sorted(df["Property Type"].dropna().unique().tolist()))
    old_new = st.multiselect("Old/New", sorted(df["Old/New"].dropna().unique().tolist()))
    duration = st.multiselect("Duration", sorted(df["Duration"].dropna().unique().tolist()))

    # Price range & options
    p_min, p_max = float(df["Price"].min()), float(df["Price"].max())
    price_range = st.slider(
        "Price range",
        min_value=float(np.floor(p_min)),
        max_value=float(np.ceil(p_max)),
        value=(float(np.floor(p_min)), float(np.ceil(p_max))),
        step=1000.0,
    )
    use_winsor = st.checkbox("Winsorise prices (1–99%)", value=False)

    show_raw_clean = st.selectbox("Data view", ["Cleaned only", "Raw + cleaned"])

# Apply filters
mask = (
        (df["Date of Transfer"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
        & (df["Price"].between(price_range[0], price_range[1]))
)
if county:
    mask &= df["County"].isin(county)
if district:
    mask &= df["District"].isin(district)
if town:
    mask &= df["Town/City"].isin(town)
if outward:
    mask &= df["Outward"].isin(outward)
if prop_types:
    mask &= df["Property Type"].isin(prop_types)
if old_new:
    mask &= df["Old/New"].isin(old_new)
if duration:
    mask &= df["Duration"].isin(duration)

base = df.loc[mask].copy()

if use_winsor:
    base["Price_w"] = winsorise(base["Price"])
    price_col = "Price_w"
else:
    price_col = "Price"

# -------------------------
# 1) KPIs
# -------------------------
if nav == "KPIs":
    st.subheader("Key figures")

    # YoY comparison for current month range vs same months prior year
    latest_year = base["Year"].max()
    prev_year = latest_year - 1

    cur = base[base["Year"] == latest_year]
    prev = base[base["Year"] == prev_year]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        kpi_card("Median price", f"£{base[price_col].median():,.0f}")
    with col2:
        kpi_card("Mean price", f"£{base[price_col].mean():,.0f}")
    with col3:
        kpi_card("Transactions", f"{len(base):,}")
    with col4:
        yoy = (
            cur[price_col].median() / prev[price_col].median() - 1.0
            if len(cur) > 0 and len(prev) > 0 and prev[price_col].median() > 0
            else np.nan
        )
        kpi_card("YoY median", f"{yoy * 100:,.1f}%" if pd.notna(yoy) else "–")
    with col5:
        mix = (
            base.groupby("Property Type")[price_col].median().sort_values(ascending=False)
            if len(base) else pd.Series(dtype=float)
        )
        top_type = mix.index[0] if len(mix) else "–"
        kpi_card("Top median type", top_type)

    # Median price by type
    st.markdown("**Median price by property type**")
    g = base.groupby("Property Type")[price_col].median().reset_index().sort_values(price_col, ascending=False)
    fig = px.bar(g, x="Property Type", y=price_col, text_auto=True, labels={price_col: "Median price"})
    st.plotly_chart(fig, use_container_width=True)

    # Tenure and vintage mix
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Freehold vs Leasehold**")
        g = base["Duration"].value_counts(dropna=False).rename_axis("Duration").reset_index(name="count")
        fig = px.pie(g, names="Duration", values="count")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.markdown("**Old vs New**")
        g = base["Old/New"].value_counts(dropna=False).rename_axis("Old/New").reset_index(name="count")
        fig = px.pie(g, names="Old/New", values="count")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 2) Trends & Seasonality
# -------------------------
if nav == "Trends & Seasonality":
    st.subheader("Monthly trend")
    metric = st.selectbox("Metric", ["Median price", "Mean price", "Transactions"])
    window = st.slider("Rolling window (months)", 1, 24, 6)

    m = base.set_index("Date of Transfer").sort_index()
    monthly = (
        m.resample("MS")[price_col].median().to_frame("Median price")
        .join(m.resample("MS")[price_col].mean().to_frame("Mean price"))
        .join(m.resample("MS")[price_col].size().to_frame("Transactions"))
    )

    if len(monthly) == 0:
        st.info("No data for current filters.")
    else:
        y = monthly[metric].rolling(window).median()
        fig = px.line(monthly.reset_index(), x="Date of Transfer", y=metric)
        fig.add_trace(go.Scatter(x=monthly.index, y=y, name=f"{window}m rolling"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonal heatmap (Year × Month)")
    m = base.copy()
    pivot = (
        m.groupby(["Year", "Month"])[price_col].median().reset_index()
        .pivot(index="Year", columns="Month", values=price_col).sort_index()
    )
    fig = px.imshow(pivot, aspect="auto", labels=dict(color="Median price"))
    fig.update_xaxes(title_text="Month")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 3) Distributions
# -------------------------
if nav == "Distributions":
    st.subheader("Price distribution")
    log_scale = st.checkbox("Log-price", value=True)
    bins = st.slider("Bins", 10, 200, 60)
    s = np.log(base[price_col]) if log_scale else base[price_col]
    fig = px.histogram(x=s, nbins=bins, labels={"x": "log(Price)" if log_scale else "Price"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Box/violin by property type")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(base, x="Property Type", y=price_col, points="suspectedoutliers")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.violin(base, x="Property Type", y=price_col, box=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 4) Area Rankings
# -------------------------
if nav == "Area Rankings":
    st.subheader("Rankings by area")
    level = st.selectbox("Group by", ["District", "Town/City", "Outward", "Postcode Sector"])
    g = base.groupby([level, "Year"]).agg(median_price=(price_col, "median"), volume=(price_col, "size")).reset_index()
    latest = g[g["Year"] == g["Year"].max()].copy()
    prev = g[g["Year"] == g["Year"].max() - 1][[level, "median_price"]].rename(columns={"median_price": "median_prev"})
    latest = latest.merge(prev, on=level, how="left")
    latest["YoY %"] = (latest["median_price"] / latest["median_prev"] - 1.0) * 100

    st.dataframe(latest.sort_values("median_price", ascending=False), use_container_width=True)

    st.markdown("**Top movers (YoY median %)**")
    movers = latest.dropna(subset=["YoY %"]).sort_values("YoY %", ascending=False).head(20)
    fig = px.bar(movers, x=level, y="YoY %")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 5) Street / Postcode detail
# -------------------------
if nav == "Street / Postcode detail":
    st.subheader("Lookup")
    area = st.text_input("Street or Postcode contains… (case-insensitive)")
    q = base
    if area:
        area_upper = area.strip().upper()
        q = q[
            q["Street"].str.contains(area_upper, na=False)
            | q["Postcode"].str.contains(area_upper, na=False)
            ]
    st.caption(f"Matches: {len(q):,}")

    if len(q) > 0:
        # timeline
        t = q.sort_values("Date of Transfer")
        fig = px.line(t, x="Date of Transfer", y=price_col, markers=True,
                      hover_data=["Street", "Postcode", "Property Type"])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(t[["Date of Transfer", "Price", "Property Type", "Duration", "Old/New", "PAON", "SAON", "Street",
                        "Postcode", "District", "Town/City"]].sort_values("Date of Transfer", ascending=False),
                     use_container_width=True)
    else:
        st.info("Type a street name or postcode fragment to see a detailed view.")

# -------------------------
# 6) Outliers & Data quality
# -------------------------
if nav == "Outliers & Data quality":
    st.subheader("Outliers")
    lo, hi = base[price_col].quantile([0.01, 0.99])
    out = base[(base[price_col] < lo) | (base[price_col] > hi)].copy()
    st.caption(f"Flagged outside 1–99th pct: {len(out):,}")
    st.dataframe(
        out[["Date of Transfer", "Price", "Property Type", "Street", "Postcode", "District", "Town/City"]].sort_values(
            "Price", ascending=False), use_container_width=True)

    st.subheader("Missingness heatmap")
    miss = base.isna().mean().sort_values(ascending=False)
    miss_df = miss.reset_index().rename(columns={"index": "Column", 0: "Missing %"})
    fig = px.bar(miss_df, x="Column", y="Missing %")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Entry types")
    if "PPD Category Type" in base and "Record Status" in base:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(base, names="PPD Category Type")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(base, names="Record Status")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 7) Hedonic model
# -------------------------
if nav == "Hedonic model":
    st.subheader("Log-price hedonic regression")
    st.caption("Model: log(Price) ~ Property Type + Old/New + Duration + Year + Month + Outward")

    model_df = base.dropna(subset=["Price", "Property Type", "Old/New", "Duration", "Year", "Month"]).copy()
    model_df["log_price"] = np.log(model_df[price_col])

    formula = "log_price ~ C(`Property Type`) + C(`Old/New`) + C(Duration) + C(Year) + C(Month) + C(Outward)"
    st.info("Work in progerss ...")
    # try:
    #     mod = smf.ols(formula=formula, data=model_df).fit()
    #     st.text(mod.summary().as_text()[:4000])
    #
    #     # Extract categorical effects for display (Property Type and Old/New)
    #     coefs = mod.params.filter(like="C(`Property Type`)").append(mod.params.filter(like="C(`Old/New`)"))
    #     ef = coefs.reset_index()
    #     ef.columns = ["Term", "Coefficient"]
    #     ef["Premium %"] = (np.exp(ef["Coefficient"]) - 1.0) * 100
    #     fig = px.bar(ef.sort_values("Premium %", ascending=False), x="Term", y="Premium %")
    #     st.markdown("**Feature effects (exp(coef)−1)**")
    #     st.plotly_chart(fig, use_container_width=True)
    # except Exception as e:
    #     st.warning(f"Model could not be fitted: {e}")

# -------------------------
# 8) Repeat-sales
# -------------------------
if nav == "Repeat-sales":
    st.subheader("Repeat-sales returns (same address)")
    g = base.sort_values("Date of Transfer").copy()
    g["ADDRESS_ID"] = _addr_id(g)
    counts = g["ADDRESS_ID"].value_counts()
    multi = g[g["ADDRESS_ID"].isin(counts[counts > 1].index)].copy()

    if len(multi) == 0:
        st.info("No repeated addresses found with current filters.")
    else:
        # Compute sequential returns per address
        multi["next_price"] = multi.groupby("ADDRESS_ID")[price_col].shift(-1)
        multi["next_date"] = multi.groupby("ADDRESS_ID")["Date of Transfer"].shift(-1)
        rs = multi.dropna(subset=["next_price", "next_date"]).copy()
        rs["ret %"] = (rs["next_price"] / rs[price_col] - 1.0) * 100
        rs["months"] = (rs["next_date"] - rs["Date of Transfer"]).dt.days / 30.44
        st.caption(f"Pairs: {len(rs):,}")
        fig = px.scatter(rs, x="months", y="ret %", hover_data=["Street", "Postcode", "Property Type"], trendline="ols")
        fig.update_xaxes(title="Months between sales")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rs[["Date of Transfer", "next_date", price_col, "next_price", "ret %", "months", "Street",
                         "Postcode", "Property Type"]].sort_values("ret %", ascending=False), use_container_width=True)

# -------------------------
# 9) Map (optional)
# -------------------------
if nav == "Map (optional)":
    st.subheader("Map of transactions (needs postcode centroids)")
    st.write(
        "Upload a CSV with postcode sector centroids having columns: `Postcode Sector` (e.g., 'GL52 3'), `lat`, `lon`."
    )
    cent = st.file_uploader("Upload centroids CSV", type=["csv"], key="centroids")
    if cent is None:
        st.info("No centroid file uploaded. Showing area bar chart instead.")
        agg = base.groupby("Outward")[price_col].median().reset_index().sort_values(price_col, ascending=False).head(30)
        fig = px.bar(agg, x="Outward", y=price_col, labels={price_col: "Median price"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        cent_df = pd.read_csv(cent)
        merged = (
            base.groupby("Postcode Sector").agg(median_price=(price_col, "median"),
                                                volume=(price_col, "size")).reset_index()
            .merge(cent_df, on="Postcode Sector", how="inner")
        )
        fig = px.scatter_mapbox(
            merged,
            lat="lat",
            lon="lon",
            size="volume",
            color="median_price",
            hover_name="Postcode Sector",
            hover_data={"median_price": ":,.0f", "volume": True},
            zoom=8,
            height=700,
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption(
    "Built by Usama Shahid using Streamlit, Plotly, and statsmodels. Export tables via the '⋮' menu on each element.")
