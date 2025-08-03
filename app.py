# app.py — Streamlit entrypoint
import numpy as np
import pandas as pd
import streamlit as st

from housing.cleaning import winsorise
from housing.dataio import load_data
from housing.ui.components import page_header, section_help
from housing.ui.sections import (
    detail,
    distributions,
    hedonic,
    kpis,
    mapview,
    quality,
    rankings,
    repeats,
    trends,
)

st.set_page_config(
    page_title="Gloucestershire Housing – Portfolio Dashboard", layout="wide"
)

page_header(
    title="UK Housing Data Trends Analysis",
    author_md="By [Usama Shahid – LinkedIn](https://www.linkedin.com/in/reach-usama/)",
    intro_md=(
        "Use this dashboard to understand housing trends, compare areas, and inspect streets/postcodes.\n\n"
        "Use the filters on the left to set a time period, geography, and property characteristics.\n\n"
        "Data Source: [UK Government Price Paid Data]"
        "(https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)."
    ),
    notice_md=(
        "Current data filtered to show housing trends for **Gloucestershire**. "
        "Upload any UK Price Paid CSV via the sidebar to analyze it here."
    ),
)

# -------------------------
# Sidebar: Data + Filters
# -------------------------
with st.sidebar:
    st.header("Data")
    file = st.file_uploader("Upload dataset (optional)", type=["csv"])  # noqa: E231
    df = load_data(file.read() if file else None)

    st.caption(
        f"Loaded **{len(df):,}** rows • {df['Date of Transfer'].min().date()} → {df['Date of Transfer'].max().date()}"
    )

    st.header("Analysis Sections")
    nav = st.radio(
        "Choose a section:",
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
        help="Charts and tables below will respect this range.",
    )

    # Area filters
    county = st.multiselect("County", sorted(df["County"].dropna().unique().tolist()))
    district = st.multiselect(
        "District", sorted(df["District"].dropna().unique().tolist())
    )
    town = st.multiselect(
        "Town/City", sorted(df["Town/City"].dropna().unique().tolist())
    )
    outward = st.multiselect(
        "Outward code (e.g., GL52)", sorted(df["Outward"].dropna().unique().tolist())
    )

    # Property/tenure
    prop_types = st.multiselect(
        "Property Type", sorted(df["Property Type"].dropna().unique().tolist())
    )
    old_new = st.multiselect(
        "Old/New", sorted(df["Old/New"].dropna().unique().tolist())
    )
    duration = st.multiselect(
        "Duration", sorted(df["Duration"].dropna().unique().tolist())
    )

    # Price
    p_min, p_max = float(df["Price"].min()), float(df["Price"].max())
    price_range = st.slider(
        "Price range",
        min_value=float(np.floor(p_min)),
        max_value=float(np.ceil(p_max)),
        value=(float(np.floor(p_min)), float(np.ceil(p_max))),
        step=1000.0,
    )
    use_winsor = st.checkbox(
        "Winsorise prices (1–99%)",
        value=False,
        help="Helps reduce the impact of extreme outliers.",
    )
    show_raw_clean = st.selectbox(
        "Data view", ["Cleaned only", "Raw + cleaned"]
    )  # retained for parity

# Apply filters
mask = (
           df["Date of Transfer"].between(
               pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
           )
       ) & (df["Price"].between(price_range[0], price_range[1]))
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
# Section router
# -------------------------
if nav == "KPIs":
    kpis.render(base, price_col)
elif nav == "Trends & Seasonality":
    trends.render(base, price_col)
elif nav == "Distributions":
    distributions.render(base, price_col)
elif nav == "Area Rankings":
    rankings.render(base, price_col)
elif nav == "Street / Postcode detail":
    detail.render(base, price_col)
elif nav == "Outliers & Data quality":
    quality.render(base, price_col)
elif nav == "Hedonic model":
    hedonic.render(base, price_col)
elif nav == "Repeat-sales":
    repeats.render(base, price_col)
elif nav == "Map (optional)":
    mapview.render(base, price_col)

st.caption(
    "Built by Usama Shahid with Love using Streamlit, Plotly, and statsmodels. Export tables via the '⋮' menu on each element."
)
