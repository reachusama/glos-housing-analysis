import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from ...charts import bar
from ...metrics import yoy_median
from ..components import kpi, section_help


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Data Preview")
    st.dataframe(base.head(20))

    st.subheader("Key Market Indicators")
    section_help(
        "The cards below summarise the filtered dataset. Use the sidebar to adjust filters."
    )

    latest_year = base["Year"].max()
    prev_year = latest_year - 1
    cur = base[base["Year"] == latest_year]
    prev = base[base["Year"] == prev_year]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Median Sale Price", f"£{base[price_col].median():,.0f}")
    with c2:
        kpi("Average Sale Price", f"£{base[price_col].mean():,.0f}")
    with c3:
        kpi("Total Transactions", f"{len(base):,}")
    with c4:
        yoy = yoy_median(base, price_col)
        kpi(
            "Median Price YoY Change",
            f"{yoy * 100:,.1f}%" if pd.notna(yoy) else "–",
            help="Change compared to the same period last year.",
        )
    with c5:
        mix = (
            base.groupby("Property Type")[price_col]
            .median()
            .sort_values(ascending=False)
        )
        top_type = mix.index[0] if len(mix) else "–"
        kpi("Most Expensive Property Type", top_type)

    # Only show time comparisons if >1 year present
    if base["Year"].nunique() > 1:
        st.markdown("### Price Trends Over Time")
        time_group = base.groupby(["Year", "Month Name"])[price_col].median().reset_index()
        time_group["Month Number"] = pd.to_datetime(time_group["Month Name"], format='%B').dt.month
        time_group = time_group.sort_values(["Year", "Month Number"])
        fig_price_time = px.line(
            time_group,
            x="Month Name",
            y=price_col,
            color="Year",
            markers=True,
            title="Median Sale Price by Month (Year Comparison)"
        )
        st.plotly_chart(fig_price_time, use_container_width=True)

        st.markdown("### Number of Transactions per Month")
        tx_group = base.groupby(["Year", "Month Name"]).size().reset_index(name="Transactions")
        tx_group["Month Number"] = pd.to_datetime(tx_group["Month Name"], format='%B').dt.month
        tx_group = tx_group.sort_values(["Year", "Month Number"])
        fig_tx = px.bar(
            tx_group,
            x="Month Name",
            y="Transactions",
            color="Year",
            barmode="group",
            title="Monthly Transactions by Year"
        )
        st.plotly_chart(fig_tx, use_container_width=True)

    # Consistent location selection for all location-based charts
    st.markdown("### Location-Based Analysis")
    location_type = st.selectbox("Choose Location Level", ["District", "County", "Town/City"])

    location_group = (
        base.groupby(location_type)[price_col]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    # Dynamic figure height: 30px per bar, minimum 500px, max 1200px
    def dynamic_height(n):
        return max(500, min(30 * n, 1200))

    # Horizontal bar chart for prices (all or top 20 if very large dataset)
    if len(location_group) > 25:
        top_prices = (
            location_group.sort_values("median", ascending=False)
            .head(20)
            .sort_values("median", ascending=True)
        )
    else:
        top_prices = location_group.sort_values("median", ascending=True)

    fig_location_price = px.bar(
        top_prices,
        x="median",
        y=location_type,
        orientation="h",
        title=f"{'Top 20 ' if len(location_group) > 25 else ''}{location_type}s by Median Price",
        labels={location_type: location_type, "median": "Median Price (£)"},
        height=dynamic_height(len(top_prices))
    )
    st.plotly_chart(fig_location_price, use_container_width=True)

    # Horizontal bar chart for transactions
    if len(location_group) > 25:
        top_tx = (
            location_group.sort_values("count", ascending=False)
            .head(20)
            .sort_values("count", ascending=True)
        )
    else:
        top_tx = location_group.sort_values("count", ascending=True)

    fig_location_tx = px.bar(
        top_tx,
        x="count",
        y=location_type,
        orientation="h",
        title=f"{'Top 20 ' if len(location_group) > 25 else ''}{location_type}s by Transaction Volume",
        labels={"count": "Number of Transactions"},
        height=dynamic_height(len(top_tx))
    )
    st.plotly_chart(fig_location_tx, use_container_width=True)

    # Top price growth (only if >1 year present)
    if base["Year"].nunique() > 1:
        # st.markdown(f"### Top {location_type}s by Price Growth")
        growth_group = base.groupby([location_type, "Year"])[price_col].median().reset_index()
        growth_pivot = growth_group.pivot(index=location_type, columns="Year", values=price_col).dropna()

        if latest_year in growth_pivot.columns and prev_year in growth_pivot.columns:
            growth_pivot["Growth %"] = (
                (growth_pivot[latest_year] - growth_pivot[prev_year]) /
                growth_pivot[prev_year]
            ) * 100

            if len(growth_pivot) > 25:
                top_growth = growth_pivot.sort_values("Growth %", ascending=False).head(20).reset_index()
            else:
                top_growth = growth_pivot.reset_index()

            fig_growth = px.bar(
                top_growth.sort_values("Growth %", ascending=True),
                x="Growth %",
                y=location_type,
                orientation="h",
                title=f"{'Top 20 ' if len(growth_pivot) > 25 else ''}{location_type}s by Yearly Price Growth",
                labels={"Growth %": "Price Growth (%)"},
                height=dynamic_height(len(top_growth))
            )
            st.plotly_chart(fig_growth, use_container_width=True)
