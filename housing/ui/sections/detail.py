import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Street / Postcode Details")
    st.caption("Step 1: Enter a postcode (or part of it), then choose a street to see insights.")

    # Step 1: Postcode filter
    postcode_input = st.text_input("Enter postcode or partial postcode (case-insensitive)")
    filtered_by_postcode = base.copy()

    if postcode_input:
        filtered_by_postcode = base[
            base["Postcode"].str.contains(postcode_input, case=False, na=False)
        ]

        # Step 2: Multi-street selection
    selected_streets = []
    if not filtered_by_postcode.empty:
        unique_streets = sorted(filtered_by_postcode["Street"].dropna().unique())
        selected_streets = st.multiselect(
            "Select one or more streets (leave empty to include all streets)",
            options=unique_streets
        )
    else:
        st.info("Enter a postcode to see available streets.")

    # Final filtered dataset
    q = filtered_by_postcode
    if selected_streets:
        q = q[q["Street"].isin(selected_streets)]

    st.caption(f"Number of matching transactions: {len(q):,}")

    if len(q) > 0:
        # -----------------------
        # Key Insights Section
        # -----------------------
        st.markdown("### Key Insights for Selected Area")

        latest_year = q["Year"].max()
        prev_year = latest_year - 1
        yoy_growth = None

        if prev_year in q["Year"].unique():
            median_latest = q[q["Year"] == latest_year][price_col].median()
            median_prev = q[q["Year"] == prev_year][price_col].median()
            yoy_growth = ((median_latest - median_prev) / median_prev) * 100

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Median Price", f"£{q[price_col].median():,.0f}")
        with col2:
            st.metric("Average Price", f"£{q[price_col].mean():,.0f}")
        with col3:
            st.metric("Total Transactions", f"{len(q):,}")
        with col4:
            st.metric(
                "YoY Median Change",
                f"{yoy_growth:.1f}%" if yoy_growth is not None else "N/A"
            )
        with col5:
            top_type = q["Property Type"].mode().iloc[0] if not q["Property Type"].mode().empty else "N/A"
            st.metric("Most Common Type", top_type)

        # Highest transaction
        highest_tx = q.loc[q[price_col].idxmax()]
        st.caption(
            f"💡 Highest recorded sale: £{highest_tx[price_col]:,.0f} "
            f"({highest_tx['Property Type']} in {highest_tx['Street']}, {highest_tx['Postcode']})"
        )

        # -----------------------
        # Price trend over time
        # -----------------------
        st.markdown("### Price Trend Over Time")
        t = q.sort_values("Date of Transfer")
        fig_price_trend = px.line(
            t,
            x="Date of Transfer",
            y=price_col,
            markers=True,
            title="Property Sale Prices Over Time",
            labels={price_col: "Sale Price (£)", "Date of Transfer": "Date"},
            hover_data=["Street", "Postcode", "Property Type"],
        )
        st.plotly_chart(fig_price_trend, use_container_width=True)

        # -----------------------
        # Monthly median price trend
        # -----------------------
        st.markdown("### Monthly Median Price Trend")
        monthly_trend = (
            q.groupby(["Year", "Month Name", "Month"])[price_col]
            .median()
            .reset_index()
            .sort_values(["Year", "Month"])
        )
        fig_monthly = px.line(
            monthly_trend,
            x="Month Name",
            y=price_col,
            color="Year",
            markers=True,
            title="Median Sale Price by Month",
            labels={price_col: "Median Price (£)", "Month Name": "Month"},
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # -----------------------
        # Transactions per year
        # -----------------------
        st.markdown("### Transactions Per Year")
        yearly_tx = q.groupby("Year").size().reset_index(name="Transactions")
        fig_tx_year = px.bar(
            yearly_tx,
            x="Year",
            y="Transactions",
            title="Number of Transactions by Year",
            labels={"Transactions": "Number of Sales"},
            text="Transactions"
        )
        st.plotly_chart(fig_tx_year, use_container_width=True)

        # -----------------------
        # Data table
        # -----------------------
        st.markdown("### Matching Transactions")
        st.dataframe(
            t[
                [
                    "Date of Transfer",
                    "Price",
                    "Property Type",
                    "Duration",
                    "Old/New",
                    "PAON",
                    "SAON",
                    "Street",
                    "Postcode",
                    "District",
                    "Town/City",
                ]
            ].sort_values("Date of Transfer", ascending=False),
            use_container_width=True,
        )

    elif postcode_input:
        st.warning("No transactions found for this postcode.")
