import pandas as pd
import plotly.express as px
import streamlit as st


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Street / Postcode detail")
    st.caption("Type a street name or postcode fragment to see a detailed view.")
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
        t = q.sort_values("Date of Transfer")
        fig = px.line(
            t,
            x="Date of Transfer",
            y=price_col,
            markers=True,
            hover_data=["Street", "Postcode", "Property Type"],
        )
        st.plotly_chart(fig, use_container_width=True)
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
    else:
        st.info("Enter a query to see results.")
