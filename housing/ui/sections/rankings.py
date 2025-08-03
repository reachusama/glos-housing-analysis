import pandas as pd
import streamlit as st

from ...charts import bar


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Rankings by area")
    st.caption(
        "Sorts areas by latest-year median price; also shows year-over-year change."
    )

    level = st.selectbox(
        "Group by", ["District", "Town/City", "Outward", "Postcode Sector"]
    )
    g = (
        base.groupby([level, "Year"])
        .agg(median_price=(price_col, "median"), volume=(price_col, "size"))
        .reset_index()
    )
    latest = g[g["Year"] == g["Year"].max()].copy()
    prev = g[g["Year"] == g["Year"].max() - 1][[level, "median_price"]].rename(
        columns={"median_price": "median_prev"}
    )
    latest = latest.merge(prev, on=level, how="left")
    latest["YoY %"] = (latest["median_price"] / latest["median_prev"] - 1.0) * 100

    st.dataframe(
        latest.sort_values("median_price", ascending=False), use_container_width=True
    )

    st.markdown("**Top movers (YoY median %)**")
    movers = (
        latest.dropna(subset=["YoY %"]).sort_values("YoY %", ascending=False).head(20)
    )
    st.plotly_chart(bar(movers, x=level, y="YoY %"), use_container_width=True)
