import pandas as pd
import streamlit as st

from ...charts import scatter


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Repeat-sales returns (same address)")
    g = base.sort_values("Date of Transfer").copy()
    g["ADDRESS_ID"] = g["ADDRESS_ID"]  # already computed; keep explicit for clarity
    counts = g["ADDRESS_ID"].value_counts()
    multi = g[g["ADDRESS_ID"].isin(counts[counts > 1].index)].copy()

    if len(multi) == 0:
        st.info("No repeated addresses found with current filters.")
    else:
        multi["next_price"] = multi.groupby("ADDRESS_ID")[price_col].shift(-1)
        multi["next_date"] = multi.groupby("ADDRESS_ID")["Date of Transfer"].shift(-1)
        rs = multi.dropna(subset=["next_price", "next_date"]).copy()
        rs["ret %"] = (rs["next_price"] / rs[price_col] - 1.0) * 100
        rs["months"] = (rs["next_date"] - rs["Date of Transfer"]).dt.days / 30.44
        st.caption(f"Pairs: {len(rs):,}")
        fig = scatter(
            rs,
            x="months",
            y="ret %",
            hover_data=["Street", "Postcode", "Property Type"],
        )
        fig.update_xaxes(title="Months between sales")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            rs[
                [
                    "Date of Transfer",
                    "next_date",
                    price_col,
                    "next_price",
                    "ret %",
                    "months",
                    "Street",
                    "Postcode",
                    "Property Type",
                ]
            ].sort_values("ret %", ascending=False),
            use_container_width=True,
        )
