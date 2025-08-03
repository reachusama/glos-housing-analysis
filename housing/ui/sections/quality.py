import pandas as pd
import streamlit as st

from ...charts import bar, pie


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Outliers & Data quality")

    lo, hi = base[price_col].quantile([0.01, 0.99])
    out = base[(base[price_col] < lo) | (base[price_col] > hi)].copy()
    st.caption(f"Flagged outside 1–99th pct: {len(out):,}")
    st.dataframe(
        out[
            [
                "Date of Transfer",
                "Price",
                "Property Type",
                "Street",
                "Postcode",
                "District",
                "Town/City",
            ]
        ].sort_values("Price", ascending=False),
        use_container_width=True,
    )

    st.subheader("Missingness heatmap")
    miss = base.isna().mean().sort_values(ascending=False)
    miss_df = miss.reset_index().rename(columns={"index": "Column", 0: "Missing %"})
    st.plotly_chart(bar(miss_df, x="Column", y="Missing %"), use_container_width=True)

    st.subheader("Entry types")
    if "PPD Category Type" in base and "Record Status" in base:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                pie(base, names="PPD Category Type"), use_container_width=True
            )
        with c2:
            st.plotly_chart(pie(base, names="Record Status"), use_container_width=True)
