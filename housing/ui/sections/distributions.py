import numpy as np
import pandas as pd
import streamlit as st

from ...charts import box, hist, violin


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Price distribution")
    st.caption("Use log scale to make the right tail easier to see.")

    log_scale = st.checkbox("Log-price", value=True)
    bins = st.slider("Bins", 10, 200, 60)
    s = np.log(base[price_col]) if log_scale else base[price_col]
    fig = hist(s, nbins=bins, labels={"x": "log(Price)" if log_scale else "Price"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Box/violin by property type")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(
            "Box plot: median line inside the box; points show suspected outliers."
        )
        st.plotly_chart(
            box(base, x="Property Type", y=price_col, points="suspectedoutliers"),
            use_container_width=True,
        )
    with c2:
        st.caption("Violin: distribution shape by type with a box overlay.")
        st.plotly_chart(
            violin(base, x="Property Type", y=price_col, box=True),
            use_container_width=True,
        )
