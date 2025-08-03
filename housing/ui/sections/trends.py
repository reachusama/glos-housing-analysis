import pandas as pd
import streamlit as st

from ...charts import heatmap, line


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Monthly trend")
    st.caption(
        "Rolling line helps you see the general direction without month-to-month noise."
    )

    metric = st.selectbox(
        "Metric",
        ["Median price", "Mean price", "Transactions"],
        help="Choose what to display on the line chart.",
    )
    window = st.slider("Rolling window (months)", 1, 24, 6)

    m = base.set_index("Date of Transfer").sort_index()
    monthly = (
        m.resample("MS")[price_col]
        .median()
        .to_frame("Median price")
        .join(m.resample("MS")[price_col].mean().to_frame("Mean price"))
        .join(m.resample("MS")[price_col].size().to_frame("Transactions"))
    )

    if len(monthly) == 0:
        st.info("No data for current filters.")
    else:
        import plotly.graph_objects as go

        fig = line(monthly.reset_index(), x="Date of Transfer", y=metric)
        y = monthly[metric].rolling(window).median()
        fig.add_trace(go.Scatter(x=monthly.index, y=y, name=f"{window}m rolling"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonal heatmap (Year × Month)")
    st.caption(
        "Rows are years, columns are months; darker cells indicate higher median prices."
    )
    pivot = (
        base.groupby(["Year", "Month"])[price_col]
        .median()
        .reset_index()
        .pivot(index="Year", columns="Month", values=price_col)
        .sort_index()
    )
    fig = heatmap(pivot, labels=dict(color="Median price"))
    fig.update_xaxes(title_text="Month")
    st.plotly_chart(fig, use_container_width=True)
