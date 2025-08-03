import pandas as pd
import streamlit as st


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Log-price hedonic regression")
    st.caption(
        "Model: log(Price) ~ Property Type + Old/New + Duration + Year + Month + Outward"
    )
    st.info("Work in progress — enable statsmodels and add fit/summary when ready.")
