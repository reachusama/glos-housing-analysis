## `housing/ui/sections/kpis.py`

import numpy as np
import pandas as pd
import streamlit as st

from ...charts import bar
from ...metrics import yoy_median
from ..components import kpi, section_help


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Key figures")
    section_help(
        "The cards summarise the filtered dataset. Use the sidebar to change filters."
    )

    latest_year = base["Year"].max()
    prev_year = latest_year - 1
    cur = base[base["Year"] == latest_year]
    prev = base[base["Year"] == prev_year]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Median price", f"£{base[price_col].median():,.0f}")
    with c2:
        kpi("Mean price", f"£{base[price_col].mean():,.0f}")
    with c3:
        kpi("Transactions", f"{len(base):,}")
    with c4:
        yoy = yoy_median(base, price_col)
        kpi(
            "YoY median",
            f"{yoy * 100:,.1f}%" if pd.notna(yoy) else "–",
            help="Median vs same period last year.",
        )
    with c5:
        mix = (
            base.groupby("Property Type")[price_col]
            .median()
            .sort_values(ascending=False)
            if len(base)
            else pd.Series(dtype=float)
        )
        top_type = mix.index[0] if len(mix) else "–"
        kpi(
            "Top median type", top_type, help="Property type with highest median price."
        )

    st.markdown("**Median price by property type**")
    g = (
        base.groupby("Property Type")[price_col]
        .median()
        .reset_index()
        .sort_values(price_col, ascending=False)
    )
    fig = bar(g, x="Property Type", y=price_col)
    st.plotly_chart(fig, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Freehold vs Leasehold**")
        g = (
            base["Duration"]
            .value_counts(dropna=False)
            .rename_axis("Duration")
            .reset_index(name="count")
        )
        st.plotly_chart(
            __import__("housing.charts", fromlist=["pie"]).pie(
                g, names="Duration", values="count"
            ),
            use_container_width=True,
        )
    with cB:
        st.markdown("**Old vs New**")
        g = (
            base["Old/New"]
            .value_counts(dropna=False)
            .rename_axis("Old/New")
            .reset_index(name="count")
        )
        st.plotly_chart(
            __import__("housing.charts", fromlist=["pie"]).pie(
                g, names="Old/New", values="count"
            ),
            use_container_width=True,
        )
