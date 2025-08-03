import pandas as pd
import plotly.express as px
import streamlit as st


def render(base: pd.DataFrame, price_col: str):
    st.subheader("Map of transactions (needs postcode centroids)")
    st.write(
        "Upload a CSV with postcode sector centroids having columns: `Postcode Sector` (e.g., 'GL52 3'), `lat`, `lon`."
    )
    cent = st.file_uploader("Upload centroids CSV", type=["csv"], key="centroids")
    if cent is None:
        st.info("No centroid file uploaded. Showing area bar chart instead.")
        agg = (
            base.groupby("Outward")[price_col]
            .median()
            .reset_index()
            .sort_values(price_col, ascending=False)
            .head(30)
        )
        fig = px.bar(agg, x="Outward", y=price_col, labels={price_col: "Median price"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        cent_df = pd.read_csv(cent)
        merged = (
            base.groupby("Postcode Sector")
            .agg(median_price=(price_col, "median"), volume=(price_col, "size"))
            .reset_index()
            .merge(cent_df, on="Postcode Sector", how="inner")
        )
        fig = px.scatter_mapbox(
            merged,
            lat="lat",
            lon="lon",
            size="volume",
            color="median_price",
            hover_name="Postcode Sector",
            hover_data={"median_price": ":,.0f", "volume": True},
            zoom=8,
            height=700,
        )
        fig.update_layout(
            mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
