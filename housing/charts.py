# housing/charts.py — small, composable Plotly chart factories
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def bar(df: pd.DataFrame, x: str, y: str, **kwargs):
    return px.bar(df, x=x, y=y, **kwargs)


def line(df: pd.DataFrame, x: str, y: str, **kwargs):
    return px.line(df, x=x, y=y, **kwargs)


def pie(df: pd.DataFrame, names: str, values: str = None, **kwargs):
    return px.pie(df, names=names, values=values, **kwargs)


def hist(series, nbins=60, **kwargs):
    return px.histogram(x=series, nbins=nbins, **kwargs)


def violin(df: pd.DataFrame, x: str, y: str, **kwargs):
    return px.violin(df, x=x, y=y, **kwargs)


def box(df: pd.DataFrame, x: str, y: str, **kwargs):
    return px.box(df, x=x, y=y, **kwargs)


def heatmap(df_wide: pd.DataFrame, **kwargs):
    return px.imshow(df_wide, aspect="auto", **kwargs)


def scatter(df: pd.DataFrame, x: str, y: str, **kwargs):
    return px.scatter(df, x=x, y=y, **kwargs)
