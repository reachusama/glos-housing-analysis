# housing/metrics.py — helper calcs used by sections
import numpy as np
import pandas as pd


def yoy_median(base: pd.DataFrame, price_col: str) -> float:
    latest_year = int(base["Year"].max())
    prev_year = latest_year - 1
    cur = base[base["Year"] == latest_year]
    prev = base[base["Year"] == prev_year]
    if len(cur) == 0 or len(prev) == 0:
        return float("nan")
    prev_med = prev[price_col].median()
    return (cur[price_col].median() / prev_med - 1.0) if prev_med > 0 else float("nan")
