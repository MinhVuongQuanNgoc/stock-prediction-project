"""
google_trends.py

Utilities to fetch Google Trends for a keyword (via pytrends),
"""

from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime

def get_google_trends(keyword: str, start_date: str, end_date: str, geo: str = "") -> pd.DataFrame:
    """
    Fetch Google Trends 'interest_over_time' for `keyword` between start_date and end_date.
    Returns DataFrame with columns: ['date' (datetime), 'GoogleTrend' (int or float)]
    NOTE: pytrends sometimes returns weekly samples; we forward-fill later.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    timeframe = f"{start_date} {end_date}"
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        # return empty DataFrame with date index between start/end
        idx = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({"date": idx, "GoogleTrend": [0]*len(idx)}).set_index("date")

    df = df.reset_index()  # date column
    # the pytrends results often contain an 'isPartial' column; drop it if present
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df.rename(columns={keyword: "GoogleTrend"}, inplace=True)
    # normalize date column to midnight, ensure datetime index
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")[["GoogleTrend"]]
    # sometimes Google returns integer 0-100; keep as-is (scaling done later)
    return df

def aggregate_and_fill_trends(trend_df: pd.DataFrame, start_date: str, end_date: str, method: str = "ffill") -> pd.DataFrame:
    idx = pd.date_range(start=start_date, end=end_date, freq='D')
    df = trend_df.reindex(idx)  # may introduce NaNs
    if method == "ffill":
        df = df.fillna(method="ffill").fillna(0)
    elif method == "zero":
        df = df.fillna(0)
    elif method == "interpolate":
        df = df.interpolate().fillna(0)
    else:
        raise ValueError("Unknown fill method")
    df.index.name = "Date"
    return df

def merge_trends_into_prices(price_df: pd.DataFrame, trend_daily_df: pd.DataFrame,
                             price_date_col: str = "Date", how: str = "left") -> pd.DataFrame:
    df = price_df.copy()
    # ensure price_df has datetime index
    if price_date_col in df.columns:
        df[price_date_col] = pd.to_datetime(df[price_date_col])
        df.set_index(price_date_col, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("price_df must have a Date column or datetime index")

    # normalize both to midnight
    df.index = df.index.normalize()
    trend_daily_df.index = pd.to_datetime(trend_daily_df.index).normalize()

    merged = df.merge(trend_daily_df, left_index=True, right_index=True, how=how)
    # After merge, there may be NaNs where trend doesn't exist (e.g., weekends vs weekly sampling).
    # Fill them with forward fill to propagate last observed trend value.
    merged["GoogleTrend"] = merged["GoogleTrend"].fillna(method="ffill").fillna(0)
    # reset index to preserve original callers expecting Date column optionally
    merged = merged.reset_index().rename(columns={"index": "Date"})
    # keep Date as datetime column
    merged["Date"] = pd.to_datetime(merged["Date"])
    return merged
