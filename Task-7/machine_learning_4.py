"""
Task 7 — Extention, Google Trends as a overlay vector
 - Candlestick chart
 - Google Trends overlay on top of the ensemble model from machine_learning_3.py
 - Future k-step forecast extension
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta
from pytrends.request import TrendReq
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import time
import hashlib

#Load Google Trends with retry and caching
def load_google_trends(keyword, start, end, max_retries=5):
    """
    Fellback to dummy trend (zeros) if Google Trends fetch fails after retries.
    """
    key = f"{keyword}_{start}_{end}"
    key_hash = hashlib.md5(key.encode()).hexdigest()
    cache_dir = "trend_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key_hash}.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["date"])

    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload([keyword], timeframe=f"{start} {end}")

    for attempt in range(max_retries):
        try:
            df = pytrends.interest_over_time()
            if df.empty:
                raise ValueError("Google Trends return empty dataset")

            df = df.reset_index()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            df.to_csv(cache_path, index=False)
            return df

        except Exception as e:
            print(f"Error. Google Trends failed (attempt {attempt+1}/{max_retries}): {e}")

            if attempt == max_retries - 1:
                print("Error. Using fallback dummy trend instead.")
                dummy_dates = pd.date_range(start=start, end=end, freq="D")
                fallback = pd.DataFrame({
                    "date": dummy_dates,
                    keyword: np.zeros(len(dummy_dates))
                })
                fallback.to_csv(cache_path, index=False)
                return fallback

            time.sleep(2 ** attempt)

    raise RuntimeError("Unexpected Trends failure")



#main plotting function
def main(ticker: str, ensemble_dir: str, pred_steps: int = 10, keyword: str = None):
    
    # Load price data
    df = yf.download(ticker, period="5y")
    df = df.reset_index()

    # Load ensemble output
    y_true = np.load(os.path.join(ensemble_dir, "y_true_raw.npy"))
    ensemble = np.load(os.path.join(ensemble_dir, "final_ensemble.npy"))

    future_pred = ensemble[-1]     # vector (pred_steps,)

    # Build future date extension
    last_date = df["Date"].iloc[-1]

    future_dates = [last_date + timedelta(days=i+1) for i in range(pred_steps)]
    pred_df = pd.DataFrame({
        "Date": future_dates,
        "PredictedClose": future_pred
    })

    # Prepare OHLC data for candlestick
    ohlc = df[["Date","Open","High","Low","Close"]].copy()
    ohlc["Date"] = mdates.date2num(ohlc["Date"].dt.to_pydatetime())
    ohlc = ohlc.values

    trends_df = None
    if keyword:
        trends_df = load_google_trends(
            keyword,
            df["Date"].min().strftime("%Y-%m-%d"),
            df["Date"].max().strftime("%Y-%m-%d")
        )
        trends_df = trends_df.reset_index()
        trends_df = trends_df.rename(columns={keyword: "Trend"})
        trends_df["DateNum"] = mdates.date2num(trends_df["date"].dt.to_pydatetime())

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # candlestick
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup="g", colordown="r")

    # Prediction overlay
    ax1.plot(
        pred_df["Date"],
        pred_df["PredictedClose"],
        color="gold",
        linewidth=3,
        label="Predicted Close"
    )

    # Google trend overlay
    if trends_df is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            trends_df["DateNum"],
            trends_df["Trend"],
            color="tab:blue",
            alpha=0.6,
            linewidth=2,
            label="Google Trend"
        )
        ax2.set_ylabel("Google Trend", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Formatting
    ax1.set_title(f"{ticker}: Forecast for the next {pred_steps} days + Google Trends Overlay")
    ax1.set_ylabel("Price (USD)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, linestyle="--", alpha=0.25)

    # Combined
    handles1, labels1 = ax1.get_legend_handles_labels()
    if trends_df is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    fig.tight_layout()
    # Zoom into last N days for better visibility
    ZOOM_DAYS = 365 #2 months for example, expand if needed to see bigger chart
    min_zoom_date = df["Date"].max() - pd.Timedelta(days=ZOOM_DAYS)
    ax1.set_xlim(min_zoom_date, pred_df["Date"].max())

    save_path = os.path.join(ensemble_dir, f"{ticker} w trend_overlay.png")
    fig.savefig(save_path)
    print("Chart saved in:", save_path)

    plt.show()

    return pred_df


if __name__ == "__main__":
    pred = main(
        ticker="AMZN",
        ensemble_dir="output_ensemble", # specify the correct output directory from machine_learning_3.py
        pred_steps=14,
        keyword="Amazon stock" #keyword for Google Trends
    )
    print(pred)